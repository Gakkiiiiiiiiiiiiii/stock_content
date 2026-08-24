"""Pipeline Replay + Idempotency 三概念 API 测试（详细修改方案 §4 P0-2/P0-3）。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from stock_content.api.dependencies import build_application
from stock_content.api.main import create_app


def _ingest_options() -> dict:
    return {
        "metadata": {"title": "replay fixture"},
        "transcript": "股票600000基本面良好。",
        "offline_fixture": True,
        "asr_model": "faster-whisper",
        "asr_model_version": "large-v3",
        "quant_market_snapshot_ids": ["market-snap-1"],
    }


def test_ingest_produces_content_snapshot_and_replay(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))

    enqueue = client.post(
        "/api/v1/videos/bilibili/ingest", json={"bv_id": "BV1replay", "options": _ingest_options()}
    )
    assert enqueue.status_code == 200
    body = enqueue.json()
    # P0-3：三个概念分开返回，禁止混用。
    assert body["source_identity_hash"]
    assert body["source_content_hash"] is None  # 源内容哈希在 resolve 后才可得
    assert body["request_idempotency_key"] is None

    application.process_next("replay-test")
    task = client.get(f"/api/v1/tasks/{body['task_id']}").json()
    assert task["status"] == "SUCCEEDED"
    snapshot_id = task["result"]["content_snapshot_id"]
    assert snapshot_id.startswith("cs-")

    fetched = client.get(f"/api/v1/content-snapshots/{snapshot_id}")
    assert fetched.status_code == 200
    data = fetched.json()["data"]
    assert data["content_snapshot_id"] == snapshot_id
    assert data["source_type"] == "bilibili"
    assert data["artifact_ids"]
    assert data["quant_market_snapshot_ids"] == ["market-snap-1"]

    replay = client.post(f"/api/v1/content-snapshots/{snapshot_id}/replay")
    assert replay.status_code == 200
    assert replay.json()["identity_match"] is True

    video_id = task["result"]["video_id"]
    listed = client.get(f"/api/v1/videos/{video_id}/snapshots")
    assert listed.status_code == 200
    assert [item["content_snapshot_id"] for item in listed.json()["items"]] == [snapshot_id]


def test_same_content_different_model_yields_new_snapshot(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))

    # 同一 source 使用不同 ASR 模型重复处理：源内容相同，快照身份必须不同。
    ids = []
    for asr_model in ["faster-whisper", "paraformer"]:
        options = _ingest_options()
        options["asr_model"] = asr_model
        enqueue = client.post(
            "/api/v1/videos/bilibili/ingest",
            json={"bv_id": "BV1model", "options": options},
        )
        application.process_next("replay-test")
        task = client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()
        assert task["status"] == "SUCCEEDED"
        ids.append(task["result"]["content_snapshot_id"])
    assert ids[0] != ids[1]

    # 同一 source 的历次快照版本均可查询。
    video_id = task["result"]["video_id"]
    listed = client.get(f"/api/v1/videos/{video_id}/snapshots").json()["items"]
    assert {item["content_snapshot_id"] for item in listed} == set(ids)


def test_snapshot_not_found_returns_404(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    assert client.get("/api/v1/content-snapshots/cs-missing").status_code == 404
    assert client.post("/api/v1/content-snapshots/cs-missing/replay").status_code == 404


# ---- P0 C-02：核心 pipeline 必须在对应 Stage 完成时立即登记 typed Artifact ----


def _run_and_capture_registry(tmp_path, monkeypatch) -> tuple[dict, "TestClient"]:
    from stock_content.application.stages import SnapshotRecordingStage

    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    captured: dict = {}
    original = SnapshotRecordingStage.execute

    def spy(self, context):
        result = original(self, context)
        captured["registry"] = context.artifacts
        return result

    monkeypatch.setattr(SnapshotRecordingStage, "execute", spy)
    enqueue = client.post(
        "/api/v1/videos/bilibili/ingest",
        json={
            "bv_id": "BV1artifacts",
            "options": {
                "metadata": {"title": "artifact fixture"},
                "transcript": "股票600000基本面良好。",
                "offline_fixture": True,
                "asr_model": "faster-whisper",
                "asr_model_version": "large-v3",
            },
        },
    )
    application.process_next("artifact-test")
    task = client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()
    assert task["status"] == "SUCCEEDED"
    return {"registry": captured["registry"], "task": task}, client


def test_core_pipeline_emits_source_artifact(tmp_path, monkeypatch):
    captured, _ = _run_and_capture_registry(tmp_path, monkeypatch)
    source = captured["registry"].source
    assert source is not None
    assert source.source_type == "bilibili"
    assert source.source_content_hash, "获得真实内容后 source_content_hash 必须补齐"


def test_core_pipeline_emits_transcript_artifact(tmp_path, monkeypatch):
    captured, _ = _run_and_capture_registry(tmp_path, monkeypatch)
    transcript = captured["registry"].transcript
    assert transcript is not None
    assert transcript.segments
    assert transcript.asr_model == "faster-whisper"  # ASR model/version 进入 lineage
    assert transcript.asr_model_version == "large-v3"


def test_core_pipeline_emits_knowledge_artifact(tmp_path, monkeypatch):
    captured, _ = _run_and_capture_registry(tmp_path, monkeypatch)
    knowledge = captured["registry"].knowledge
    assert knowledge is not None
    assert knowledge.knowledge_units, "knowledge/claim ref 必须进入 lineage"


def test_core_pipeline_emits_summary_artifact(tmp_path, monkeypatch):
    captured, _ = _run_and_capture_registry(tmp_path, monkeypatch)
    summary = captured["registry"].summary
    assert summary is not None
    assert summary.core_summary


def test_summary_references_knowledge_artifact(tmp_path, monkeypatch):
    captured, _ = _run_and_capture_registry(tmp_path, monkeypatch)
    registry = captured["registry"]
    assert registry.summary.knowledge_artifact_id == registry.knowledge.artifact_id


def test_replay_uses_artifact_lineage(tmp_path, monkeypatch):
    """P0 C-04：Snapshot identity 使用 pipeline 实际 Artifact IDs，不引用不存在的 Artifact。"""
    captured, client = _run_and_capture_registry(tmp_path, monkeypatch)
    registry = captured["registry"]
    snapshot_id = captured["task"]["result"]["content_snapshot_id"]
    data = client.get(f"/api/v1/content-snapshots/{snapshot_id}").json()["data"]
    assert data["artifact_ids"] == registry.artifact_ids()
    # 每个被引用的 artifact id 都必须真实存在于 registry。
    known = {artifact.artifact_id for artifact in registry.artifacts()}
    assert set(data["artifact_ids"].values()) <= known


def test_reprocess_fixture_is_exact_and_closes_task(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    options = {
        **_ingest_options(),
        "available_from": "2025-01-02T03:04:05+00:00",
    }
    enqueue = client.post(
        "/api/v1/videos/bilibili/ingest", json={"bv_id": "BV1golden", "options": options}
    )
    application.process_next("replay-golden")
    snapshot_id = client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()["result"]["content_snapshot_id"]

    replay = client.post(
        f"/api/v1/content-snapshots/{snapshot_id}/replay", json={"mode": "REPROCESS"}
    )
    assert replay.status_code == 200
    body = replay.json()
    assert "differences" in body
    assert not body.get("differences")
    replay_task = client.get(f"/api/v1/tasks/{body['replay_id']}")
    assert replay_task.status_code == 200
    assert replay_task.json()["status"] == "SUCCEEDED"


def test_migration_replay_creates_child_snapshot(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    enqueue = client.post(
        "/api/v1/videos/bilibili/ingest", json={"bv_id": "BV1migration", "options": _ingest_options()}
    )
    application.process_next("replay-migration")
    snapshot_id = client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()["result"]["content_snapshot_id"]
    replay = client.post(
        f"/api/v1/content-snapshots/{snapshot_id}/replay",
        json={"mode": "MIGRATION_REPLAY", "pipeline_version": "pipeline.v4"},
    )
    assert replay.status_code == 200
    candidate = client.get(f"/api/v1/content-snapshots/{replay.json()['candidate_snapshot_id']}").json()["data"]
    assert candidate["snapshot_kind"] == "MIGRATION"
    assert candidate["parent_snapshot_id"] == snapshot_id
    assert candidate["supersedes_snapshot_id"] == snapshot_id
    assert candidate["pipeline_version"] == "pipeline.v4"


def test_task_specific_options_do_not_change_snapshot_identity(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    ids = []
    for key, trace in (("request-a", "trace-a"), ("request-b", "trace-b")):
        options = {
            **_ingest_options(),
            "available_from": "2025-01-02T03:04:05+00:00",
            "idempotency_key": key,
            "trace_id": trace,
        }
        enqueue = client.post(
            "/api/v1/videos/bilibili/ingest", json={"bv_id": "BV1identity", "options": options}
        )
        application.process_next("replay-identity")
        ids.append(client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()["result"]["content_snapshot_id"])
    assert ids[0] == ids[1]
    configuration = client.get(f"/api/v1/content-snapshots/{ids[0]}").json()["data"]["configuration"]
    assert "pipeline_options" not in configuration


def test_reprocess_difference_is_structured_and_task_fails(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    enqueue = client.post(
        "/api/v1/videos/bilibili/ingest", json={"bv_id": "BV1different", "options": _ingest_options()}
    )
    application.process_next("replay-different")
    snapshot_id = client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()["result"]["content_snapshot_id"]
    replay = client.post(
        f"/api/v1/content-snapshots/{snapshot_id}/replay",
        json={"mode": "REPROCESS", "overrides": {"transcript": "different fixture"}},
    )
    assert replay.status_code == 409
    body = replay.json()["detail"]
    assert body["error"] == "REPLAY_NONDETERMINISTIC"
    assert body["differences"]
    assert client.get(f"/api/v1/tasks/{body['replay_id']}").json()["status"] == "FAILED"


def test_reprocess_pipeline_failure_closes_created_task(tmp_path, monkeypatch):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    enqueue = client.post(
        "/api/v1/videos/bilibili/ingest", json={"bv_id": "BV1failure", "options": _ingest_options()}
    )
    application.process_next("replay-failure")
    snapshot_id = client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()["result"]["content_snapshot_id"]

    def fail_pipeline(_context):
        raise RuntimeError("fixture pipeline failed")

    monkeypatch.setattr(application._pipeline, "process", fail_pipeline)
    replay = client.post(
        f"/api/v1/content-snapshots/{snapshot_id}/replay", json={"mode": "REPROCESS"}
    )
    assert replay.status_code == 500
    detail = replay.json()["detail"]
    assert detail["error"] == "REPLAY_FAILED"
    assert client.get(f"/api/v1/tasks/{detail['replay_id']}").json()["status"] == "FAILED"

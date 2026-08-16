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

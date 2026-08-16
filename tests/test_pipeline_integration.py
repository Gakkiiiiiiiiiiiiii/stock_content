from datetime import UTC, datetime, timedelta

import pytest

from stock_content.api.dependencies import build_application


def test_ingest_worker_persists_searchable_knowledge_and_factor_signal(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'content.db'}"
    application = build_application(database_url, enable_qdrant=False)
    as_of = datetime.now(UTC).replace(microsecond=0)
    task = application.enqueue(
        "bilibili",
        "BV1fixture",
        {
            "metadata": {"title": "新能源行业分析", "author": "researcher", "duration_seconds": 42},
            "transcript": "宁德时代300750业绩增长，毛利率改善。这是明确利好，但仍需关注价格战风险。",
            "as_of": as_of.isoformat(),
            "offline_fixture": True,
        },
    )

    result = application.process_next("test-worker")

    assert result is not None
    assert result["status"] == "SUCCEEDED"
    assert result["chapter_count"] == 1
    assert result["knowledge_count"] == 2
    persisted = application.get_task(task["task_id"])
    assert persisted["status"] == "SUCCEEDED"
    assert persisted["stage"] == "completed"
    video = application.get_video(result["video_id"])
    assert video["title"] == "新能源行业分析"
    assert len(video["segments"]) == 1

    search = application.search_knowledge("毛利率", {"ticker": "300750"}, 10)
    assert len(search) == 1
    assert search[0]["support_status"] == "SOURCE_SUPPORTED"
    signals = application.factor_signals(
        ["300750"], as_of - timedelta(seconds=1), as_of + timedelta(seconds=1), "SOURCE_SUPPORTED"
    )
    assert len(signals) == 2
    assert all(item["symbol"] == "300750" for item in signals)


def test_snapshot_store_healthy_yields_content_snapshot_id(tmp_path):
    """P0 C-03：snapshot store 正常 → content_snapshot_id != null。"""
    database_url = f"sqlite:///{tmp_path / 'content.db'}"
    application = build_application(database_url, enable_qdrant=False)
    application.enqueue(
        "bilibili",
        "BV1snapshotok",
        {
            "metadata": {"title": "快照正常路径", "author": "tester", "duration_seconds": 10},
            "transcript": "宁德时代300750业绩增长。这是明确利好。",
            "offline_fixture": True,
        },
    )

    result = application.process_next("test-worker")

    assert result is not None
    assert result["status"] == "SUCCEEDED"
    assert result.get("content_snapshot_id"), "snapshot store 正常时必须产出 content_snapshot_id"
    snapshot = application.get_content_snapshot(result["content_snapshot_id"])
    assert snapshot is not None


def test_snapshot_store_failure_never_silently_succeeds(tmp_path, monkeypatch):
    """P0 C-03：snapshot store 抛错 → 不允许普通 SUCCEEDED，必须明确失败。"""
    database_url = f"sqlite:///{tmp_path / 'content.db'}"
    application = build_application(database_url, enable_qdrant=False)

    def _broken_save(*args, **kwargs):
        raise RuntimeError("snapshot store unavailable")

    monkeypatch.setattr(application._snapshots._store, "save", _broken_save)  # noqa: SLF001

    task = application.enqueue(
        "bilibili",
        "BV1snapshotfail",
        {
            "metadata": {"title": "快照失败路径", "author": "tester", "duration_seconds": 10},
            "transcript": "宁德时代300750业绩增长。这是明确利好。",
            "offline_fixture": True,
        },
    )

    result = application.process_next("test-worker")

    assert result is not None
    assert result["status"] == "FAILED"
    assert result["status"] != "SUCCEEDED", "ContentSnapshot 失败后禁止静默成功"
    assert "CONTENT_SNAPSHOT_PERSIST_FAILED" in result["error"] or "ContentSnapshotPersistError" in result["error"]
    persisted = application.get_task(task["task_id"])
    assert persisted["status"] != "SUCCEEDED"
    assert "CONTENT_SNAPSHOT_PERSIST_FAILED" in (persisted.get("error") or "") or "ContentSnapshotPersistError" in (
        persisted.get("error") or ""
    )


@pytest.mark.parametrize("content_snapshot_id", [None, ""])
def test_downstream_v3_signal_rejects_missing_snapshot(content_snapshot_id):
    """P0 C-03：downstream v3 signal 不接受无 snapshot 的 normal signal。"""
    from stock_content.domain.signal_contract import is_normal_v3_signal, upgrade_signal_v3

    signal = upgrade_signal_v3(
        {
            "knowledge_uid": "k-1",
            "sentiment": "BULLISH",
            "content_snapshot_id": content_snapshot_id,
        }
    )
    assert signal["signal_status"] == "DEGRADED_NO_SNAPSHOT"
    assert is_normal_v3_signal(signal) is False

    normal = upgrade_signal_v3(
        {"knowledge_uid": "k-2", "sentiment": "BULLISH", "content_snapshot_id": "cs-1"}
    )
    assert normal["signal_status"] == "NORMAL"
    assert is_normal_v3_signal(normal) is True


def test_task_is_persistent_across_application_instances(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'content.db'}"
    first = build_application(database_url, enable_qdrant=False)
    task = first.enqueue(
        "bilibili", "BV1persistent", {"transcript": "这是一段足够长的测试文本。", "offline_fixture": True}
    )

    second = build_application(database_url, enable_qdrant=False)

    assert second.get_task(task["task_id"])["source_ref"] == "BV1persistent"

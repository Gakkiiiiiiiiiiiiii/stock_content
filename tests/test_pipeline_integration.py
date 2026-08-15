from datetime import UTC, datetime, timedelta

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


def test_task_is_persistent_across_application_instances(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'content.db'}"
    first = build_application(database_url, enable_qdrant=False)
    task = first.enqueue(
        "bilibili", "BV1persistent", {"transcript": "这是一段足够长的测试文本。", "offline_fixture": True}
    )

    second = build_application(database_url, enable_qdrant=False)

    assert second.get_task(task["task_id"])["source_ref"] == "BV1persistent"

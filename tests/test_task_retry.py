from stock_content.api.dependencies import build_application


def test_worker_records_stage_and_requeues_retryable_failure(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    task = application.enqueue(
        "bilibili",
        "BV1retry",
        {"metadata": {"title": "empty"}, "transcript": ""},
    )

    result = application.process_next("test-worker")
    persisted = application.get_task(task["task_id"])

    assert result["status"] == "FAILED"
    assert result["stage"] == "asr"
    assert persisted["status"] == "PENDING"
    assert persisted["retry_count"] == 1
    assert persisted["error"].startswith("ValueError")

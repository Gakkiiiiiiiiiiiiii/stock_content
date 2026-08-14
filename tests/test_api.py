from fastapi.testclient import TestClient

from stock_content.api.dependencies import build_application
from stock_content.api.main import create_app


def test_api_vertical_slice(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    response = client.post(
        "/api/v1/videos/bilibili/ingest",
        json={
            "bv_id": "BV1api",
            "options": {
                "metadata": {"title": "API fixture"},
                "transcript": "股票600000的利润增长是利好。",
            },
        },
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    application.process_next("api-test")
    task = client.get(f"/api/v1/tasks/{task_id}")
    search = client.post("/api/v1/knowledge/search", json={"query": "利润", "limit": 5})
    video_id = task.json()["result"]["video_id"]
    videos = client.get("/api/v1/videos")
    segments = client.get(f"/api/v1/videos/{video_id}/segments")
    chapters = client.get(f"/api/v1/videos/{video_id}/chapters")
    summary = client.get(f"/api/v1/videos/{video_id}/summary")
    knowledge = client.get(f"/api/v1/videos/{video_id}/knowledge")
    unit = client.get(f"/api/v1/knowledge/{search.json()['items'][0]['knowledge_uid']}")
    signals = client.post(
        "/internal/v1/factor-signals",
        json={"symbols": ["600000"], "start": "2026-01-01T00:00:00Z", "end": "2026-12-31T00:00:00Z"},
    )

    assert task.json()["status"] == "SUCCEEDED"
    assert search.json()["items"][0]["ticker"] == "600000"
    assert search.json()["contract_version"] == "content.v1"
    assert videos.json()["items"][0]["video_id"] == video_id
    assert segments.json()["items"]
    assert chapters.json()["items"]
    assert summary.json()["data"]["video_id"] == video_id
    assert knowledge.json()["items"][0]["video_id"] == video_id
    assert unit.json()["data"]["knowledge_uid"] == search.json()["items"][0]["knowledge_uid"]
    assert signals.json()["contract_version"] == "content-factor-signal.v2"
    assert signals.json()["items"][0]["knowledge_uid"] == search.json()["items"][0]["knowledge_uid"]
    assert signals.json()["items"][0]["truth_status"] == "NOT_CHECKED"
    assert signals.json()["items"][0]["available_from"]

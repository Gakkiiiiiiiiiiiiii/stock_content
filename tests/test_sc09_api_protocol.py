from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from stock_content.api.main import create_app
from stock_content.api.security import ServiceAuthorizer


class _Application:
    def __init__(self) -> None:
        self.calls = 0
        self.replays = []
        self._tasks = type("Tasks", (), {"_sessions": None, "claim_pending": lambda *_args, **_kw: None})()
        self._knowledge_bundle_service = type("Bundles", (), {"create": lambda *_args: {}, "get": lambda *_args: {}})()

    def enqueue_ingestion(self, command):
        self.calls += 1
        return {"task_id": f"task-{self.calls}", "status": "PENDING", "source": command.source_type}

    def get_task(self, task_id):
        return {"task_id": task_id, "status": "PENDING"}

    def replay_content_snapshot(self, content_snapshot_id, *, mode=None, pipeline_version=None, overrides=None):
        self.replays.append((content_snapshot_id, mode, pipeline_version, overrides))
        return {"content_snapshot_id": content_snapshot_id, "identity_match": True}

    def create_knowledge_bundle(self, _request):
        return {"bundle_id": "bundle-1"}

    def get_knowledge_bundle(self, bundle_id):
        return {"bundle_id": bundle_id}


def _client(tmp_path: Path) -> TestClient:
    current = tmp_path / "current"
    previous = tmp_path / "previous"
    current.write_text("current-token\n", encoding="utf-8")
    previous.write_text("previous-token\n", encoding="utf-8")
    return TestClient(create_app(_Application(), authorizer=ServiceAuthorizer((current, previous), ("stock_agent",))))


def _headers(token="current-token", caller="stock_agent", **extra):
    return {
        "Authorization": f"Bearer {token}",
        "x-caller-service": caller,
        "x-trace-id": "trace-09",
        **extra,
    }


def test_private_ingestion_requires_rotating_bearer_and_allowed_caller(tmp_path):
    client = _client(tmp_path)
    body = {"source_type": "bilibili", "source_ref": "BV1abc", "part": 1, "transcript_policy": "subtitle_first"}
    for headers, expected in (({}, 401), (_headers("wrong"), 401), (_headers(caller="forged"), 403)):
        response = client.post("/v1/content/ingestions", json=body, headers=headers)
        assert response.status_code == expected
        assert set(response.json()["error"]) >= {"code", "message", "retryable", "trace_id"}
    assert client.post("/v1/content/ingestions", json=body, headers=_headers("previous-token")).status_code == 200


def test_replay_and_task_readback_require_the_service_acl(tmp_path):
    client = _client(tmp_path)
    replay_path = "/api/v1/content-snapshots/cs-private/replay"
    task_path = "/api/v1/tasks/task-private"

    assert client.post(
        replay_path, json={"mode": "MIGRATION_REPLAY", "pipeline_version": "pipeline.v4.043.audit"}
    ).status_code == 401
    assert client.get(task_path).status_code == 401

    assert client.get(task_path, headers=_headers()).status_code == 200
    response = client.post(
        replay_path,
        json={"mode": "MIGRATION_REPLAY", "pipeline_version": "pipeline.v4.043.audit"},
        headers=_headers(),
    )
    assert response.status_code == 200
    assert response.json()["identity_match"] is True


def test_trace_content_type_validation_and_redacted_exception(tmp_path):
    client = _client(tmp_path)
    headers = _headers(**{"x-trace-id": "bad trace"})
    response = client.post(
        "/v1/content/ingestions", json={"source_type": "bilibili", "source_ref": "BV1abc"}, headers=headers
    )
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_TRACE_ID"
    response = client.post("/v1/content/ingestions", content="{}", headers=_headers())
    assert response.status_code == 415 and response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_bundle_requires_stock_agent_and_contract_checksum_is_unchanged(tmp_path):
    client = _client(tmp_path)
    body = {
        "content_snapshot_id": "cs_1",
        "query": "q",
        "symbol": "600000",
        "business_as_of": datetime(2026, 9, 6, tzinfo=UTC).isoformat(),
        "knowledge_as_of": datetime(2026, 9, 6, tzinfo=UTC).isoformat(),
        "availability_as_of": datetime(2026, 9, 6, tzinfo=UTC).isoformat(),
        "minimum_support_status": "SOURCE_SUPPORTED",
        "max_items": 1,
    }
    assert client.post("/v1/content/knowledge-bundles", json=body, headers=_headers()).status_code == 200
    assert client.get("/v1/content/knowledge-bundles/bundle-1", headers=_headers(caller="other")).status_code == 403
    contract = Path(__file__).parents[1] / "contracts" / "content-knowledge-bundle.v1.json"
    assert "sha256:" + hashlib.sha256(contract.read_bytes()).hexdigest().upper() == (
        "sha256:EBFD13B78622C3846890438A4FB3CB858278F571FDAB247CDD72EF18CA211621"
    )


def test_readiness_is_truthful_and_qdrant_cannot_block_bundle(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setenv("CONTENT_VIDEO_WORKER_HEARTBEAT_FILE", str(tmp_path / "missing-heartbeat"))
    video = client.get("/health/video-ingestion-ready")
    assert video.status_code == 503
    assert video.json()["components"]["yt_dlp"]["ready"] is False
    bundle = client.get("/health/knowledge-bundle-ready")
    assert bundle.status_code == 503  # schema/auth authority is deliberately incomplete in this fixture
    assert bundle.json()["contract"] == "content-knowledge-bundle.v1"
    assert bundle.json()["canonicalization_version"] == "content-bundle-c14n-v1"
    assert bundle.json()["contract_checksum"] == (
        "sha256:EBFD13B78622C3846890438A4FB3CB858278F571FDAB247CDD72EF18CA211621"
    )
    assert "qdrant" not in json.dumps(bundle.json()).lower()

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from stock_content.adapters.postgres.models import ClaimVerificationJobRow
from stock_content.api.dependencies import build_application
from stock_content.api.main import create_app
from stock_content.domain.lineage import default_code_sha
from stock_content.domain.models import KnowledgeUnit


def test_ingest_trace_is_persisted_and_lineage_resources_are_queryable(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'lineage.db'}"
    application = build_application(database_url, enable_qdrant=False)
    client = TestClient(create_app(application))
    response = client.post(
        "/api/v1/videos/bilibili/ingest",
        headers={"X-Trace-Id": "trace-e2e", "X-Decision-Id": "decision-e2e"},
        json={
            "bv_id": "BVlineage",
            "options": {
                "metadata": {"title": "lineage"},
                "transcript": "600000 收入增长。",
                "offline_fixture": True,
            },
        },
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    assert response.headers["x-trace-id"] == "trace-e2e"
    assert response.headers["x-decision-id"] == "decision-e2e"

    result = application.process_next("lineage-worker")
    assert result and result["status"] == "SUCCEEDED"
    task = client.get(f"/api/v1/tasks/{task_id}").json()
    assert task["trace_id"] == "trace-e2e"
    snapshot_id = task["result"]["content_snapshot_id"]
    snapshot = client.get(f"/api/v1/content-snapshots/{snapshot_id}").json()["data"]
    snapshot_lineage = client.get(f"/api/v1/content-snapshots/{snapshot_id}/lineage")
    assert snapshot_lineage.status_code == 200
    assert [item["slot"] for item in snapshot_lineage.json()["data"]["artifacts"]] == sorted(
        snapshot["artifact_ids"]
    )

    source_id = snapshot["artifact_ids"]["source"]
    artifact = client.get(f"/api/v1/artifacts/{source_id}")
    assert artifact.status_code == 200
    assert client.get(f"/api/v1/artifacts/{source_id}/lineage").status_code == 200
    assert client.get("/api/v1/artifacts/missing").status_code == 404

    claims_artifact = application.get_artifact(snapshot["artifact_ids"]["claims"])
    claim_id = str(claims_artifact["claims"][0])
    evidence = client.get(f"/api/v1/claims/{claim_id}/evidence")
    assert evidence.status_code == 200
    assert evidence.json()["items"]
    assert client.get(f"/api/v1/claims/{claim_id}/verifications").status_code == 200
    assert client.get("/api/v1/claims/missing/evidence").status_code == 404

    with application._claim_repository._sessions() as session:  # noqa: SLF001 - integration assertion
        jobs = session.query(ClaimVerificationJobRow).filter_by(claim_id=claim_id).all()
    assert all(job.trace_id == "trace-e2e" for job in jobs)


def test_production_startup_rejects_unknown_code_sha_before_database(monkeypatch):
    from stock_content.api import dependencies

    monkeypatch.setenv("CONTENT_ENV", "production")
    monkeypatch.delenv("CONTENT_GIT_COMMIT", raising=False)
    called = False

    class UnexpectedDatabase:
        def __init__(self, *args, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr(dependencies, "Database", UnexpectedDatabase)
    try:
        dependencies.build_application("sqlite:///should-not-open.db", enable_qdrant=False)
    except ValueError as exc:
        assert "CONTENT_GIT_COMMIT" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("production startup accepted unknown code SHA")
    assert called is False


def test_environment_fallback_also_enforces_production_code_sha(monkeypatch):
    monkeypatch.delenv("CONTENT_ENV", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("CONTENT_GIT_COMMIT", raising=False)
    try:
        default_code_sha()
    except ValueError as exc:
        assert "CONTENT_GIT_COMMIT" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("ENVIRONMENT=production accepted unknown code SHA")


def test_default_code_sha_keeps_nonproduction_compatibility(monkeypatch):
    monkeypatch.setenv("CONTENT_ENV", "test")
    monkeypatch.delenv("CONTENT_GIT_COMMIT", raising=False)
    assert default_code_sha() == "unknown"


def test_qdrant_projection_contains_lineage_and_filter_payload(monkeypatch):
    from stock_content.adapters.qdrant.knowledge_index import QdrantKnowledgeIndex

    captured = {}

    class PointStruct:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeClient:
        def collection_exists(self, collection):
            return True

        def upsert(self, collection, points):
            captured["collection"] = collection
            captured["points"] = points

    models = types.ModuleType("qdrant_client.models")
    models.PointStruct = PointStruct
    models.Distance = types.SimpleNamespace(COSINE="COSINE")
    models.VectorParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)

    index = QdrantKnowledgeIndex.__new__(QdrantKnowledgeIndex)
    index._client = FakeClient()
    index._collection = "knowledge_v3"
    index._embed = lambda statement: [1.0]
    index._vector_size = 1
    index.index(
        [
            KnowledgeUnit(
                knowledge_uid="k-lineage",
                video_id="video",
                chapter_id=None,
                statement="statement",
                ticker="600000",
                knowledge_kind="EVENT",
                support_status="SOURCE_SUPPORTED",
                truth_status="VERIFIED",
                available_from=datetime(2026, 1, 1, tzinfo=UTC),
                attributes={"content_snapshot_id": "cs-1"},
                provenance={"claim_ids": ["claim-1"]},
            )
        ]
    )
    payload = captured["points"][0].payload
    assert {
        "knowledge_uid": "k-lineage",
        "content_snapshot_id": "cs-1",
        "claim_ids": ["claim-1"],
        "ticker": "600000",
        "knowledge_kind": "EVENT",
        "kind": "CLAIM",
        "support_status": "SOURCE_SUPPORTED",
        "verification_status": "VERIFIED",
        "available_from": "2026-01-01T00:00:00+00:00",
    }.items() <= payload.items()
    assert {
        "occurrence_id", "claim_id", "semantic_segment_id", "temporal_scope",
        "primary_temporal_role", "period_label", "forecast_start", "forecast_end",
        "granularity", "assertion_status", "source_available_at",
        "source_availability_quality", "lifecycle_status", "lifecycle_artifact_id",
    } <= payload.keys()

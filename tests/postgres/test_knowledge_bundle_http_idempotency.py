"""Real PostgreSQL + HTTP coverage for durable Bundle retry binding."""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from stock_content.adapters.postgres.models import (
    ContentKnowledgeBundleIdempotencyRow,
    ContentKnowledgeBundleRow,
)
from stock_content.adapters.postgres.repositories.knowledge_bundle_repository import PostgresKnowledgeBundleRepository
from stock_content.api.main import create_app
from stock_content.api.security import ServiceAuthorizer
from stock_content.application.knowledge_bundle_service import BundleProducerMetadata, KnowledgeBundleService
from stock_content.domain.knowledge_bundle import KnowledgeBundleRequest, canonical_json

pytestmark = pytest.mark.skipif(
    not os.getenv("CONTENT_TEST_POSTGRES_URL"),
    reason="CONTENT_TEST_POSTGRES_URL is required for real PostgreSQL tests",
)

NOW = datetime(2026, 9, 6, tzinfo=UTC)
CHECKSUM = "sha256:EBFD13B78622C3846890438A4FB3CB858278F571FDAB247CDD72EF18CA211621"


class StaticAuthority:
    def __init__(self, *, fail_on_read: bool = False) -> None:
        self.fail_on_read = fail_on_read

    def read_bundle_source(self, _request):
        if self.fail_on_read:
            raise AssertionError("durable idempotency lookup must precede authority reads")
        quote = "增长"
        return {
            "snapshot_available": True,
            "source": {
                "source_type": "xiaoe",
                "source_identity_hash": "identity-http",
                "source_version_id": "version-http",
                "canonical_url": "https://x.test/course",
                "source_content_hash": "source-http",
            },
            "items": [{
                "knowledge_id": "ku_http", "claim_id": "cl_http", "occurrence_id": "co_http",
                "statement": "收入增长约20%", "subject": {"type": "EQUITY", "key": "600000"},
                "predicate": "revenue_growth", "object": {"value": 20, "unit": "percent"},
                "support_status": "SOURCE_SUPPORTED", "lifecycle_status": "ACTIVE",
                "temporal": {
                    "target_start": "2026-10-01T00:00:00Z", "target_end": "2026-12-31T00:00:00Z",
                    "precision": "EXACT",
                },
                "evidence": [{
                    "evidence_id": "ev_http", "ownership": "PRIMARY", "start_ms": 1, "end_ms": 2,
                    "quote": quote, "artifact_id": "tr_http", "segment_id": "seg_http",
                    "quote_hash": "sha256:" + hashlib.sha256(canonical_json(quote).encode("utf-8")).hexdigest(),
                    "modality": "transcript",
                }],
                "verification": {"status": "SOURCE_VERIFIED", "reason_codes": ["SOURCE"]},
                "known_from": "2026-09-05T00:00:00Z", "available_from": "2026-09-05T00:00:00Z",
                "claim_schema_version": "claim.atomic.v1", "grounding_status": "GROUNDED",
                "legacy_grounding_incomplete": False,
            }],
        }


class BundleOnlyApplication:
    _tasks = type("Tasks", (), {"_sessions": None, "claim_pending": lambda *_args, **_kwargs: None})()

    def __init__(self, service: KnowledgeBundleService) -> None:
        self.service = service

    def create_knowledge_bundle(self, request, *, idempotency_key=None):
        return self.service.create(request, idempotency_key=idempotency_key)

    def get_knowledge_bundle(self, bundle_id):
        return self.service.get(bundle_id)


def _service(postgres_database, *, fail_on_read=False):
    return KnowledgeBundleService(
        StaticAuthority(fail_on_read=fail_on_read),
        PostgresKnowledgeBundleRepository(postgres_database.session_factory),
        BundleProducerMetadata("stock_content", "test", "test-commit", "pipeline.test", CHECKSUM),
    )


def _request_body():
    request = KnowledgeBundleRequest(
        content_snapshot_id="cs_http", query="核心逻辑", symbol="600000",
        business_as_of=NOW, knowledge_as_of=NOW, availability_as_of=NOW,
        minimum_support_status="SOURCE_SUPPORTED", max_items=30,
    )
    return {
        key: value.isoformat().replace("+00:00", "Z") if isinstance(value, datetime) else value
        for key, value in request.canonical_request().items()
    }


def _authorizer(tmp_path: Path):
    token_file = tmp_path / "token"
    token_file.write_text("bundle-token\n", encoding="utf-8")
    return ServiceAuthorizer((token_file,), ("stock_agent",))


def _headers():
    return {
        "Authorization": "Bearer bundle-token", "x-caller-service": "stock_agent",
        "x-trace-id": "bundle-pg-http", "Idempotency-Key": "bundle-real-postgres-key",
    }


def test_real_postgres_http_bundle_idempotency_replays_across_service_instances(postgres_database, tmp_path):
    body = _request_body()
    first_app = create_app(BundleOnlyApplication(_service(postgres_database)), authorizer=_authorizer(tmp_path))
    with TestClient(first_app) as client:
        first = client.post("/v1/content/knowledge-bundles", json=body, headers=_headers())
    assert first.status_code == 200

    # A new HTTP app/service/repository instance only has the PostgreSQL
    # mapping to consult; it must return the original Bundle without reading
    # an authority that is intentionally unavailable.
    replay_app = create_app(
        BundleOnlyApplication(_service(postgres_database, fail_on_read=True)), authorizer=_authorizer(tmp_path)
    )
    with TestClient(replay_app) as client:
        replay = client.post("/v1/content/knowledge-bundles", json=body, headers=_headers())
        conflict = client.post(
            "/v1/content/knowledge-bundles", json={**deepcopy(body), "max_items": 19}, headers=_headers()
        )
    assert replay.status_code == 200 and replay.json() == first.json()
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert "bundle-real-postgres-key" not in str(conflict.json())

    with postgres_database.session_factory() as session:
        assert len(list(session.scalars(select(ContentKnowledgeBundleRow)))) == 1
        mappings = list(session.scalars(select(ContentKnowledgeBundleIdempotencyRow)))
    assert len(mappings) == 1
    assert mappings[0].idempotency_key_hash != "bundle-real-postgres-key"

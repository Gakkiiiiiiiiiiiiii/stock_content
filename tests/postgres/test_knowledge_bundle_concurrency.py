"""Real PostgreSQL atomic-idempotency coverage for Bundle persistence (opt-in)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select

from stock_content.adapters.postgres.models import ContentKnowledgeBundleRow
from stock_content.adapters.postgres.repositories.knowledge_bundle_repository import PostgresKnowledgeBundleRepository

pytestmark = pytest.mark.skipif(
    not os.getenv("CONTENT_TEST_POSTGRES_URL"),
    reason="CONTENT_TEST_POSTGRES_URL is required for real PostgreSQL tests",
)


def _bundle() -> dict[str, object]:
    digest = "a" * 64
    return {
        "bundle_id": "ckb_" + digest,
        "bundle_hash": "sha256:" + digest,
        "content_snapshot_id": "snapshot-concurrent",
        "request_hash": "sha256:" + "b" * 64,
        "contract": "content-knowledge-bundle.v1",
        "producer": {"git_commit": "test-commit", "pipeline_version": "test-pipeline"},
    }


def test_concurrent_identical_bundle_inserts_return_one_immutable_row(postgres_database):
    repository = PostgresKnowledgeBundleRepository(postgres_database.session_factory)
    bundle = _bundle()
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(repository.insert, [bundle] * 8))
    assert results == [bundle] * 8
    with postgres_database.session_factory() as session:
        rows = list(session.scalars(select(ContentKnowledgeBundleRow)))
    assert len(rows) == 1 and dict(rows[0].payload) == bundle

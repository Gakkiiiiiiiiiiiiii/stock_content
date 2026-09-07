"""Real psycopg regressions for immutable first-write selection."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.claim_occurrence_repository import ClaimOccurrenceRepository
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import (
    SnapshotIntegrityError,
    SqlSnapshotStore,
)
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    VerificationJobIntegrityError,
    persist_verification_result,
)
from stock_content.domain.artifacts import SourceArtifact
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.lineage import build_content_snapshot
from stock_content.domain.temporal_semantics import OccurrenceTimes

pytestmark = pytest.mark.skipif(
    not os.getenv("CONTENT_TEST_POSTGRES_URL"),
    reason="CONTENT_TEST_POSTGRES_URL is required for real PostgreSQL tests",
)


def test_psycopg_first_write_uses_returning_without_early_integrity_checks(postgres_database):
    """Replays must preserve first writes; altered payloads must still fail closed."""
    now = datetime(2026, 1, 3, tzinfo=UTC)

    artifacts = SqlArtifactRepository(postgres_database.session_factory)
    source = SourceArtifact(
        artifact_id="psycopg-source",
        artifact_type="source",
        source_type="fixture",
        source_ref="psycopg",
        source_content_hash="source-hash",
    )
    artifacts.put(source)
    snapshots = SqlSnapshotStore(postgres_database.session_factory)
    snapshot = build_content_snapshot(
        source_type="fixture",
        source_ref="psycopg",
        source_content_hash="source-hash",
        artifact_ids={"source": source.artifact_id},
        code_sha="psycopg-test",
        created_at=now,
    )
    snapshots.save(snapshot)
    snapshots.save(snapshot)
    stored_snapshot = snapshots.get(snapshot.content_snapshot_id)
    assert stored_snapshot is not None
    assert stored_snapshot.content_snapshot_id == snapshot.content_snapshot_id
    assert stored_snapshot.artifact_ids == snapshot.artifact_ids
    conflicting_manifest = dict(snapshot.producer_manifest)
    conflicting_manifest["code_sha"] = "other-code"
    with pytest.raises(SnapshotIntegrityError, match="different payload|changed"):
        snapshots.save(
            replace(snapshot, code_sha="other-code", producer_manifest=conflicting_manifest)
        )

    occurrences = ClaimOccurrenceRepository(postgres_database.session_factory)
    first = ClaimOccurrence(
        claim_id="psycopg-claim",
        source_artifact_id=source.artifact_id,
        transcript_artifact_id="psycopg-transcript",
        semantic_segment_id="psycopg-segment",
        evidence_refs=["psycopg-evidence"],
        source_support_status="SOURCE_LOCATED",
        source_confidence=0.4,
        extractor_confidence=0.5,
        times=OccurrenceTimes(
            source_published_at=now - timedelta(days=3),
            source_available_at=now - timedelta(days=2),
            ingested_at=now - timedelta(days=2),
            extraction_completed_at=now - timedelta(days=1),
            snapshot_committed_at=now,
            available_from=now,
        ),
    )
    replay = first.model_copy(update={"evidence_refs": [], "condition_evidence_refs": ["psycopg-evidence"]})
    assert replay.occurrence_id == first.occurrence_id
    occurrences.save(first)
    assert occurrences.save(replay).evidence_refs == ["psycopg-evidence"]
    stored_occurrence = occurrences.get(first.occurrence_id)
    assert stored_occurrence is not None
    assert stored_occurrence.condition_evidence_refs == []

    claim = FinancialClaim(
        claim_type="PRICE",
        subject_type="EQUITY",
        subject_id="600000.SH",
        predicate="price",
        value=10,
        evidence_refs=["psycopg-evidence"],
        source_confidence=0.9,
        extractor_confidence=0.9,
    )
    SqlClaimRepository(postgres_database.session_factory).save(claim)
    result = VerificationResult(
        claim_id=claim.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-1",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    values = {
        "verification_id": "psycopg-verification",
        "claim_id": claim.claim_id,
        "provider": "quant",
        "status": result.status,
        "market_snapshot_id": result.market_snapshot_id,
        "market_data_version": result.market_data_version,
        "result_payload": result.model_dump(mode="json"),
        "trace_id": "trace-1",
        "fact_date": result.fact_date,
        "adjustment": result.adjustment,
        "verification_timestamp": result.verification_timestamp,
        "verification_rule_version": result.verification_rule_version,
        "verified_at": result.verification_timestamp,
        "available_at": now,
        "created_at": now,
    }
    with postgres_database.session_factory.begin() as session:
        persist_verification_result(session, values)
        persist_verification_result(session, values)
        with pytest.raises(VerificationJobIntegrityError, match="different result"):
            persist_verification_result(
                session,
                {**values, "result_payload": {**values["result_payload"], "reason": "different"}},
            )

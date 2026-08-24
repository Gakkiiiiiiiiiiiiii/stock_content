from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ClaimVerificationJobRow,
    ContentSourceHeadRow,
    FinancialClaimRow,
    SignalOutboxRow,
)
from stock_content.adapters.postgres.repositories import (
    PostgresVerificationJobRepository,
    SignalOutboxRepository,
    SqlArtifactRepository,
    SqlClaimRepository,
    SqlSnapshotStore,
)
from stock_content.domain.artifacts import SourceArtifact
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.lineage import build_content_snapshot
from stock_content.domain.signal_contract import validate_signal_v4
from stock_content.domain.signal_policy import SignalPolicy


def test_source_head_upsert_is_monotonic_when_older_snapshot_commits_last(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'source-head.db'}")
    database.create_schema()
    snapshots = SqlSnapshotStore(database.session_factory)
    artifacts = SqlArtifactRepository(database.session_factory)
    artifacts.put(
        SourceArtifact(
            artifact_id="source-old",
            artifact_type="source",
            source_type="fixture",
            source_ref="same-source",
            source_content_hash="old-content",
        )
    )
    artifacts.put(
        SourceArtifact(
            artifact_id="source-new",
            artifact_type="source",
            source_type="fixture",
            source_ref="same-source",
            source_content_hash="new-content",
        )
    )
    older = replace(
        build_content_snapshot(
            source_type="fixture",
            source_ref="same-source",
            source_content_hash="old-content",
            artifact_ids={"source": "source-old"},
        ),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = replace(
        build_content_snapshot(
            source_type="fixture",
            source_ref="same-source",
            source_content_hash="new-content",
            artifact_ids={"source": "source-new"},
        ),
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    snapshots.save(newer)
    snapshots.save(older)

    with database.session_factory() as session:
        head = session.scalar(select(ContentSourceHeadRow))
        assert head is not None
        assert head.latest_snapshot_id == newer.content_snapshot_id
        assert head.updated_at.replace(tzinfo=UTC) == newer.created_at


def test_claim_job_and_outbox_duplicate_inserts_are_idempotent(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'idempotency.db'}")
    database.create_schema()
    claim = FinancialClaim(
        claim_type="PRICE",
        subject_type="EQUITY",
        subject_id="600000.SH",
        predicate="price",
        value=10,
        evidence_refs=["evidence-1"],
        source_confidence=0.9,
        extractor_confidence=0.9,
        fact_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    claims = SqlClaimRepository(database.session_factory)
    claims.save(claim)
    claims.save(claim)

    jobs = PostgresVerificationJobRepository(database.session_factory)
    jobs.enqueue([claim], trace_id="first")
    jobs.enqueue([claim], trace_id="second")

    outbox = SignalOutboxRepository(database.session_factory)
    payload = {
        "signal_id": "signal-idempotent",
        "signal_schema_version": "content-factor-signal.v4",
        "content_snapshot_id": "snapshot-1",
        "claim_id": claim.claim_id,
    }
    outbox.enqueue(payload)
    outbox.enqueue(payload)

    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(FinancialClaimRow)) == 1
        assert session.scalar(select(func.count()).select_from(ClaimVerificationJobRow)) == 1
        assert session.scalar(select(func.count()).select_from(SignalOutboxRow)) == 1


def test_v4_validation_rejects_decision_mismatch_and_nested_additional_properties():
    policy = SignalPolicy()
    claim = FinancialClaim(
        claim_type="PRICE",
        subject_type="EQUITY",
        subject_id="600000.SH",
        predicate="price",
        value=10,
        evidence_refs=["evidence-1"],
        source_confidence=0.9,
        extractor_confidence=0.9,
    )
    signal = policy.build_signal(
        {"content_snapshot_id": "snapshot-1", "source_type": "fixture", "source_ref": "source-1", "created_at": "now"},
        claim,
        {"artifact_id": "verification-1", "status": "VERIFIED"},
    )
    validate_signal_v4(signal)

    with pytest.raises(ValueError, match="decision_id"):
        validate_signal_v4({**signal, "producer": {**signal["producer"], "decision_id": "different"}})
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_signal_v4({**signal, "producer": {**signal["producer"], "unexpected": True}})
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_signal_v4({**signal, "support": {**signal["support"], "unexpected": True}})

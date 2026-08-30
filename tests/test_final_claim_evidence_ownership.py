from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ClaimEvidenceRow
from stock_content.adapters.postgres.repositories.claim_occurrence_repository import ClaimOccurrenceRepository
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.api.dependencies import build_application
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.temporal_semantics import OccurrenceTimes


def _claim() -> FinancialClaim:
    return FinancialClaim(
        claim_id="claim-final-evidence",
        claim_type="FINANCIAL_METRIC",
        subject_type="EQUITY",
        subject_id="600000.SH",
        predicate="revenue",
        value=100,
        unit="CNY",
        claim_schema_version="claim.final.v1",
        evidence_refs=["stale-global-evidence"],
        source_confidence=0.9,
        extractor_confidence=0.9,
    )


def _occurrence(*, source: str, evidence: str, claim_id: str = "claim-final-evidence") -> ClaimOccurrence:
    now = datetime(2026, 1, 3, tzinfo=UTC)
    return ClaimOccurrence(
        claim_id=claim_id,
        source_artifact_id=source,
        transcript_artifact_id=f"transcript-{source}",
        semantic_segment_id=f"segment-{source}",
        evidence_refs=[evidence],
        times=OccurrenceTimes(
            source_published_at=now - timedelta(days=2),
            source_available_at=now - timedelta(days=1),
            ingested_at=now - timedelta(days=1),
            extraction_completed_at=now,
            snapshot_committed_at=now,
            available_from=now,
        ),
        source_confidence=0.8,
        extractor_confidence=0.8,
    )


def test_final_claim_never_reads_or_writes_legacy_evidence_membership(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'final-ownership.db'}")
    database.create_schema()
    claims = SqlClaimRepository(database.session_factory)

    # The compatibility argument is intentionally accepted for old callers,
    # but is ignored for final claims (SQLite equivalent of the DB trigger).
    claim = _claim()
    claims.save(claim, compatibility_evidence_refs=["legacy-evidence"])
    with database.session_factory() as session:
        assert session.scalars(select(ClaimEvidenceRow)).all() == []
    assert claims.get(claim.claim_id).evidence_refs == []
    assert claims.evidence(claim.claim_id) == []


def test_final_evidence_reverse_lookup_uses_occurrence_membership(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'reverse-ownership.db'}")
    database.create_schema()
    claims = SqlClaimRepository(database.session_factory)
    occurrences = ClaimOccurrenceRepository(database.session_factory)
    claims.save(_claim())
    first = _occurrence(source="source-a", evidence="evidence-a")
    second = _occurrence(source="source-b", evidence="evidence-b")
    occurrences.save(first)
    occurrences.save(second)

    assert [item.occurrence_id for item in occurrences.list_by_evidence("evidence-a")] == [first.occurrence_id]
    assert [item.occurrence_id for item in occurrences.list_by_evidence("evidence-b")] == [second.occurrence_id]
    assert [item.claim_id for item in claims.claims_for_evidence("evidence-a")] == [_claim().claim_id]
    assert [item.claim_id for item in claims.claims_for_evidence("evidence-b")] == [_claim().claim_id]


def test_application_final_claim_evidence_uses_occurrence_roles_only(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'application-evidence.db'}", enable_qdrant=False)
    claim = _claim()
    application._claim_repository.save(claim, compatibility_evidence_refs=["legacy-global"])  # noqa: SLF001
    occurrence = _occurrence(source="source-role", evidence="primary-evidence", claim_id=claim.claim_id)
    occurrence = occurrence.model_copy(update={
        "occurrence_id": "", "assertion_locator_hash": "",
        "condition_evidence_refs": ["condition-evidence"],
        "invalidation_evidence_refs": ["invalidation-evidence"],
        "temporal_evidence_refs": ["temporal-evidence"],
    })
    # Re-validate so the occurrence locator reflects all role coordinates.
    occurrence = ClaimOccurrence.model_validate(occurrence.model_dump())
    application._occurrence_repository.save(occurrence)  # noqa: SLF001
    assert application._claim_repository.evidence(claim.claim_id) == []  # noqa: SLF001
    assert application.get_claim_evidence(claim.claim_id) == [
        "condition-evidence", "invalidation-evidence", "primary-evidence", "temporal-evidence",
    ]
    assert application.get_claim_evidence("missing-claim") is None


def test_final_ownership_migration_is_guarded_and_idempotent_by_construction():
    migration = Path(__file__).parents[1] / "migrations" / "024_final_claim_evidence_ownership.sql"
    sql = migration.read_text(encoding="utf-8")
    assert "RAISE EXCEPTION" in sql
    assert "claim_occurrence_evidence" in sql
    assert "DELETE FROM claim_evidence" in sql
    assert "CREATE OR REPLACE FUNCTION reject_final_claim_evidence" in sql
    assert "DROP TRIGGER IF EXISTS trg_reject_final_claim_evidence" in sql
    assert "BEFORE INSERT OR UPDATE ON claim_evidence" in sql


def test_replay_of_source_a_is_not_contaminated_by_future_source_b(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'cross-source.db'}", enable_qdrant=False)
    options = {
        "metadata": {"title": "same canonical proposition"},
        "transcript": "股票600000营收增长10%。",
        "offline_fixture": True,
    }
    application.enqueue("bilibili", "BV-source-a", options)
    first_result = application.process_next("source-a-worker")
    assert first_result["status"] == "SUCCEEDED"
    first_snapshot_id = first_result["content_snapshot_id"]

    first_snapshot = application._snapshots.get(first_snapshot_id)  # noqa: SLF001
    assert first_snapshot is not None
    claim_ids = application._artifact_repository.get(  # noqa: SLF001
        first_snapshot.artifact_ids["claims"]
    ).claims
    assert len(claim_ids) == 1
    claim_id = str(claim_ids[0])

    # Add the later source's assertion directly at the authoritative
    # occurrence boundary.  This avoids exercising the unrelated knowledge
    # projection uniqueness policy while keeping the exact P0-1 scenario:
    # one canonical claim, two source-owned evidence memberships.
    occurrences = application._occurrence_repository  # noqa: SLF001
    occurrences.save(_occurrence(source="source-b", evidence="evidence-b", claim_id=claim_id))
    assert [item.claim_id for item in application._claim_repository.claims_for_evidence(  # noqa: SLF001
        "evidence-b"
    )] == [claim_id]

    replay = application.replay_content_snapshot(first_snapshot_id)
    assert replay["identity_match"] is True
    claims = application._claim_repository  # noqa: SLF001
    assert all(claims.get(claim_id).evidence_refs == [] for claim_id in claim_ids)

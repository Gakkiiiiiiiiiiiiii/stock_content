from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ClaimVerificationJobRow
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import SqlSnapshotStore
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    PostgresVerificationJobRepository,
)
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.artifacts import VerificationArtifact, artifact_id_of
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.initial_verification import build_initial_verification_plan


def _claim(claim_type="PRICE"):
    return FinancialClaim(
        claim_type=claim_type, subject_type="EQUITY", subject_id="600000",
        predicate="price", value=10, evidence_refs=["e"],
        source_confidence=1, extractor_confidence=1,
    )


def _publish(database, snapshots, artifacts, plan, source_ref, when):
    verification = VerificationArtifact(
        artifact_id="verification", artifact_type="verification", producer_stage="test",
        results=plan.artifact_results,
    )
    verification = VerificationArtifact(
        **{**verification.__dict__, "artifact_id": artifact_id_of(verification)}
    )
    artifacts.put(verification)
    return snapshots.record_bundle_from_artifacts(
        source_type="fixture", source_ref=source_ref, source_content_hash=source_ref,
        artifact_ids={"verification": verification.artifact_id}, code_sha="sha", created_at=when,
        verification_jobs=tuple(plan.pending_jobs_to_insert),
        verification_results=tuple(plan.terminal_results_to_insert),
    )


def test_two_sources_share_one_real_active_job(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'active.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    claim = _claim()
    first_time = datetime(2026, 1, 1, tzinfo=UTC)
    first = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=first_time,
        verification_repository=jobs, job_repository=jobs,
    )
    _publish(database, snapshots, artifacts, first, "source-a", first_time)
    second = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=first_time + timedelta(days=1),
        verification_repository=jobs, job_repository=jobs,
    )
    _publish(database, snapshots, artifacts, second, "source-b", first_time + timedelta(days=1))
    assert not second.pending_jobs_to_insert
    assert second.reused_job_ids == [first.artifact_results[0].verification_job_id]
    with database.session_factory() as session:
        count = session.scalar(
            select(func.count()).select_from(ClaimVerificationJobRow).where(
                ClaimVerificationJobRow.claim_id == claim.claim_id,
                ClaimVerificationJobRow.provider == "quant",
            )
        )
    assert count == 1


def test_two_sources_reuse_one_pit_terminal_result(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'terminal.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    claim = _claim("CORPORATE_EVENT")
    first_time = datetime(2026, 1, 1, tzinfo=UTC)
    first = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=first_time,
        verification_repository=jobs, job_repository=jobs,
    )
    _publish(database, snapshots, artifacts, first, "source-a", first_time)
    second = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=first_time + timedelta(days=1),
        verification_repository=jobs, job_repository=jobs,
    )
    _publish(database, snapshots, artifacts, second, "source-b", first_time + timedelta(days=1))
    assert not second.terminal_results_to_insert
    assert second.reused_terminal_result_ids == [first.artifact_results[0].verification_id]

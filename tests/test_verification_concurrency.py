from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from threading import Barrier

from sqlalchemy import func, select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ClaimVerificationJobRow, ContentSnapshotRow
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import SqlSnapshotStore
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    PostgresVerificationJobRepository,
)
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.artifacts import VerificationArtifact, artifact_id_of
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.initial_verification import build_initial_verification_plan


def test_concurrent_planners_publish_without_duplicate_or_missing_job(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'concurrent.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    claim = FinancialClaim(
        claim_type="PRICE", subject_type="EQUITY", subject_id="600000", predicate="price", value=10,
        evidence_refs=["e"], source_confidence=1, extractor_confidence=1,
    )
    barrier = Barrier(2)
    commit = datetime(2026, 1, 1, tzinfo=UTC)

    def publish(source_ref):
        barrier.wait()
        with jobs.planning_uow([(claim.claim_id, "quant")]) as session:
            plan = build_initial_verification_plan(
                claims=[claim], provider="quant", snapshot_candidate_time=commit,
                verification_repository=jobs, job_repository=jobs, session=session,
            )
            artifact = VerificationArtifact(
                artifact_id="verification", artifact_type="verification", producer_stage="test",
                results=plan.artifact_results,
            )
            artifact = VerificationArtifact(
                **{**artifact.__dict__, "artifact_id": artifact_id_of(artifact)}
            )
            artifacts.put_in_session(session, artifact)
            snapshots.record_bundle_from_artifacts(
                source_type="fixture", source_ref=source_ref, source_content_hash=source_ref,
                artifact_ids={"verification": artifact.artifact_id}, code_sha="sha", created_at=commit,
                verification_jobs=tuple(plan.pending_jobs_to_insert),
                verification_results=tuple(plan.terminal_results_to_insert), session=session,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(publish, ["source-a", "source-b"]))
    with database.session_factory() as session:
        job_count = session.scalar(select(func.count()).select_from(ClaimVerificationJobRow))
        snapshot_count = session.scalar(select(func.count()).select_from(ContentSnapshotRow))
    assert job_count == 1
    assert snapshot_count == 2

    # Race a real worker completion against a third-source planner.  Either
    # legal order is acceptable: the planner may close over the active job, or
    # it may reuse the terminal result after the worker commits.
    leased = jobs.claim_due("worker", lease_seconds=7 * 86400, now=commit)
    assert len(leased) == 1
    race_barrier = Barrier(2)

    def publish_after_worker():
        race_barrier.wait()
        with jobs.planning_uow([(claim.claim_id, "quant")]) as session:
            plan = build_initial_verification_plan(
                claims=[claim], provider="quant", snapshot_candidate_time=commit.replace(day=3),
                verification_repository=jobs, job_repository=jobs, session=session,
            )
            artifact = VerificationArtifact(
                artifact_id="verification-race", artifact_type="verification", producer_stage="test",
                results=plan.artifact_results,
            )
            artifact = VerificationArtifact(
                **{**artifact.__dict__, "artifact_id": artifact_id_of(artifact)}
            )
            artifacts.put_in_session(session, artifact)
            snapshots.record_bundle_from_artifacts(
                source_type="fixture", source_ref="source-race", source_content_hash="source-race",
                artifact_ids={"verification": artifact.artifact_id}, code_sha="sha", created_at=commit.replace(day=3),
                verification_jobs=tuple(plan.pending_jobs_to_insert),
                verification_results=tuple(plan.terminal_results_to_insert), session=session,
            )

    def complete_worker():
        race_barrier.wait()
        jobs.complete_result(
            leased[0].job_id, "worker",
            VerificationResult(
                claim_id=claim.claim_id, status="VERIFIED", market_snapshot_id="m1",
                market_data_version="v1", fact_date=date(2026, 1, 1),
                verification_timestamp=commit,
            ), now=commit.replace(day=2),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda fn: fn(), [publish_after_worker, complete_worker]))
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ClaimVerificationJobRow)) == 1
        assert session.scalar(select(func.count()).select_from(ContentSnapshotRow)) == 3
    race_snapshot = snapshots.list_for_source("fixture", "source-race")[0]
    race_artifact = artifacts.get(race_snapshot.artifact_ids["verification"])
    race_entry = race_artifact.results[0]
    if race_entry.status == "VERIFICATION_PENDING":
        assert race_entry.verification_job_id
        assert jobs.get_job(race_entry.verification_job_id) is not None
    else:
        assert race_entry.verification_id
        race_result = jobs.get_result(race_entry.verification_id)
        assert race_result is not None
        available = race_result.available_at
        if available.tzinfo is None:
            available = available.replace(tzinfo=UTC)
        committed = race_snapshot.created_at
        if committed.tzinfo is None:
            committed = committed.replace(tzinfo=UTC)
        assert available <= committed

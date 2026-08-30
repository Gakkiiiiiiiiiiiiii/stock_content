from datetime import UTC, datetime, timedelta

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import SqlSnapshotStore
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    PostgresVerificationJobRepository,
)
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.verification_refresh import VerificationRefreshService
from stock_content.domain.artifacts import ClaimArtifact, VerificationArtifact, artifact_id_of
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.initial_verification import build_initial_verification_plan


def test_source_b_reuses_refresh_terminal_result_and_refresh_replays_exactly(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'refresh-reuse.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    claims = SqlClaimRepository(database.session_factory)
    claim = FinancialClaim(
        claim_type="PRICE", subject_type="EQUITY", subject_id="600000", predicate="price", value=10,
        evidence_refs=["e"], source_confidence=1, extractor_confidence=1,
    )
    claims.save(claim)
    candidate = datetime(2026, 1, 1, tzinfo=UTC)
    initial_plan = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=candidate,
        verification_repository=jobs, job_repository=jobs,
    )
    claim_artifact = ClaimArtifact(
        artifact_id="claims", artifact_type="claims", producer_stage="test", claims=[claim]
    )
    claim_artifact = ClaimArtifact(
        **{**claim_artifact.__dict__, "artifact_id": artifact_id_of(claim_artifact)}
    )
    verification = VerificationArtifact(
        artifact_id="verification", artifact_type="verification", producer_stage="test",
        claim_artifact_id=claim_artifact.artifact_id, parent_artifact_ids=(claim_artifact.artifact_id,),
        results=initial_plan.artifact_results,
    )
    verification = VerificationArtifact(**{**verification.__dict__, "artifact_id": artifact_id_of(verification)})
    artifacts.put(claim_artifact)
    artifacts.put_claim_members(claim_artifact)
    artifacts.put(verification)
    source_a = snapshots.record_bundle_from_artifacts(
        source_type="fixture", source_ref="source-a", source_content_hash="source-a",
        artifact_ids={"claims": claim_artifact.artifact_id, "verification": verification.artifact_id},
        code_sha="sha", created_at=candidate, verification_jobs=tuple(initial_plan.pending_jobs_to_insert),
    )
    job = jobs.claim_due("refresh-worker", now=candidate)[0]
    completed_at = candidate + timedelta(hours=1)
    result = VerificationResult(
        claim_id=claim.claim_id, status="VERIFIED", market_snapshot_id="market-1",
        market_data_version="bars.v1", fact_date=candidate.date(), adjustment="NONE",
        verification_timestamp=completed_at,
    )
    refresh = VerificationRefreshService(database.session_factory, jobs, claims)
    refreshed = refresh.complete(
        job.job_id, "refresh-worker", result,
        parent_snapshot_id=source_a.content_snapshot_id, now=completed_at,
    )
    refresh_snapshot = snapshots.get(refreshed["snapshot_id"])
    refresh_artifact = artifacts.get(refresh_snapshot.artifact_ids["verification"])
    current_entry = next(item for item in refresh_artifact.results if item.claim_id == claim.claim_id)
    assert current_entry.verification_id == refreshed["verification_id"]
    assert current_entry.provider == "quant"
    row = jobs.get_result(refreshed["verification_id"])
    assert row is not None and row.available_at is not None
    available_at = row.available_at.replace(tzinfo=UTC) if row.available_at.tzinfo is None else row.available_at
    assert available_at <= completed_at

    source_b_candidate = completed_at + timedelta(minutes=1)
    second_plan = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=source_b_candidate,
        verification_repository=jobs, job_repository=jobs,
    )
    assert second_plan.pending_jobs_to_insert == []
    assert second_plan.reused_terminal_result_ids == [refreshed["verification_id"]]
    assert second_plan.artifact_results[0].verification_id == refreshed["verification_id"]
    source_b_verification = VerificationArtifact(
        artifact_id="verification-b", artifact_type="verification", producer_stage="test",
        claim_artifact_id=claim_artifact.artifact_id, parent_artifact_ids=(claim_artifact.artifact_id,),
        results=second_plan.artifact_results,
    )
    source_b_verification = VerificationArtifact(
        **{**source_b_verification.__dict__, "artifact_id": artifact_id_of(source_b_verification)}
    )
    artifacts.put(source_b_verification)
    source_b = snapshots.record_bundle_from_artifacts(
        source_type="fixture", source_ref="source-b", source_content_hash="source-b",
        artifact_ids={"claims": claim_artifact.artifact_id, "verification": source_b_verification.artifact_id},
        code_sha="sha", created_at=source_b_candidate,
    )
    assert source_b.content_snapshot_id
    replay = ReplayService(
        snapshots, artifact_repository=artifacts, verification_repository=jobs
    ).replay(refresh_snapshot.content_snapshot_id)
    assert replay["identity_match"] is True


def test_refresh_replay_rejects_tampered_null_and_future_available_at(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'refresh-integrity.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    claims = SqlClaimRepository(database.session_factory)
    claim = FinancialClaim(
        claim_type="PRICE", subject_type="EQUITY", subject_id="600001", predicate="price", value=10,
        evidence_refs=["e"], source_confidence=1, extractor_confidence=1,
    )
    claims.save(claim)
    commit = datetime(2026, 1, 1, tzinfo=UTC)
    plan = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=commit,
        verification_repository=jobs, job_repository=jobs,
    )
    ca = ClaimArtifact(artifact_id="claims", artifact_type="claims", producer_stage="test", claims=[claim])
    ca = ClaimArtifact(**{**ca.__dict__, "artifact_id": artifact_id_of(ca)})
    va = VerificationArtifact(
        artifact_id="verification", artifact_type="verification", producer_stage="test",
        claim_artifact_id=ca.artifact_id, parent_artifact_ids=(ca.artifact_id,), results=plan.artifact_results,
    )
    va = VerificationArtifact(**{**va.__dict__, "artifact_id": artifact_id_of(va)})
    artifacts.put(ca)
    artifacts.put_claim_members(ca)
    artifacts.put(va)
    source = snapshots.record_bundle_from_artifacts(
        source_type="fixture", source_ref="source", source_content_hash="source",
        artifact_ids={"claims": ca.artifact_id, "verification": va.artifact_id}, code_sha="sha",
        created_at=commit, verification_jobs=tuple(plan.pending_jobs_to_insert),
    )
    job = jobs.claim_due("refresh-worker", now=commit)[0]
    refresh = VerificationRefreshService(database.session_factory, jobs, claims)
    result = VerificationResult(
        claim_id=claim.claim_id, status="VERIFIED", market_snapshot_id="m",
        market_data_version="v", fact_date=commit.date(), verification_timestamp=commit,
    )
    refreshed = refresh.complete(
        job.job_id, "refresh-worker", result,
        parent_snapshot_id=source.content_snapshot_id, now=commit,
    )
    replay = ReplayService(snapshots, artifact_repository=artifacts, verification_repository=jobs)
    result_id = refreshed["verification_id"]
    with database.session_factory.begin() as session:
        row = jobs.get_result(result_id, session=session)
        payload = dict(row.result_payload)
        payload["reason"] = "tampered"
        row.result_payload = payload
    assert replay.replay(refreshed["snapshot_id"])["error"] == "REPLAY_LINEAGE_REFERENCE_INVALID"
    with database.session_factory.begin() as session:
        row = jobs.get_result(result_id, session=session)
        refresh_snapshot = snapshots.get(refreshed["snapshot_id"])
        refresh_artifact = artifacts.get(refresh_snapshot.artifact_ids["verification"])
        entry = refresh_artifact.results[0]
        row.result_payload = entry.result.model_dump(mode="json")
        row.available_at = refresh_snapshot.created_at + timedelta(days=1)
    assert replay.replay(refreshed["snapshot_id"])["error"] == "REPLAY_LINEAGE_REFERENCE_INVALID"
    with database.session_factory.begin() as session:
        jobs.get_result(result_id, session=session).available_at = None
    assert replay.replay(refreshed["snapshot_id"])["error"] == "REPLAY_LINEAGE_REFERENCE_INVALID"

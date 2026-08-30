from datetime import UTC, datetime, timedelta

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ClaimVerificationJobRow, ClaimVerificationResultRow
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import SqlSnapshotStore
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    PostgresVerificationJobRepository,
)
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.artifacts import ClaimArtifact, VerificationArtifact, artifact_id_of
from stock_content.domain.claims import FinancialClaim, VerificationArtifactEntry, VerificationResult
from stock_content.domain.initial_verification import VerificationResultWrite, verification_id_of

COMMIT = datetime(2026, 1, 1, tzinfo=UTC)


def _artifact(entry):
    claim = FinancialClaim(
        claim_id=entry.claim_id, claim_type="PRICE", subject_type="EQUITY",
        subject_id="600000", predicate="price", value=10, evidence_refs=["e"],
        source_confidence=1, extractor_confidence=1,
    )
    claims = ClaimArtifact(artifact_id="claims", artifact_type="claims", producer_stage="test", claims=[claim])
    claims = ClaimArtifact(**{**claims.__dict__, "artifact_id": artifact_id_of(claims)})
    artifact = VerificationArtifact(
        artifact_id="verification", artifact_type="verification", producer_stage="test", results=[entry],
        claim_artifact_id=claims.artifact_id, parent_artifact_ids=(claims.artifact_id,),
    )
    return (
        VerificationArtifact(**{**artifact.__dict__, "artifact_id": artifact_id_of(artifact)}),
        claims,
    )


def test_replay_historical_pending_passes_after_job_completes(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'pending.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    job_id = "verify-job-historical-quant"
    artifact, claims = _artifact(VerificationArtifactEntry(
        claim_id="claim-historical", provider="quant", status="VERIFICATION_PENDING",
        verification_job_id=job_id,
    ))
    artifacts.put(claims)
    artifacts.put(artifact)
    snapshot = snapshots.record_bundle_from_artifacts(
        source_type="fixture", source_ref="historical", source_content_hash="historical",
        artifact_ids={"verification": artifact.artifact_id, "claims": claims.artifact_id},
        code_sha="sha", created_at=COMMIT,
        verification_jobs=({
            "job_id": job_id, "claim_id": "claim-historical", "provider": "quant",
            "status": "VERIFICATION_PENDING", "created_at": COMMIT,
        },),
    )
    with database.session_factory.begin() as session:
        session.get(ClaimVerificationJobRow, job_id).status = "VERIFIED"
    replay = ReplayService(
        snapshots, artifact_repository=artifacts, verification_repository=jobs
    ).replay(snapshot.content_snapshot_id)
    assert replay["identity_match"] is True
    assert replay["artifact_validation"]["checked"] is True


def test_replay_rejects_terminal_result_that_became_available_in_future(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'future.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    result = VerificationResult(
        claim_id="claim-future", status="NOT_VERIFIABLE", reason="test", available_at=COMMIT
    )
    result_id = verification_id_of("claim-future", "quant", result)
    artifact, claims = _artifact(VerificationArtifactEntry.from_result(
        result, provider="quant", verification_id=result_id
    ))
    artifacts.put(claims)
    artifacts.put(artifact)
    snapshot = snapshots.record_bundle_from_artifacts(
        source_type="fixture", source_ref="future", source_content_hash="future",
        artifact_ids={"verification": artifact.artifact_id, "claims": claims.artifact_id},
        code_sha="sha", created_at=COMMIT,
        verification_results=(VerificationResultWrite(
            verification_id=result_id, claim_id="claim-future", provider="quant",
            result=result, available_at=COMMIT,
        ),),
    )
    with database.session_factory.begin() as session:
        session.get(ClaimVerificationResultRow, result_id).available_at = COMMIT + timedelta(days=1)
    replay = ReplayService(
        snapshots, artifact_repository=artifacts, verification_repository=jobs
    ).replay(snapshot.content_snapshot_id)
    assert replay["error"] == "REPLAY_LINEAGE_REFERENCE_INVALID"


def test_current_head_liveness_distinguishes_active_terminal_and_dangling(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'liveness.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    job_id = "verify-job-live-quant"
    artifact, claims = _artifact(VerificationArtifactEntry(
        claim_id="claim-live", provider="quant", status="VERIFICATION_PENDING",
        verification_job_id=job_id,
    ))
    artifacts.put(claims)
    artifacts.put(artifact)
    snapshot = snapshots.record_bundle_from_artifacts(
        source_type="fixture", source_ref="live", source_content_hash="live",
        artifact_ids={"verification": artifact.artifact_id, "claims": claims.artifact_id},
        code_sha="sha", created_at=COMMIT,
        verification_jobs=({
            "job_id": job_id, "claim_id": "claim-live", "provider": "quant",
            "status": "VERIFICATION_PENDING", "created_at": COMMIT,
        },),
    )
    replay = ReplayService(
        snapshots, artifact_repository=artifacts, verification_repository=jobs
    )
    assert replay.check_current_verification_liveness(snapshot)["status"] == "LIVE"
    with database.session_factory.begin() as session:
        session.get(ClaimVerificationJobRow, job_id).status = "VERIFIED"
    assert replay.check_current_verification_liveness(snapshot)["error"] == (
        "DANGLING_CURRENT_VERIFICATION_PENDING"
    )
    newer = snapshots.record_from_artifacts(
        source_type="fixture", source_ref="live", source_content_hash="live-new",
        code_sha="sha", created_at=COMMIT + timedelta(days=1),
    )
    assert newer.content_snapshot_id != snapshot.content_snapshot_id
    assert replay.check_current_verification_liveness(snapshot)["status"] == "NOT_CURRENT_HEAD"
    with database.session_factory.begin() as session:
        session.delete(session.get(ClaimVerificationJobRow, job_id))

from datetime import UTC, datetime

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import SqlSnapshotStore
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    PostgresVerificationJobRepository,
)
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.artifacts import VerificationArtifact, artifact_id_of
from stock_content.domain.claims import VerificationArtifactEntry, VerificationResult
from stock_content.domain.initial_verification import (
    VerificationJobWrite,
    VerificationResultWrite,
    verification_id_of,
)

COMMIT = datetime(2026, 1, 1, tzinfo=UTC)


def _bundle(database, *, source_ref, failure_hook=None):
    artifacts = SqlArtifactRepository(database.session_factory)
    snapshots = SnapshotService(SqlSnapshotStore(database.session_factory))
    jobs = PostgresVerificationJobRepository(database.session_factory)
    claim_pending = "claim-pending"
    claim_terminal = "claim-terminal"
    job_id = "verify-job-claim-pending-quant"
    result = VerificationResult(
        claim_id=claim_terminal, status="NOT_VERIFIABLE",
        reason="FACT_NOT_QUANT_VERIFIABLE", available_at=COMMIT,
    )
    result_id = verification_id_of(claim_terminal, "quant", result)
    verification = VerificationArtifact(
        artifact_id="verification-pending", artifact_type="verification",
        producer_stage="test", results=[
            VerificationArtifactEntry(
                claim_id=claim_pending, provider="quant", status="VERIFICATION_PENDING",
                verification_job_id=job_id,
            ),
            VerificationArtifactEntry.from_result(
                result, provider="quant", verification_id=result_id,
            ),
        ],
    )
    verification = VerificationArtifact(
        **{**verification.__dict__, "artifact_id": artifact_id_of(verification)}
    )
    artifacts.put(verification)
    job = VerificationJobWrite(
        job_id=job_id, claim_id=claim_pending, provider="quant", created_at=COMMIT
    )
    result_write = VerificationResultWrite(
        verification_id=result_id, claim_id=claim_terminal, provider="quant",
        result=result, available_at=COMMIT,
    )
    kwargs = dict(
        source_type="fixture", source_ref=source_ref, source_content_hash=source_ref,
        artifact_ids={"verification": verification.artifact_id}, code_sha="sha", created_at=COMMIT,
        verification_jobs=(job,), verification_results=(result_write,), failure_hook=failure_hook,
    )
    return snapshots, jobs, job_id, result_id, kwargs


@pytest.mark.parametrize("failure_point", ["job", "result"])
def test_initial_verification_bundle_rollback_includes_planned_rows(tmp_path, failure_point):
    database = Database(f"sqlite:///{tmp_path / (failure_point + '.db')}")
    database.create_schema()
    snapshots, jobs, job_id, result_id, kwargs = _bundle(
        database, source_ref="atomic-" + failure_point,
        failure_hook=lambda point: (_ for _ in ()).throw(RuntimeError("injected"))
        if point == failure_point else None,
    )
    with pytest.raises(RuntimeError, match="injected"):
        snapshots.record_bundle_from_artifacts(**kwargs)
    assert snapshots.list_for_source("fixture", "atomic-" + failure_point) == []
    assert jobs.get_job(job_id) is None
    assert jobs.get_result(result_id) is None


def test_initial_verification_bundle_success_closes_exact_job_and_result(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'success.db'}")
    database.create_schema()
    snapshots, jobs, job_id, result_id, kwargs = _bundle(database, source_ref="atomic-success")
    snapshot = snapshots.record_bundle_from_artifacts(**kwargs)
    assert snapshots.get(snapshot.content_snapshot_id) is not None
    assert jobs.get_job(job_id) is not None
    assert jobs.get_result(result_id) is not None

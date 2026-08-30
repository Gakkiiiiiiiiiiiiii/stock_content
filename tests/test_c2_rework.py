from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from stock_content.adapters.postgres.models import (
    ClaimVerificationJobRow,
    ClaimVerificationResultRow,
    ContentArtifactRow,
    ContentSnapshotRow,
    ContentSourceHeadRow,
    SignalOutboxRow,
)
from stock_content.adapters.postgres.repositories import (
    PostgresVerificationJobRepository,
    SqlClaimRepository,
)
from stock_content.api.dependencies import build_application
from stock_content.api.main import create_app
from stock_content.application.verification_refresh import VerificationRefreshService
from stock_content.application.verification_worker import VerificationWorkerApplication
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claims import VerificationResult
from stock_content.domain.signal_policy import SignalPolicy
from stock_content.workers.signal_publisher_worker import HttpSignalPublisher


class _Provider:
    def verify(self, claim):
        return VerificationResult(
            claim_id=claim.claim_id,
            status="VERIFIED",
            market_snapshot_id="m-prod",
            market_data_version="bars.v1",
            fact_date=datetime(2026, 1, 1, tzinfo=UTC).date(),
            adjustment="NONE",
            verification_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_production_worker_uses_atomic_refresh_and_claim_specific_parent(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'worker.db'}", enable_qdrant=False)
    app.enqueue(
        "bilibili",
        "BVworker",
        {"metadata": {"title": "worker"}, "transcript": "600000 盈利增长。", "offline_fixture": True},
    )
    ingest = app.process_next("ingest")
    assert ingest["status"] == "SUCCEEDED"
    sessions = app._pipeline._stages[0]._artifact_repository._sessions  # noqa: SLF001
    jobs = PostgresVerificationJobRepository(sessions)
    claims = SqlClaimRepository(sessions)
    refresh = VerificationRefreshService(sessions, jobs, claims)
    claim = next(iter(claims.get_many() if hasattr(claims, "get_many") else []), None)
    if claim is None:
        # Claims are available through the claim member rows; use the job row.
        job_row = jobs.claim_due("probe", now=datetime.now(UTC))[0]
        claim = claims.get(job_row.claim_id)
        jobs.requeue(job_row.job_id)
    parent = refresh.resolve_parent_snapshot_id(claim.claim_id)
    assert parent == ingest["content_snapshot_id"]
    worker = VerificationWorkerApplication(jobs, claims, _Provider(), refresh)
    result = worker.run_once("production-worker")
    assert result["completed"] == 1
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(ContentSnapshotRow)) == 2
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == 1
        assert session.scalar(select(func.count()).select_from(SignalOutboxRow)) == 1


def test_worker_completion_failure_rolls_back_atomic_uow_and_retries(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'worker-rollback.db'}", enable_qdrant=False)
    app.enqueue(
        "bilibili",
        "BVworker-rollback",
        {"metadata": {"title": "worker"}, "transcript": "600000 盈利增长。", "offline_fixture": True},
    )
    assert app.process_next("ingest")["status"] == "SUCCEEDED"
    sessions = app._pipeline._stages[0]._artifact_repository._sessions  # noqa: SLF001
    jobs = PostgresVerificationJobRepository(sessions)
    claims = SqlClaimRepository(sessions)
    refresh = VerificationRefreshService(sessions, jobs, claims)
    with sessions() as session:
        baseline_artifacts = session.scalar(select(func.count()).select_from(ContentArtifactRow))
        baseline_snapshots = session.scalar(select(func.count()).select_from(ContentSnapshotRow))

    class FailingCompletion:
        def resolve_parent_snapshot_id(self, claim_id):
            return refresh.resolve_parent_snapshot_id(claim_id)

        def complete(self, job_id, worker_id, result, *, parent_snapshot_id, now):
            def fail(stage):
                if stage == "outbox":
                    raise RuntimeError("injected completion failure")

            return refresh.complete(
                job_id,
                worker_id,
                result,
                parent_snapshot_id=parent_snapshot_id,
                now=now,
                failure_hook=fail,
            )

    worker = VerificationWorkerApplication(jobs, claims, _Provider(), FailingCompletion())
    result = worker.run_once("rollback-worker")
    assert result["retried"] == 1
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == 0
        assert session.scalar(select(func.count()).select_from(ContentArtifactRow)) == baseline_artifacts
        assert session.scalar(select(func.count()).select_from(ContentSnapshotRow)) == baseline_snapshots
        assert session.scalar(select(func.count()).select_from(SignalOutboxRow)) == 0
    with sessions() as session:
        job = session.scalar(select(ClaimVerificationJobRow))
        assert job is not None and job.status == "VERIFICATION_PENDING"


def test_http_publisher_idempotency_and_trace_headers(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(202, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    publisher = HttpSignalPublisher("http://factor.test/signals")
    publisher.publish({"producer": {"trace_id": "trace-1"}}, "signal-1")
    assert captured["headers"]["Idempotency-Key"] == "signal-1"
    assert captured["headers"]["X-Trace-Id"] == "trace-1"


def test_refresh_preserves_sibling_claim_results(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'multi.db'}", enable_qdrant=False)
    app.enqueue(
        "bilibili",
        "BVmulti",
        {
            "metadata": {"title": "multi"},
            "transcript": "宁德时代300750业绩增长，毛利率改善。这是明确利好。",
            "offline_fixture": True,
        },
    )
    initial = app.process_next("ingest")
    sessions = app._pipeline._stages[0]._artifact_repository._sessions  # noqa: SLF001
    jobs = PostgresVerificationJobRepository(sessions)
    claims = SqlClaimRepository(sessions)
    refresh = VerificationRefreshService(sessions, jobs, claims)
    parent = app._snapshots.get(initial["content_snapshot_id"])
    artifact_repo = app._pipeline._stages[0]._artifact_repository  # noqa: SLF001
    prior = artifact_repo.get(parent.artifact_ids["verification"])
    job = jobs.claim_due("multi-worker")[0]
    now = datetime.now(UTC)
    result = VerificationResult(
        claim_id=job.claim_id,
        status="VERIFIED",
        market_snapshot_id="multi-market",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    refreshed = refresh.complete(job.job_id, "multi-worker", result, parent_snapshot_id=parent.content_snapshot_id)
    new_snapshot = app._snapshots.get(refreshed["snapshot_id"])
    new_artifact = artifact_repo.get(new_snapshot.artifact_ids["verification"])
    assert {item.claim_id for item in new_artifact.results} == {item.claim_id for item in prior.results}


def test_stale_parent_refresh_merges_against_locked_current_head(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'stale-parent.db'}", enable_qdrant=False)
    app.enqueue(
        "bilibili",
        "BVstale-parent",
        {
            "metadata": {"title": "stale-parent"},
            "transcript": "宁德时代300750业绩增长，毛利率改善。这是明确利好。",
            "offline_fixture": True,
        },
    )
    initial = app.process_next("ingest")
    sessions = app._pipeline._stages[0]._artifact_repository._sessions  # noqa: SLF001
    jobs = PostgresVerificationJobRepository(sessions)
    claims = SqlClaimRepository(sessions)
    refresh = VerificationRefreshService(sessions, jobs, claims)
    parent_id = initial["content_snapshot_id"]
    leased_one = jobs.claim_due("worker-one", limit=1)[0]
    leased_two = jobs.claim_due("worker-two", limit=1)[0]
    first_time = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    second_time = datetime(2026, 1, 1, 9, 1, tzinfo=UTC)
    first = VerificationResult(
        claim_id=leased_one.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-one",
        market_data_version="bars.v1",
        fact_date=first_time.date(),
        adjustment="NONE",
        verification_timestamp=first_time,
    )
    second = VerificationResult(
        claim_id=leased_two.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-two",
        market_data_version="bars.v1",
        fact_date=second_time.date(),
        adjustment="NONE",
        verification_timestamp=second_time,
    )
    first_refresh = refresh.complete(
        leased_one.job_id,
        "worker-one",
        first,
        parent_snapshot_id=parent_id,
        now=first_time,
    )
    second_refresh = refresh.complete(
        leased_two.job_id,
        "worker-two",
        second,
        # Deliberately stale: the service must use the locked source head.
        parent_snapshot_id=parent_id,
        now=second_time,
    )
    assert second_refresh["snapshot_id"] != first_refresh["snapshot_id"]
    final = app._snapshots.get(second_refresh["snapshot_id"])
    artifact = app._pipeline._stages[0]._artifact_repository.get(final.artifact_ids["verification"])
    assert {item.claim_id for item in artifact.results} == {leased_one.claim_id, leased_two.claim_id}
    with sessions() as session:
        head = session.scalar(select(ContentSourceHeadRow))
        assert head is not None
        assert head.latest_snapshot_id == second_refresh["snapshot_id"]


def test_refresh_does_not_roll_back_head_for_newer_content_without_claim(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'head-branch.db'}", enable_qdrant=False)
    app.enqueue(
        "bilibili",
        "BVhead-branch",
        {"metadata": {"title": "old"}, "transcript": "股票600000基本面良好。", "offline_fixture": True},
    )
    first = app.process_next("ingest")
    assert first["status"] == "SUCCEEDED"
    sessions = app._pipeline._stages[0]._artifact_repository._sessions  # noqa: SLF001
    artifact_repo = app._pipeline._stages[0]._artifact_repository  # noqa: SLF001
    first_snapshot = app._snapshots.get(first["content_snapshot_id"])
    first_claims = artifact_repo.get(first_snapshot.artifact_ids["claims"])
    first_claim_id = str(first_claims.claims[0])
    jobs = PostgresVerificationJobRepository(sessions)
    claims = SqlClaimRepository(sessions)
    leased = jobs.claim_due("old-content-worker", limit=1)[0]
    assert leased.claim_id == first_claim_id

    app.enqueue(
        "bilibili",
        "BVhead-branch",
        {"metadata": {"title": "new"}, "transcript": "股票600001基本面良好。", "offline_fixture": True},
    )
    second = app.process_next("ingest")
    assert second["status"] == "SUCCEEDED"
    assert second["content_snapshot_id"] != first["content_snapshot_id"]

    now = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    result = VerificationResult(
        claim_id=leased.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-old-content",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    refreshed = VerificationRefreshService(sessions, jobs, claims).complete(
        leased.job_id,
        "old-content-worker",
        result,
        parent_snapshot_id=first["content_snapshot_id"],
        now=now,
    )
    with sessions() as session:
        head = session.scalar(select(ContentSourceHeadRow))
        assert head is not None
        assert head.latest_snapshot_id == second["content_snapshot_id"]
        assert head.latest_snapshot_id != refreshed["snapshot_id"]


def test_claim_api_persists_idempotent_job_and_reads_db_first(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'claim-api.db'}", enable_qdrant=False)
    client = TestClient(create_app(app))
    payload = {
        "claim_type": "PRICE",
        "subject_type": "EQUITY",
        "subject_id": "600000",
        "predicate": "price",
        "value": 10,
        "evidence_refs": ["evidence-api"],
        "source_confidence": 0.9,
        "extractor_confidence": 0.9,
    }
    first = client.post("/api/v1/claims", headers={"X-Trace-Id": "claim-trace"}, json=payload)
    second = client.post("/api/v1/claims", headers={"X-Trace-Id": "claim-trace-2"}, json=payload)
    assert first.status_code == 200 and second.status_code == 200
    claim_id = first.json()["claim_id"]
    sessions = app._claim_repository._sessions  # noqa: SLF001
    with sessions() as session:
        jobs = list(
            session.scalars(
                select(ClaimVerificationJobRow).where(ClaimVerificationJobRow.claim_id == claim_id)
            )
        )
        assert len(jobs) == 1
        assert jobs[0].trace_id == "claim-trace"
    app._claims_registry.clear()  # DB must remain the API authority after restart-like loss of memory.
    claim = client.get(f"/api/v1/claims/{claim_id}")
    verification = client.get(f"/api/v1/claims/{claim_id}/verification")
    assert claim.status_code == 200
    assert claim.json()["data"]["verification_status"] == "VERIFICATION_PENDING"
    assert verification.status_code == 200
    assert verification.json()["data"]["job_id"] == jobs[0].job_id


def test_refresh_signal_lineage_api_contains_source_claim_evidence_and_decision(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'refresh-lineage.db'}", enable_qdrant=False)
    client = TestClient(create_app(app))
    app.enqueue(
        "bilibili",
        "BVrefresh-lineage",
        {
            "metadata": {"title": "refresh-lineage"},
            "transcript": "股票600000基本面良好。",
            "offline_fixture": True,
            "trace_id": "refresh-trace",
        },
    )
    initial = app.process_next("ingest")
    assert initial["status"] == "SUCCEEDED"
    sessions = app._claim_repository._sessions  # noqa: SLF001
    jobs = PostgresVerificationJobRepository(sessions)
    claims = SqlClaimRepository(sessions)
    job = jobs.claim_due("refresh-lineage-worker")[0]
    now = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    result = VerificationResult(
        claim_id=job.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-lineage",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    refreshed = VerificationRefreshService(sessions, jobs, claims).complete(
        job.job_id,
        "refresh-lineage-worker",
        result,
        parent_snapshot_id=initial["content_snapshot_id"],
        now=now,
    )
    # Persist a later/source-external occurrence for the same claim, but do
    # not attach it to this snapshot's occurrence/evidence artifacts.  The
    # endpoint must never return it through a global/latest lookup.
    future_ids = {"evidence-from-future-source"}
    artifact_repo = app._pipeline._stages[0]._artifact_repository  # noqa: SLF001
    current_snapshot = app._snapshots.get(refreshed["snapshot_id"])
    exact_evidence = artifact_repo.get(current_snapshot.artifact_ids["evidence"])
    exact_occurrences = artifact_repo.get(current_snapshot.artifact_ids["occurrences"])
    exact_ids = {item.evidence_id for item in exact_evidence.evidences}
    assert exact_ids.isdisjoint(future_ids)
    source_occurrence = app._occurrence_repository.get(exact_occurrences.occurrence_ids[0])  # noqa: SLF001
    external_occurrence = ClaimOccurrence.model_validate({**source_occurrence.model_dump(),
        "occurrence_id": "", "source_artifact_id": "future-source-artifact",
        "evidence_refs": sorted(future_ids), "condition_evidence_refs": [],
        "invalidation_evidence_refs": [], "temporal_evidence_refs": [],
        "assertion_locator_hash": "",
    })
    app._occurrence_repository.save(external_occurrence)  # noqa: SLF001
    signal_id = str(refreshed["signal_id"])
    response = client.get(f"/api/v1/signals/{signal_id}/lineage")
    assert response.status_code == 200
    data = response.json()["data"]
    signal = data["signal"]
    assert signal["decision_id"] == signal["producer"]["decision_id"]
    assert data["snapshot"]["content_snapshot_id"] == refreshed["snapshot_id"]
    assert data["claim"]["claim_id"] == job.claim_id
    assert data["evidence_ids"]
    assert set(data["evidence_ids"]) <= exact_ids
    assert not set(data["evidence_ids"]) & future_ids
    assert data["source"]["artifact_type"] == "source"


def test_invalid_refresh_signal_contract_rolls_back_complete_uow(tmp_path):
    app = build_application(f"sqlite:///{tmp_path / 'invalid-signal.db'}", enable_qdrant=False)
    app.enqueue(
        "bilibili",
        "BVinvalid-signal",
        {"metadata": {"title": "invalid-signal"}, "transcript": "股票600000基本面良好。", "offline_fixture": True},
    )
    initial = app.process_next("ingest")
    sessions = app._claim_repository._sessions  # noqa: SLF001
    jobs = PostgresVerificationJobRepository(sessions)
    claims = SqlClaimRepository(sessions)
    job = jobs.claim_due("invalid-signal-worker")[0]
    with sessions() as session:
        before_snapshots = session.scalar(select(func.count()).select_from(ContentSnapshotRow))
        before_results = session.scalar(select(func.count()).select_from(ClaimVerificationResultRow))
        before_outbox = session.scalar(select(func.count()).select_from(SignalOutboxRow))

    class InvalidSignalService:
        policy = SignalPolicy()

        @staticmethod
        def build_signal(*args, **kwargs):
            return {"signal_schema_version": "content-factor-signal.v4"}

    now = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    result = VerificationResult(
        claim_id=job.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-invalid-signal",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    with pytest.raises(ValueError, match="v4 signal missing"):
        VerificationRefreshService(
            sessions, jobs, claims, signal_service=InvalidSignalService()
        ).complete(
            job.job_id,
            "invalid-signal-worker",
            result,
            parent_snapshot_id=initial["content_snapshot_id"],
            now=now,
        )
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(ContentSnapshotRow)) == before_snapshots
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == before_results
        assert session.scalar(select(func.count()).select_from(SignalOutboxRow)) == before_outbox

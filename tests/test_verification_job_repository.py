from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    MANUAL_REVIEW,
    PENDING,
    PostgresVerificationJobRepository,
    VerificationJobIntegrityError,
)
from stock_content.application.verification_worker import VerificationWorkerApplication
from stock_content.domain.claims import FinancialClaim, VerificationResult


def _claim(
    claim_type: str = "PRICE", suffix: str = "one", source_support_status: str = "UNSUPPORTED"
) -> FinancialClaim:
    return FinancialClaim(
        claim_type=claim_type,
        subject_type="EQUITY",
        subject_id=f"600000-{suffix}",
        predicate="price",
        value=10,
        evidence_refs=[f"evidence-{suffix}"],
        source_support_status=source_support_status,
        source_confidence=0.9,
        extractor_confidence=0.9,
    )


def _repos(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'verification.db'}")
    database.create_schema()
    return database, SqlClaimRepository(database.session_factory), PostgresVerificationJobRepository(
        database.session_factory
    )


def test_enqueue_is_idempotent_and_quant_eligible_only(tmp_path):
    _, claims, jobs = _repos(tmp_path)
    eligible = _claim()
    forecast = _claim("FORECAST", "forecast")
    inference = _claim("INFERENCE", "inference")
    for claim in (eligible, forecast, inference):
        claims.save(claim)
    assert jobs.enqueue([eligible, forecast, inference], now=datetime(2026, 1, 1, tzinfo=UTC))
    assert jobs.enqueue([eligible, forecast, inference]) == []
    assert jobs.get_job(f"verify-job-{eligible.claim_id}-quant") is not None
    assert jobs.get_job(f"verify-job-{forecast.claim_id}-quant") is None
    assert jobs.get_job(f"verify-job-{inference.claim_id}-quant") is None
    assert jobs.read_result(forecast.claim_id).status == "NOT_REQUIRED"
    assert jobs.read_result(inference.claim_id).status == "NOT_REQUIRED"
    corporate = _claim("CORPORATE_EVENT", "corporate")
    claims.save(corporate)
    jobs.enqueue([corporate])
    assert jobs.read_result(corporate.claim_id).status == "NOT_VERIFIABLE"


def test_legacy_partial_source_support_normalizes_to_canonical_value():
    claim = _claim(source_support_status="PARTIAL")
    assert claim.source_support_status == "PARTIALLY_SUPPORTED"


def test_lease_exclusion_expiry_reclaim_and_owner_validation(tmp_path):
    _, claims, jobs = _repos(tmp_path)
    claim = _claim()
    claims.save(claim)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    jobs.enqueue([claim], now=start)
    first = jobs.claim_due("worker-a", now=start, lease_seconds=60)
    assert len(first) == 1
    assert jobs.claim_due("worker-b", now=start + timedelta(seconds=30)) == []
    with pytest.raises(VerificationJobIntegrityError):
        jobs.renew_lease(first[0].job_id, "worker-b", now=start + timedelta(seconds=30))
    reclaimed = jobs.claim_due("worker-b", now=start + timedelta(seconds=61))
    assert len(reclaimed) == 1


def test_retry_schedule_restart_durability_and_manual_review(tmp_path):
    database, claims, jobs = _repos(tmp_path)
    claim = _claim()
    claims.save(claim)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    jobs.enqueue([claim], now=now)
    expected = (60, 300, 1800, 7200, 43200)
    for delay in expected:
        leased = jobs.claim_due("worker", now=now)
        assert leased
        updated = jobs.mark_retry(leased[0].job_id, "worker", "provider down", now=now)
        assert updated.status == PENDING
        assert updated.next_retry_at == now + timedelta(seconds=delay)
        now = updated.next_retry_at
    leased = jobs.claim_due("worker", now=now)
    updated = jobs.mark_retry(leased[0].job_id, "worker", "still down", now=now)
    assert updated.status == MANUAL_REVIEW
    # A fresh repository instance sees the durable state.
    restarted = PostgresVerificationJobRepository(database.session_factory)
    assert restarted.get_job(updated.job_id).status == MANUAL_REVIEW


def test_result_snapshot_binding_and_duplicate_idempotency(tmp_path):
    _, claims, jobs = _repos(tmp_path)
    claim = _claim()
    claims.save(claim)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    jobs.enqueue([claim], now=now)
    job = jobs.claim_due("worker", now=now)[0]
    with pytest.raises(ValueError):
        VerificationResult(claim_id=claim.claim_id, status="VERIFIED")
    result = VerificationResult(
        claim_id=claim.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-snapshot-1",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    first = jobs.complete_result(job.job_id, "worker", result, now)
    second = jobs.complete_result(job.job_id, "other-worker", result, now)
    assert first.verification_id == second.verification_id
    assert jobs.read_result(claim.claim_id) == result


def test_worker_recovers_after_provider_failure(tmp_path):
    _, claims, jobs = _repos(tmp_path)
    claim = _claim()
    claims.save(claim)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    jobs.enqueue([claim], now=now)

    class RecoveringProvider:
        def __init__(self):
            self.calls = 0

        def verify(self, claim):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary outage")
            return {
                "status": "MATCH",
                "market_fact": {
                    "data_snapshot_id": "market-1",
                    "data_version": "bars.v1",
                    "trading_date": "2026-01-01",
                    "close": 10,
                },
            }

    provider = RecoveringProvider()
    worker = VerificationWorkerApplication(jobs, claims, provider)
    first = worker.run_once("worker", now=now)
    assert first["retried"] == 1
    second = worker.run_once("worker", now=now + timedelta(minutes=1))
    assert second["completed"] == 1
    assert jobs.read_result(claim.claim_id) is not None


def test_worker_treats_provider_pending_as_durable_retry(tmp_path):
    _, claims, jobs = _repos(tmp_path)
    claim = _claim()
    claims.save(claim)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    jobs.enqueue([claim], now=now)

    class PendingProvider:
        def verify(self, claim):
            return {"status": "PENDING", "reason": "provider unavailable"}

    worker = VerificationWorkerApplication(jobs, claims, PendingProvider())
    result = worker.run_once("worker", now=now)
    assert result["retried"] == 1
    assert jobs.read_result(claim.claim_id) is None
    assert jobs.get_job(f"verify-job-{claim.claim_id}-quant").status == PENDING


def test_production_worker_entry_does_not_use_memory_as_authority():
    source = Path("src/stock_content/workers/verification_worker.py").read_text(encoding="utf-8")
    assert "VerificationService(provider=None)" not in source
    assert "run_db_once" in source

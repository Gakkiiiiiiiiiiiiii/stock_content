from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.initial_verification import (
    VerificationRequirement,
    build_initial_verification_plan,
    verification_requirement,
    verify_initial_verification_closure,
)


def _claim(claim_type: str) -> FinancialClaim:
    return FinancialClaim(
        claim_type=claim_type,
        subject_type="EQUITY",
        subject_id="600000",
        predicate="value",
        value=1,
        evidence_refs=["evidence-1"],
        source_confidence=1,
        extractor_confidence=1,
    )


class _Repository:
    def __init__(self, terminal=None, job=None):
        self.terminal = terminal
        self.job = job

    def latest_terminal_as_of(self, **kwargs):
        if self.terminal is not None and self.terminal.available_at <= kwargs["as_of"]:
            return self.terminal
        return None

    def find_active_as_of(self, **_kwargs):
        return self.job


def test_requirement_matrix_is_separate_from_snapshot_state():
    expected = {
        "PRICE": VerificationRequirement.ASYNC_REQUIRED,
        "RETURN": VerificationRequirement.ASYNC_REQUIRED,
        "VALUATION": VerificationRequirement.ASYNC_REQUIRED,
        "FINANCIAL_METRIC": VerificationRequirement.ASYNC_REQUIRED,
        "CORPORATE_EVENT": VerificationRequirement.NOT_VERIFIABLE,
        "INDUSTRY_RELATION": VerificationRequirement.NOT_VERIFIABLE,
        "FORECAST": VerificationRequirement.NOT_REQUIRED,
        "OPINION": VerificationRequirement.NOT_REQUIRED,
        "INFERENCE": VerificationRequirement.NOT_REQUIRED,
    }
    assert {kind: verification_requirement(_claim(kind)) for kind in expected} == expected


def test_active_job_is_reused_and_future_terminal_does_not_leak():
    claim = _claim("PRICE")
    candidate = datetime(2026, 1, 1, tzinfo=UTC)
    job = SimpleNamespace(
        job_id="job-1", claim_id=claim.claim_id, provider="quant",
        status="LEASED", created_at=candidate - timedelta(seconds=1),
    )
    repo = _Repository(job=job)
    plan = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=candidate,
        verification_repository=repo, job_repository=repo,
    )
    assert plan.reused_job_ids == ["job-1"]
    assert plan.artifact_results[0].verification_job_id == "job-1"

    future = SimpleNamespace(
        verification_id="v-future", claim_id=claim.claim_id, provider="quant",
        status="VERIFIED", available_at=candidate + timedelta(seconds=1),
        result_payload=VerificationResult(
            claim_id=claim.claim_id, status="VERIFIED", market_snapshot_id="m",
            market_data_version="v", fact_date=candidate.date(),
            verification_timestamp=candidate,
        ).model_dump(mode="json"),
    )
    no_future = _Repository(terminal=future)
    plan = build_initial_verification_plan(
        claims=[claim], provider="quant", snapshot_candidate_time=candidate,
        verification_repository=no_future, job_repository=_Repository(),
    )
    assert plan.pending_jobs_to_insert


def test_historical_pending_closure_ignores_current_job_status():
    candidate = datetime(2026, 1, 1, tzinfo=UTC)
    entry = {
        "claim_id": "claim-1", "provider": "quant", "status": "VERIFICATION_PENDING",
        "verification_job_id": "job-1",
    }
    job = SimpleNamespace(
        job_id="job-1", claim_id="claim-1", provider="quant",
        status="COMPLETED", created_at=candidate,
    )
    verify_initial_verification_closure(
        artifact_results=[entry], jobs={"job-1": job}, results={},
        snapshot_committed_at=candidate,
    )

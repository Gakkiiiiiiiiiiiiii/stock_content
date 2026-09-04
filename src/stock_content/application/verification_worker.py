"""Database-backed verification worker application."""
from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from stock_content.adapters.postgres.repositories.verification_job_repository import (
    PostgresVerificationJobRepository,
)
from stock_content.application.task_lease_service import TaskLeaseService
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.ports.repositories import ClaimRepository


class VerificationProvider(Protocol):
    def verify(self, claim: FinancialClaim) -> VerificationResult | dict[str, Any]: ...


class VerificationWorkerApplication:
    """Claim jobs from PostgreSQL, call the provider, and persist outcomes."""

    def __init__(
        self,
        jobs: PostgresVerificationJobRepository,
        claims: ClaimRepository,
        provider: VerificationProvider | None,
        completion_service: Any | None = None,
        task_lease_service: TaskLeaseService | None = None,
    ) -> None:
        self._jobs = jobs
        self._claims = claims
        self._provider = provider
        self._completion = completion_service
        self._task_lease = task_lease_service

    def run_once(
        self,
        worker_id: str,
        limit: int = 10,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> dict[str, int]:
        current = now or datetime.now(UTC)
        jobs = self._jobs.claim_due(worker_id, limit, lease_seconds, current)
        completed = 0
        retried = 0
        manual_review = 0
        for job in jobs:
            task_run = None
            if self._task_lease is not None:
                task_run_id = f"verification:{job.job_id}"
                if self._task_lease.repository.get(task_run_id) is None:
                    self._task_lease.create("verification", task_run_id)
                task_run = self._task_lease.acquire(
                    task_run_id, worker_id, now=current, ttl=timedelta(seconds=lease_seconds)
                )
            claim = self._claims.get(job.claim_id)
            if claim is None:
                self._jobs.mark_manual_review(job.job_id, worker_id, "CLAIM_NOT_FOUND", current)
                manual_review += 1
                continue
            try:
                if self._provider is None:
                    raise RuntimeError("verification provider unavailable")
                outcome = self._invoke_provider(claim, job.trace_id)
                result = self._to_result(claim, outcome, current)
                # A provider timeout/no-data response is not a terminal
                # verification result.  Leave the job in the durable retry
                # state; only terminal outcomes enter the refresh UoW.
                if result.status == "VERIFICATION_PENDING":
                    raise RuntimeError("verification provider returned pending")
                if self._completion is not None:
                    parent_snapshot_id = self._completion.resolve_parent_snapshot_id(job.claim_id)
                    self._completion.complete(
                        job.job_id,
                        worker_id,
                        result,
                        parent_snapshot_id=parent_snapshot_id,
                        now=current,
                    )
                else:
                    self._jobs.complete_result(job.job_id, worker_id, result, current)
                if task_run is not None:
                    self._task_lease.transition(
                        task_run.task_run_id, "SUCCEEDED", worker_id, task_run.fencing_token, now=current
                    )
                completed += 1
            except Exception as exc:  # provider failures do not block ingest
                updated = self._jobs.mark_retry(job.job_id, worker_id, str(exc), current)
                if task_run is not None:
                    self._task_lease.transition(
                        task_run.task_run_id, "FAILED", worker_id, task_run.fencing_token, now=current
                    )
                if updated.status == "MANUAL_REVIEW":
                    manual_review += 1
                else:
                    retried += 1
        return {
            "claimed": len(jobs),
            "completed": completed,
            "retried": retried,
            "manual_review": manual_review,
        }

    def _invoke_provider(self, claim: FinancialClaim, trace_id: str | None) -> Any:
        if self._provider is None:
            raise RuntimeError("verification provider unavailable")
        verify = self._provider.verify
        parameters = inspect.signature(verify).parameters
        payload = claim.model_dump(mode="json")
        if "trace_id" in parameters:
            return verify(payload, trace_id=trace_id)
        # QuantExternalFactProvider consumes a mapping; injected domain
        # providers consume FinancialClaim.  The parameter name is the
        # explicit adapter boundary, not a cross-service import.
        first = next(iter(parameters.values()), None)
        if first is not None and first.name in {"unit", "payload", "data"}:
            return verify(payload)
        return verify(claim)

    @staticmethod
    def _to_result(
        claim: FinancialClaim, outcome: VerificationResult | dict[str, Any], now: datetime
    ) -> VerificationResult:
        if isinstance(outcome, VerificationResult):
            if outcome.available_at is None:
                return outcome.model_copy(update={"available_at": now})
            return outcome
        payload = dict(outcome or {})
        status = str(payload.get("status") or "NOT_VERIFIABLE").upper()
        status = {
            "MATCH": "VERIFIED",
            "CONFLICT": "CONTRADICTED",
            "PARTIAL": "PARTIALLY_VERIFIED",
            "PENDING": "VERIFICATION_PENDING",
            "NOT_FOUND": "NOT_VERIFIABLE",
        }.get(status, status)
        market = dict(payload.get("market_fact") or payload.get("market_snapshot") or {})
        fact_date = market.get("trading_date") or payload.get("fact_date")
        return VerificationResult(
            claim_id=claim.claim_id,
            status=status,
            market_snapshot_id=market.get("data_snapshot_id") or market.get("snapshot_id"),
            market_data_version=market.get("data_version") or market.get("market_data_version"),
            fact_date=fact_date,
            adjustment=market.get("adjustment", "NONE"),
            verification_timestamp=now,
            verification_rule_version=str(
                payload.get("verification_rule_version") or "verification_rule.v1"
            ),
            available_at=now,
            reference_value=market.get("close"),
            reason=payload.get("reason"),
        )


__all__ = ["VerificationProvider", "VerificationWorkerApplication"]

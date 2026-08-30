"""Deterministic initial verification planning.

This module only contains the requirement/state policy and the small write
plan.  Database repositories provide the PIT queries and persistence hooks;
the planner deliberately does not read a "latest" mutable result.
"""
from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterator

from .claims import (
    FinancialClaim,
    VerificationArtifactEntry,
    VerificationResult,
    is_quant_verifiable,
)


class VerificationRequirement(str, Enum):
    ASYNC_REQUIRED = "ASYNC_REQUIRED"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"
    NOT_REQUIRED = "NOT_REQUIRED"


def verification_requirement(claim: FinancialClaim) -> VerificationRequirement:
    if is_quant_verifiable(claim):
        return VerificationRequirement.ASYNC_REQUIRED
    if claim.fact_category == "FACT":
        return VerificationRequirement.NOT_VERIFIABLE
    return VerificationRequirement.NOT_REQUIRED


TERMINAL_STATUSES = frozenset(
    {
        "VERIFIED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE",
        "NOT_REQUIRED", "EXPIRED", "MANUAL_REVIEW",
    }
)
ACTIVE_JOB_STATUSES = frozenset({"VERIFICATION_PENDING", "PENDING", "LEASED", "RETRYABLE"})


@dataclass(frozen=True)
class VerificationJobWrite:
    job_id: str
    claim_id: str
    provider: str
    status: str = "VERIFICATION_PENDING"
    created_at: datetime | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class VerificationResultWrite:
    verification_id: str
    claim_id: str
    provider: str
    result: VerificationResult
    available_at: datetime


@dataclass(frozen=True)
class ResolvedInitialVerificationState:
    requirement: VerificationRequirement
    artifact_status: str
    terminal_result: VerificationResultWrite | None = None
    pending_job: VerificationJobWrite | None = None
    reused_terminal_result_id: str | None = None
    reused_job_id: str | None = None


@dataclass
class InitialVerificationPersistencePlan:
    artifact_results: list[VerificationArtifactEntry] = field(default_factory=list)
    terminal_results_to_insert: list[VerificationResultWrite] = field(default_factory=list)
    pending_jobs_to_insert: list[VerificationJobWrite] = field(default_factory=list)
    reused_terminal_result_ids: list[str] = field(default_factory=list)
    reused_job_ids: list[str] = field(default_factory=list)


def _utc(value: datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def deterministic_terminal_result(
    claim: FinancialClaim,
    requirement: VerificationRequirement,
    *,
    available_at: datetime,
    provider: str,
    policy_version: str,
) -> VerificationResultWrite:
    status = (
        "NOT_VERIFIABLE" if requirement is VerificationRequirement.NOT_VERIFIABLE else "NOT_REQUIRED"
    )
    result = VerificationResult(
        claim_id=claim.claim_id,
        status=status,
        reason=("FACT_NOT_QUANT_VERIFIABLE" if status == "NOT_VERIFIABLE" else "NON_FACT_CLAIM"),
        verification_rule_version=policy_version,
        available_at=_utc(available_at),
    )
    return VerificationResultWrite(
        verification_id=verification_id_of(claim.claim_id, provider, result),
        claim_id=claim.claim_id,
        provider=provider,
        result=result,
        available_at=_utc(available_at),
    )


def verification_id_of(claim_id: str, provider: str, result: VerificationResult) -> str:
    # ``available_at`` is an envelope, not semantic result identity.  This
    # keeps deterministic IDs reusable by later snapshots.
    payload = result.model_dump(mode="json")
    payload.pop("available_at", None)
    identity = {"claim_id": claim_id, "provider": provider, "result": payload}
    raw = _canonical(identity)
    return "verification-" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _canonical(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class VerificationCoordination:
    """Process lock fallback; SQL repositories add transaction locks on PG."""

    _locks: dict[str, threading.RLock] = {}
    _guard = threading.Lock()

    @classmethod
    @contextmanager
    def lock(cls, claim_id: str, provider: str) -> Iterator[None]:
        key = f"{claim_id}:{provider}"
        with cls._guard:
            lock = cls._locks.setdefault(key, threading.RLock())
        with lock:
            yield


class InitialVerificationStateResolver:
    def resolve(
        self,
        *,
        claim: FinancialClaim,
        provider: str,
        snapshot_candidate_time: datetime,
        current_policy_version: str,
        verification_repository: Any,
        job_repository: Any,
        coordination: Any = VerificationCoordination,
        session: Any = None,
    ) -> ResolvedInitialVerificationState:
        candidate = _utc(snapshot_candidate_time)
        requirement = verification_requirement(claim)
        if requirement is not VerificationRequirement.ASYNC_REQUIRED:
            terminal = deterministic_terminal_result(
                claim, requirement, available_at=candidate, provider=provider,
                policy_version=current_policy_version,
            )
            existing = _latest_terminal(
                verification_repository, claim.claim_id, provider, candidate, current_policy_version,
                session=session,
            )
            if existing is not None and _result_status(existing) == terminal.result.status:
                return ResolvedInitialVerificationState(
                    requirement,
                    terminal.result.status,
                    reused_terminal_result_id=_row_id(existing, terminal.verification_id),
                )
            return ResolvedInitialVerificationState(requirement, terminal.result.status, terminal_result=terminal)

        lock_factory = getattr(verification_repository, "coordination_lock", coordination.lock)
        try:
            lock = lock_factory(claim.claim_id, provider, session=session)
        except TypeError:
            lock = lock_factory(claim.claim_id, provider)
        with lock:
            existing = _latest_terminal(
                verification_repository, claim.claim_id, provider, candidate, current_policy_version,
                session=session,
            )
            if existing is not None:
                return ResolvedInitialVerificationState(
                    requirement, _result_status(existing), reused_terminal_result_id=_row_id(existing, None)
                )
            job = _active_job(job_repository, claim.claim_id, provider, candidate, session=session)
            if job is None:
                job_id = f"verify-job-{claim.claim_id}-{provider}"
                job = VerificationJobWrite(
                    job_id=job_id,
                    claim_id=claim.claim_id,
                    provider=provider,
                    created_at=candidate,
                )
                return ResolvedInitialVerificationState(requirement, "VERIFICATION_PENDING", pending_job=job)
            return ResolvedInitialVerificationState(
                requirement, "VERIFICATION_PENDING", reused_job_id=_row_id(job, getattr(job, "job_id", None))
            )


def build_initial_verification_plan(
    *,
    claims: list[FinancialClaim],
    provider: str,
    snapshot_candidate_time: datetime,
    verification_repository: Any,
    job_repository: Any,
    coordination: Any = VerificationCoordination,
    current_policy_version: str = "verification_rule.v1",
    trace_id: str | None = None,
    session: Any = None,
) -> InitialVerificationPersistencePlan:
    """Resolve every claim and return one transaction-ready write plan."""
    plan = InitialVerificationPersistencePlan()
    resolver = InitialVerificationStateResolver()
    for claim in claims:
        state = resolver.resolve(
            claim=claim,
            provider=provider,
            snapshot_candidate_time=snapshot_candidate_time,
            current_policy_version=current_policy_version,
            verification_repository=verification_repository,
            job_repository=job_repository,
            coordination=coordination,
            session=session,
        )
        if state.terminal_result is not None:
            plan.terminal_results_to_insert.append(state.terminal_result)
            plan.artifact_results.append(
                VerificationArtifactEntry.from_result(
                    state.terminal_result.result,
                    provider=provider,
                    verification_id=state.terminal_result.verification_id,
                )
            )
        elif state.reused_terminal_result_id:
            row = _find_terminal(
                verification_repository, claim.claim_id, provider,
                snapshot_candidate_time, current_policy_version, state.reused_terminal_result_id,
                session=session,
            )
            result = _as_result(row, claim.claim_id, state.artifact_status)
            plan.reused_terminal_result_ids.append(state.reused_terminal_result_id)
            plan.artifact_results.append(
                VerificationArtifactEntry.from_result(
                    result, provider=provider, verification_id=state.reused_terminal_result_id,
                )
            )
        else:
            job_id = state.pending_job.job_id if state.pending_job else state.reused_job_id
            if not job_id:
                raise ValueError(f"initial verification state has no job lineage for {claim.claim_id}")
            if state.pending_job is not None:
                plan.pending_jobs_to_insert.append(
                    VerificationJobWrite(
                        job_id=state.pending_job.job_id, claim_id=claim.claim_id,
                        provider=provider, status="VERIFICATION_PENDING",
                        created_at=_utc(state.pending_job.created_at or snapshot_candidate_time),
                        trace_id=trace_id,
                    )
                )
            else:
                plan.reused_job_ids.append(job_id)
            pending = VerificationResult(claim_id=claim.claim_id, status="VERIFICATION_PENDING")
            plan.artifact_results.append(
                VerificationArtifactEntry.from_result(
                    pending, provider=provider, verification_job_id=job_id,
                )
            )
    return plan


def _latest_terminal(repository, claim_id, provider, as_of, policy_version, *, session=None):
    method = getattr(repository, "latest_terminal_as_of", None)
    if method is None:
        return None
    try:
        return method(
            claim_id=claim_id, provider=provider, as_of=as_of,
            policy_version=policy_version, session=session,
        )
    except TypeError:
        return method(
            claim_id=claim_id, provider=provider, as_of=as_of,
            policy_version=policy_version,
        )


def _find_terminal(repository, claim_id, provider, as_of, policy_version, verification_id, *, session=None):
    method = getattr(repository, "get_result", None)
    if method is not None:
        try:
            row = method(verification_id, session=session)
        except TypeError:
            row = method(verification_id)
        if row is not None:
            return row
    return _latest_terminal(
        repository, claim_id, provider, as_of, policy_version, session=session
    )


def _as_result(row, claim_id, status):
    if isinstance(row, VerificationResult):
        return row
    payload = getattr(row, "result_payload", None) if row is not None else None
    if payload is None and isinstance(row, dict):
        payload = row.get("result") or row
    payload = dict(payload or {})
    payload.setdefault("claim_id", claim_id)
    payload.setdefault("status", status)
    return VerificationResult.model_validate(payload)


def _active_job(repository, claim_id, provider, as_of, *, session=None):
    method = getattr(repository, "find_active_as_of", None)
    if method is None:
        return None
    try:
        return method(claim_id=claim_id, provider=provider, as_of=as_of, session=session)
    except TypeError:
        return method(claim_id=claim_id, provider=provider, as_of=as_of)


def _row_id(row, fallback):
    if isinstance(row, dict):
        return str(row.get("verification_id") or row.get("job_id") or fallback or "")
    return str(getattr(row, "verification_id", None) or getattr(row, "job_id", None) or fallback or "")


def _result_status(row):
    if isinstance(row, VerificationResult):
        return row.status
    if isinstance(row, dict):
        return str(row.get("status") or (row.get("result") or {}).get("status"))
    return str(getattr(row, "status", ""))


def verify_initial_verification_closure(
    *,
    artifact_results: list[VerificationArtifactEntry] | tuple[VerificationArtifactEntry, ...],
    jobs: dict[str, Any] | None = None,
    results: dict[str, Any] | None = None,
    snapshot_committed_at: datetime,
) -> None:
    """Validate the immutable verification references before snapshot commit.

    ``jobs`` and ``results`` contain rows already visible in the current
    transaction (including rows staged by the bundle).  Legacy bare result
    entries remain readable, but every new lineage entry must point at the
    exact durable row it observed.  Current job status is deliberately not
    inspected here: a historical PENDING artifact remains closed after its
    job completes.
    """
    candidate = _utc(snapshot_committed_at)
    jobs = jobs or {}
    results = results or {}
    for raw_entry in artifact_results:
        entry = (
            VerificationArtifactEntry.model_validate(raw_entry)
            if isinstance(raw_entry, dict) else raw_entry
        )
        claim_id = str(entry.claim_id)
        if entry.status == "VERIFICATION_PENDING":
            if not entry.verification_job_id:
                raise ValueError(f"pending verification artifact missing job lineage: {claim_id}")
            row = jobs.get(str(entry.verification_job_id))
            if row is None:
                raise ValueError(f"verification job row missing: {entry.verification_job_id}")
            if str(_field(row, "claim_id")) != claim_id or str(_field(row, "provider")) != str(entry.provider):
                raise ValueError(f"verification job lineage mismatch: {entry.verification_job_id}")
            created_at = _field(row, "created_at")
            if created_at is None or _utc(created_at) > candidate:
                raise ValueError(f"verification job is not visible at snapshot: {entry.verification_job_id}")
            continue

        if entry.status not in TERMINAL_STATUSES:
            raise ValueError(f"unknown verification artifact status: {entry.status}")

        # Bare terminal VerificationResult is the legacy artifact shape.  It
        # cannot be checked against an exact row and is retained for old
        # snapshots only; all planner-produced entries carry an id.
        if not entry.verification_id:
            if entry.result is not None:
                continue
            raise ValueError(f"terminal verification artifact missing result lineage: {claim_id}")
        row = results.get(str(entry.verification_id))
        if row is None:
            raise ValueError(f"verification result row missing: {entry.verification_id}")
        if (
            str(_field(row, "claim_id")) != claim_id
            or str(_field(row, "provider")) != str(entry.provider)
            or str(_field(row, "status")) != str(entry.status)
            or (
                entry.result is not None
                and _canonical(_field(row, "result_payload") or {})
                != _canonical(entry.result.model_dump(mode="json"))
            )
        ):
            raise ValueError(f"verification result lineage mismatch: {entry.verification_id}")
        available_at = _field(row, "available_at")
        if available_at is None or _utc(available_at) > candidate:
            raise ValueError(f"verification result is not visible at snapshot: {entry.verification_id}")


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


__all__ = [
    "ACTIVE_JOB_STATUSES", "InitialVerificationPersistencePlan", "InitialVerificationStateResolver",
    "ResolvedInitialVerificationState", "TERMINAL_STATUSES", "VerificationArtifactEntry",
    "VerificationCoordination", "VerificationJobWrite", "VerificationRequirement",
    "VerificationResultWrite", "deterministic_terminal_result", "verification_id_of",
    "verification_requirement",
    "build_initial_verification_plan",
    "verify_initial_verification_closure",
]

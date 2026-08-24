"""Durable verification job lease/retry repository.

PostgreSQL uses row locks with ``SKIP LOCKED``; SQLite keeps the same
state-machine semantics for deterministic local tests.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from typing import Iterable

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ClaimVerificationJobRow,
    ClaimVerificationResultRow,
)
from stock_content.domain.claims import FinancialClaim, VerificationResult, is_quant_verifiable

RETRY_SCHEDULE_SECONDS: tuple[int, ...] = (60, 300, 1800, 7200, 43200)
PENDING = "VERIFICATION_PENDING"
LEASED = "LEASED"
MANUAL_REVIEW = "MANUAL_REVIEW"


class VerificationJobIntegrityError(ValueError):
    """Lease ownership or immutable verification result violation."""


def verification_id_of(
    claim_id: str, provider: str, result: VerificationResult | None = None
) -> str:
    identity = {"claim_id": claim_id, "provider": provider}
    if result is not None:
        identity["result"] = result.model_dump(mode="json")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:32]
    return f"verification-{digest}"


class PostgresVerificationJobRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def enqueue(
        self,
        claims: Iterable[FinancialClaim],
        provider: str = "quant",
        trace_id: str | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        """Idempotently enqueue only Quant-verifiable claims."""
        current = now or datetime.now(UTC)
        created: list[str] = []
        with self._sessions.begin() as session:
            for claim in claims:
                if not is_quant_verifiable(claim):
                    status = "NOT_VERIFIABLE" if claim.fact_category == "FACT" else "NOT_REQUIRED"
                    result = VerificationResult(claim_id=claim.claim_id, status=status)
                    verification_id = verification_id_of(claim.claim_id, provider, result)
                    payload = result.model_dump(mode="json")
                    _insert_ignore(
                        session,
                        ClaimVerificationResultRow,
                        {
                            "verification_id": verification_id,
                            "claim_id": claim.claim_id,
                            "provider": provider,
                            "status": status,
                            "result_payload": payload,
                            "trace_id": trace_id,
                            "fact_date": result.fact_date,
                            "adjustment": result.adjustment,
                            "verification_timestamp": result.verification_timestamp,
                            "verification_rule_version": result.verification_rule_version,
                            "verified_at": result.verification_timestamp,
                            "created_at": current,
                        },
                        [ClaimVerificationResultRow.verification_id],
                    )
                    existing_result = session.get(ClaimVerificationResultRow, verification_id)
                    if existing_result is None:
                        raise RuntimeError("verification result disappeared after a unique-key conflict")
                    if dict(existing_result.result_payload or {}) != payload:
                        raise VerificationJobIntegrityError(
                            f"verification id {verification_id} already stores a different result"
                        )
                    continue
                job_id = f"verify-job-{claim.claim_id}-{provider}"
                inserted = _insert_ignore(
                    session,
                    ClaimVerificationJobRow,
                    {
                        "job_id": job_id,
                        "claim_id": claim.claim_id,
                        "provider": provider,
                        "status": PENDING,
                        "retry_count": 0,
                        "max_retries": len(RETRY_SCHEDULE_SECONDS),
                        "next_retry_at": current,
                        "trace_id": trace_id,
                    },
                    [ClaimVerificationJobRow.claim_id, ClaimVerificationJobRow.provider],
                )
                row = session.scalar(
                    select(ClaimVerificationJobRow).where(
                        ClaimVerificationJobRow.claim_id == claim.claim_id,
                        ClaimVerificationJobRow.provider == provider,
                    )
                )
                if row is None:
                    raise RuntimeError("verification job disappeared after a unique-key conflict")
                if trace_id and not row.trace_id:
                    row.trace_id = trace_id
                if inserted:
                    created.append(row.job_id)
        return created

    def get_job(self, job_id: str) -> ClaimVerificationJobRow | None:
        with self._sessions() as session:
            return session.get(ClaimVerificationJobRow, job_id)

    def claim_due(
        self,
        worker_id: str,
        limit: int = 10,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> list[ClaimVerificationJobRow]:
        current = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            due = or_(
                ClaimVerificationJobRow.next_retry_at.is_(None),
                ClaimVerificationJobRow.next_retry_at <= current,
            )
            reclaimable = or_(
                ClaimVerificationJobRow.lease_expires_at.is_(None),
                ClaimVerificationJobRow.lease_expires_at <= current,
            )
            statement = (
                select(ClaimVerificationJobRow)
                .where(
                    ClaimVerificationJobRow.status.in_((PENDING, LEASED)),
                    due,
                    reclaimable,
                )
                .order_by(ClaimVerificationJobRow.created_at, ClaimVerificationJobRow.job_id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            rows = list(session.scalars(statement).all())
            expiry = current + timedelta(seconds=lease_seconds)
            for row in rows:
                row.status = LEASED
                row.lease_owner = worker_id
                row.lease_expires_at = expiry
            return rows

    def renew_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> bool:
        current = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.get(ClaimVerificationJobRow, job_id)
            self._assert_owner(row, worker_id, current)
            row.lease_expires_at = current + timedelta(seconds=lease_seconds)
        return True

    def mark_retry(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        now: datetime | None = None,
    ) -> ClaimVerificationJobRow:
        current = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.get(ClaimVerificationJobRow, job_id)
            self._assert_owner(row, worker_id, current)
            row.retry_count += 1
            row.last_error = str(error)
            row.lease_owner = None
            row.lease_expires_at = None
            if row.retry_count > row.max_retries:
                row.status = MANUAL_REVIEW
                row.next_retry_at = None
            else:
                row.status = PENDING
                delay = RETRY_SCHEDULE_SECONDS[row.retry_count - 1]
                row.next_retry_at = current + timedelta(seconds=delay)
            return row

    def mark_manual_review(
        self,
        job_id: str,
        worker_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> ClaimVerificationJobRow:
        current = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.get(ClaimVerificationJobRow, job_id)
            self._assert_owner(row, worker_id, current)
            row.status = MANUAL_REVIEW
            row.last_error = str(reason)
            row.next_retry_at = None
            row.lease_owner = None
            row.lease_expires_at = None
            return row

    def requeue(self, job_id: str, now: datetime | None = None) -> ClaimVerificationJobRow:
        """Explicitly reopen a terminal job for a fresh provider snapshot."""
        current = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.get(ClaimVerificationJobRow, job_id)
            if row is None:
                raise KeyError("verification job not found")
            row.status = PENDING
            row.next_retry_at = current
            row.lease_owner = None
            row.lease_expires_at = None
            row.last_error = None
            return row

    def complete_result(
        self,
        job_id: str,
        worker_id: str,
        result: VerificationResult,
        now: datetime | None = None,
    ) -> ClaimVerificationResultRow:
        current = now or datetime.now(UTC)
        if result.status in {PENDING, LEASED, MANUAL_REVIEW}:
            raise VerificationJobIntegrityError("completion requires a terminal verification result")
        with self._sessions.begin() as session:
            job = session.get(ClaimVerificationJobRow, job_id)
            if job is None:
                raise KeyError("verification job not found")
            if result.claim_id != job.claim_id:
                raise VerificationJobIntegrityError("result claim does not match leased job")
            verification_id = verification_id_of(job.claim_id, job.provider, result)
            payload = result.model_dump(mode="json")
            existing = session.get(ClaimVerificationResultRow, verification_id)
            if existing is not None:
                return persist_verification_result(
                    session,
                    {
                        "verification_id": verification_id,
                        "claim_id": job.claim_id,
                        "provider": job.provider,
                        "status": result.status,
                        "market_snapshot_id": result.market_snapshot_id,
                        "market_data_version": result.market_data_version,
                        "result_payload": payload,
                        "trace_id": job.trace_id,
                        "fact_date": result.fact_date,
                        "adjustment": result.adjustment,
                        "verification_timestamp": result.verification_timestamp,
                        "verification_rule_version": result.verification_rule_version,
                        "verified_at": result.verification_timestamp,
                        "created_at": current,
                    },
                )
            self._assert_owner(job, worker_id, current)
            row = persist_verification_result(
                session,
                {
                    "verification_id": verification_id,
                    "claim_id": job.claim_id,
                    "provider": job.provider,
                    "status": result.status,
                    "market_snapshot_id": result.market_snapshot_id,
                    "market_data_version": result.market_data_version,
                    "result_payload": payload,
                    "trace_id": job.trace_id,
                    "fact_date": result.fact_date,
                    "adjustment": result.adjustment,
                    "verification_timestamp": result.verification_timestamp,
                    "verification_rule_version": result.verification_rule_version,
                    "verified_at": result.verification_timestamp,
                    "created_at": current,
                },
            )
            job.status = result.status
            job.lease_owner = None
            job.lease_expires_at = None
            job.next_retry_at = None
            return row

    def read_result(
        self, claim_id: str, provider: str = "quant"
    ) -> VerificationResult | None:
        with self._sessions() as session:
            rows = session.scalars(
                select(ClaimVerificationResultRow)
                .where(
                    ClaimVerificationResultRow.claim_id == claim_id,
                    ClaimVerificationResultRow.provider == provider,
                )
                .order_by(ClaimVerificationResultRow.created_at.desc())
            ).all()
            if not rows:
                return None
            return VerificationResult.model_validate(dict(rows[0].result_payload or {}))

    @staticmethod
    def _assert_owner(row: ClaimVerificationJobRow | None, worker_id: str, now: datetime) -> None:
        if row is None:
            raise KeyError("verification job not found")
        if row.status != LEASED or row.lease_owner != worker_id:
            raise VerificationJobIntegrityError("verification job lease owner mismatch")
        if row.lease_expires_at is not None:
            lease_expires = row.lease_expires_at
            if lease_expires.tzinfo is None:
                lease_expires = lease_expires.replace(tzinfo=UTC)
            if lease_expires <= now:
                raise VerificationJobIntegrityError("verification job lease expired")


__all__ = [
    "LEASED",
    "MANUAL_REVIEW",
    "PENDING",
    "PostgresVerificationJobRepository",
    "RETRY_SCHEDULE_SECONDS",
    "VerificationJobIntegrityError",
    "persist_verification_result",
    "verification_id_of",
]


def _insert_ignore(session, model, values: dict, conflict_columns: list) -> bool:
    """Insert once without aborting the caller's transaction on a race."""
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        statement = postgres_insert(model).values(**values)
    elif dialect == "sqlite":
        statement = sqlite_insert(model).values(**values)
    else:
        try:
            with session.begin_nested():
                session.add(model(**values))
                session.flush()
            return True
        except IntegrityError:
            return False
    result = session.execute(statement.on_conflict_do_nothing(index_elements=conflict_columns))
    return result.rowcount == 1


def persist_verification_result(session, values: dict) -> ClaimVerificationResultRow:
    """Insert a deterministic result without poisoning a surrounding UoW.

    Refresh and the standalone job repository can race to persist the same
    deterministic verification.  Native ``ON CONFLICT DO NOTHING`` (or the
    dialect-neutral savepoint fallback) keeps that race from aborting the
    outer transaction.  The row is then re-read and all immutable result
    columns are checked before the caller continues.
    """
    verification_id = values["verification_id"]
    inserted = _insert_ignore(
        session,
        ClaimVerificationResultRow,
        values,
        [ClaimVerificationResultRow.verification_id],
    )
    del inserted  # The re-read is authoritative for both insert and conflict.
    row = session.get(ClaimVerificationResultRow, verification_id)
    if row is None:
        raise RuntimeError("verification result disappeared after a unique-key conflict")
    immutable_fields = (
        "claim_id",
        "provider",
        "status",
        "market_snapshot_id",
        "market_data_version",
        "result_payload",
        "fact_date",
        "adjustment",
        "verification_timestamp",
        "verification_rule_version",
        "verified_at",
    )
    for field in immutable_fields:
        if _canonical_immutable(getattr(row, field), field) != _canonical_immutable(values.get(field), field):
            raise VerificationJobIntegrityError(
                f"verification id {verification_id} already stores a different result"
            )
    return row


def _canonical_immutable(value, field: str | None = None):
    """Compare persisted SQLite/PostgreSQL temporal values consistently."""
    if field == "fact_date" and isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value

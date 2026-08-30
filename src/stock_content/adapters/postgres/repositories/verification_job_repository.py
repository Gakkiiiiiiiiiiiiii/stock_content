"""Durable verification job lease/retry repository.

PostgreSQL uses row locks with ``SKIP LOCKED``; SQLite keeps the same
state-machine semantics for deterministic local tests.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack, contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Iterable

from sqlalchemy import or_, select, text
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ClaimVerificationJobRow,
    ClaimVerificationResultRow,
)
from stock_content.domain.claims import FinancialClaim, VerificationResult, is_quant_verifiable
from stock_content.domain.initial_verification import VerificationCoordination

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
        result_payload = result.model_dump(mode="json")
        # Availability is a PIT envelope, not semantic result identity.
        result_payload.pop("available_at", None)
        identity["result"] = result_payload
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
                            "available_at": current,
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

    @contextmanager
    def planning_uow(self, keys: Iterable[tuple[str, str]]):
        """Hold all claim/provider coordination locks through publication.

        The planner must not resolve PIT state on short-lived sessions and then
        publish later: a competing worker could change the decision in between.
        This UoW owns one SQL transaction, takes deterministic keyed locks on
        that same connection, and yields its session to all subsequent writes.
        """
        normalized = sorted({(str(claim_id), str(provider)) for claim_id, provider in keys})
        with self._sessions.begin() as session:
            with ExitStack() as stack:
                for claim_id, provider in normalized:
                    stack.enter_context(self.coordination_lock(claim_id, provider, session=session))
                yield session

    @contextmanager
    def coordination_lock(self, claim_id: str, provider: str, session=None):
        """Acquire the planner/worker claim-provider lock for one UoW.

        SQLite has no transaction advisory locks, so the domain keyed lock is
        held for the duration of the caller's transaction.  PostgreSQL adds a
        transaction-scoped advisory lock on the same key; callers must pass
        their active Session so the lock is released exactly at commit/rollback.
        """
        with VerificationCoordination.lock(claim_id, provider):
            if session is not None and session.get_bind().dialect.name == "postgresql":
                key = f"verification:{claim_id}:{provider}"
                session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                    {"key": key},
                )
            yield

    def find_active_as_of(self, *, claim_id: str, provider: str, as_of: datetime, session=None):
        """Return an active job that existed at the candidate snapshot time."""
        candidate = _as_utc(as_of)
        def query(active_session):
            return active_session.scalar(
                select(ClaimVerificationJobRow)
                .where(
                    ClaimVerificationJobRow.claim_id == claim_id,
                    ClaimVerificationJobRow.provider == provider,
                    ClaimVerificationJobRow.created_at <= candidate,
                    ClaimVerificationJobRow.status.in_((PENDING, LEASED, "RETRYABLE")),
                )
                .order_by(ClaimVerificationJobRow.created_at, ClaimVerificationJobRow.job_id)
            )
        if session is not None:
            return query(session)
        with self._sessions() as session:
            return query(session)

    def get_result(self, verification_id: str, session=None) -> ClaimVerificationResultRow | None:
        if session is not None:
            return session.get(ClaimVerificationResultRow, verification_id)
        with self._sessions() as session:
            return session.get(ClaimVerificationResultRow, verification_id)

    def latest_terminal_as_of(
        self, *, claim_id: str, provider: str, as_of: datetime, policy_version: str | None = None,
        session=None,
    ):
        """Read the newest *available* compatible immutable result at PIT."""
        candidate = _as_utc(as_of)
        terminal = (
            "VERIFIED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE",
            "NOT_REQUIRED", "EXPIRED", "MANUAL_REVIEW",
        )
        def query(active_session):
            rows = active_session.scalars(
                select(ClaimVerificationResultRow)
                .where(
                    ClaimVerificationResultRow.claim_id == claim_id,
                    ClaimVerificationResultRow.provider == provider,
                    ClaimVerificationResultRow.status.in_(terminal),
                    ClaimVerificationResultRow.available_at.is_not(None),
                    ClaimVerificationResultRow.available_at <= candidate,
                )
                .order_by(
                    ClaimVerificationResultRow.available_at.desc(),
                    ClaimVerificationResultRow.created_at.desc(),
                    ClaimVerificationResultRow.verification_id.desc(),
                )
            ).all()
            for row in rows:
                if policy_version:
                    payload = dict(row.result_payload or {})
                    version = payload.get("verification_rule_version") or row.verification_rule_version
                    if version != policy_version:
                        continue
                return row
            return None
        if session is not None:
            return query(session)
        with self._sessions() as session:
            return query(session)

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
        result = result.model_copy(update={"available_at": _result_available_at(result, current)})
        with self._sessions.begin() as session:
            job = session.get(ClaimVerificationJobRow, job_id)
            if job is None:
                raise KeyError("verification job not found")
            with self.coordination_lock(job.claim_id, job.provider, session):
                # Re-read after acquiring the lock: a planner or another
                # worker may have completed this canonical claim meanwhile.
                job = session.get(ClaimVerificationJobRow, job_id)
                if job is None:
                    raise KeyError("verification job not found")
                if result.claim_id != job.claim_id:
                    raise VerificationJobIntegrityError("result claim does not match leased job")
                verification_id = verification_id_of(job.claim_id, job.provider, result)
                payload = result.model_dump(mode="json")
                existing = session.get(ClaimVerificationResultRow, verification_id)
                values = {
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
                        "available_at": _result_available_at(result, current),
                    "created_at": current,
                }
                if existing is not None:
                    # Deterministic IDs intentionally omit the PIT envelope.
                    # A retry at a later wall-clock time must preserve the
                    # first completion's immutable payload/visibility rather
                    # than attempting to move ``available_at`` forward.
                    values["available_at"] = existing.available_at
                    values["result_payload"] = dict(existing.result_payload or {})
                    row = persist_verification_result(session, values)
                    job.status = result.status
                    job.lease_owner = None
                    job.lease_expires_at = None
                    job.next_retry_at = None
                    return row
                self._assert_owner(job, worker_id, current)
                row = persist_verification_result(session, values)
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
            # ``available_at`` is the persistence/PIT envelope.  Preserve the
            # historical read API, where provider result semantics did not
            # expose this operational timestamp; PIT callers use the row
            # repository methods directly.
            payload = dict(rows[0].result_payload or {})
            payload.pop("available_at", None)
            return VerificationResult.model_validate(payload)

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
        "available_at",
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


def _as_utc(value: datetime) -> datetime:
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).astimezone(UTC)


def _result_available_at(result: VerificationResult, fallback: datetime) -> datetime:
    values = [_as_utc(result.available_at or fallback), _as_utc(fallback)]
    if result.verification_timestamp is not None:
        values.append(_as_utc(result.verification_timestamp))
    return max(values)

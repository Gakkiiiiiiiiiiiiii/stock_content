"""Transactional signal outbox with durable leases and retry state."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import SignalOutboxRow

OUTBOX_PENDING = "PENDING"
OUTBOX_PUBLISHING = "PUBLISHING"
OUTBOX_RETRY = "RETRY"
OUTBOX_PUBLISHED = "PUBLISHED"
OUTBOX_DEAD_LETTER = "DEAD_LETTER"
OUTBOX_RETRY_SECONDS = (60, 300, 1800, 7200, 43200)


class SignalOutboxIntegrityError(ValueError):
    pass


def outbox_id_of(signal_id: str) -> str:
    return "outbox-" + hashlib.sha256(signal_id.encode()).hexdigest()[:32]


class SignalOutboxRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def enqueue(self, payload: dict[str, Any], now: datetime | None = None) -> SignalOutboxRow:
        current = now or datetime.now(UTC)
        signal_id = str(payload.get("signal_id") or "")
        if not signal_id:
            raise ValueError("outbox payload requires signal_id")
        with self._sessions.begin() as session:
            return self.enqueue_in_session(session, payload, current)

    def enqueue_in_session(self, session, payload: dict[str, Any], now: datetime | None = None) -> SignalOutboxRow:
        current = now or datetime.now(UTC)
        signal_id = str(payload.get("signal_id") or "")
        if not signal_id:
            raise ValueError("outbox payload requires signal_id")
        _insert_ignore(
                session,
                SignalOutboxRow,
                {
                    "outbox_id": outbox_id_of(signal_id),
                    "signal_id": signal_id,
                    "content_snapshot_id": payload.get("content_snapshot_id"),
                    "claim_id": payload.get("claim_id"),
                    "schema_version": str(payload.get("signal_schema_version") or "content-factor-signal.v4"),
                    "payload": dict(payload),
                    "status": OUTBOX_PENDING,
                    "attempts": 0,
                    "next_attempt_at": current,
                    "created_at": current,
                },
                [SignalOutboxRow.signal_id],
            )
        row = session.scalar(select(SignalOutboxRow).where(SignalOutboxRow.signal_id == signal_id))
        if row is None:
            raise RuntimeError("outbox row disappeared after a unique-key conflict")
        if dict(row.payload or {}) != payload:
            raise SignalOutboxIntegrityError(f"signal {signal_id} already has a different payload")
        return row

    def claim_due(
        self, worker_id: str, limit: int = 10, lease_seconds: int = 60, now: datetime | None = None
    ) -> list[SignalOutboxRow]:
        current = now or datetime.now(UTC)
        due = or_(SignalOutboxRow.next_attempt_at.is_(None), SignalOutboxRow.next_attempt_at <= current)
        reclaimable = or_(SignalOutboxRow.lease_expires_at.is_(None), SignalOutboxRow.lease_expires_at <= current)
        with self._sessions.begin() as session:
            rows = list(
                session.scalars(
                    select(SignalOutboxRow)
                    .where(
                        SignalOutboxRow.status.in_((OUTBOX_PENDING, OUTBOX_RETRY, OUTBOX_PUBLISHING)),
                        due,
                        reclaimable,
                    )
                    .order_by(SignalOutboxRow.created_at, SignalOutboxRow.outbox_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                ).all()
            )
            expiry = current + timedelta(seconds=lease_seconds)
            for row in rows:
                row.status = OUTBOX_PUBLISHING
                row.lease_owner = worker_id
                row.lease_expires_at = expiry
            return rows

    def mark_published(self, outbox_id: str, worker_id: str, now: datetime | None = None) -> SignalOutboxRow:
        current = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.get(SignalOutboxRow, outbox_id)
            self._assert_owner(row, worker_id, current)
            row.status = OUTBOX_PUBLISHED
            row.published_at = current
            row.lease_owner = None
            row.lease_expires_at = None
            row.next_attempt_at = None
            return row

    def mark_retry(self, outbox_id: str, worker_id: str, error: str, now: datetime | None = None) -> SignalOutboxRow:
        current = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.get(SignalOutboxRow, outbox_id)
            self._assert_owner(row, worker_id, current)
            row.attempts += 1
            row.last_error = str(error)
            row.lease_owner = None
            row.lease_expires_at = None
            if row.attempts > len(OUTBOX_RETRY_SECONDS):
                row.status = OUTBOX_DEAD_LETTER
                row.next_attempt_at = None
            else:
                row.status = OUTBOX_RETRY
                row.next_attempt_at = current + timedelta(seconds=OUTBOX_RETRY_SECONDS[row.attempts - 1])
            return row

    def get(self, outbox_id: str) -> SignalOutboxRow | None:
        with self._sessions() as session:
            return session.get(SignalOutboxRow, outbox_id)

    def get_by_signal_id(self, signal_id: str) -> SignalOutboxRow | None:
        with self._sessions() as session:
            return session.scalar(select(SignalOutboxRow).where(SignalOutboxRow.signal_id == signal_id))

    def list_for_snapshot(self, snapshot_id: str, *, claim_id: str | None = None) -> list[SignalOutboxRow]:
        with self._sessions() as session:
            statement = select(SignalOutboxRow).where(SignalOutboxRow.content_snapshot_id == snapshot_id)
            if claim_id:
                statement = statement.where(SignalOutboxRow.claim_id == claim_id)
            return list(
                session.scalars(statement.order_by(SignalOutboxRow.created_at, SignalOutboxRow.signal_id)).all()
            )

    def list_all(self, *, include_published: bool = True) -> list[SignalOutboxRow]:
        with self._sessions() as session:
            statement = select(SignalOutboxRow)
            if not include_published:
                statement = statement.where(SignalOutboxRow.status != OUTBOX_PUBLISHED)
            return list(
                session.scalars(statement.order_by(SignalOutboxRow.created_at, SignalOutboxRow.signal_id)).all()
            )

    @staticmethod
    def _assert_owner(row: SignalOutboxRow | None, worker_id: str, now: datetime) -> None:
        if row is None:
            raise KeyError("outbox row not found")
        if row.status != OUTBOX_PUBLISHING or row.lease_owner != worker_id:
            raise SignalOutboxIntegrityError("outbox lease owner mismatch")
        expiry = row.lease_expires_at
        if expiry is not None:
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            if expiry <= now:
                raise SignalOutboxIntegrityError("outbox lease expired")


__all__ = [
    "OUTBOX_DEAD_LETTER",
    "OUTBOX_PENDING",
    "OUTBOX_PUBLISHED",
    "OUTBOX_PUBLISHING",
    "OUTBOX_RETRY",
    "SignalOutboxIntegrityError",
    "SignalOutboxRepository",
    "outbox_id_of",
]


def _insert_ignore(session, model, values: dict, conflict_columns: list) -> bool:
    """Insert once without poisoning the surrounding unit of work on a race."""
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

"""Fenced write boundary for replayable SQL and external business effects."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ContentTaskEffectRow, ContentTaskRow
from stock_content.ports.repositories import StaleTaskLease


@dataclass(frozen=True, slots=True)
class EffectIntent:
    effect_key: str
    effect_kind: str
    payload: dict[str, Any]

    @property
    def payload_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()


class PostgresFencedEffectUnitOfWork:
    """Make the task-lease predicate part of the durable effect transaction.

    SQL callbacks receive the very session that holds the current task row
    lock.  External callers first persist an intent, then must re-check the
    fence immediately before dispatch and before accepting completion.  The
    stable ``effect_key`` is the external idempotency/deduplication key.
    """

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def execute_sql(
        self,
        task_id: str,
        worker_id: str,
        fencing_token: int,
        intent: EffectIntent,
        callback: Callable[[Any], Any],
    ) -> Any:
        with self._sessions.begin() as session:
            self._require_current(session, task_id, worker_id, fencing_token)
            row = self._intent(session, task_id, fencing_token, intent)
            if row.state == "COMPLETED":
                return None
            value = callback(session)
            row.state = "COMPLETED"
            row.completed_at = datetime.now(UTC)
            row.attempt_count += 1
            return value

    def require_current_in_session(self, session, task_id: str, worker_id: str, fencing_token: int) -> None:
        """Assert a caller-owned SQL transaction still owns the task fence."""
        self._require_current(session, task_id, worker_id, fencing_token)

    @contextmanager
    def fenced_transaction(self, task_id: str, worker_id: str, fencing_token: int):
        """Yield a task-fenced transaction for coordinated SQL planning.

        Snapshot planning sometimes has no verification key (for example an
        empty offline extraction), so it cannot borrow a verification planner
        transaction.  It still must not downgrade to an unfenced session.
        """
        with self._sessions.begin() as session:
            self._require_current(session, task_id, worker_id, fencing_token)
            yield session

    def prepare_external(self, task_id: str, worker_id: str, fencing_token: int, intent: EffectIntent) -> str:
        """Persist an external-effect intent only while the current lease owns it."""
        with self._sessions.begin() as session:
            self._require_current(session, task_id, worker_id, fencing_token)
            row = self._intent(session, task_id, fencing_token, intent)
            return row.effect_id

    def dispatch_external(
        self,
        task_id: str,
        worker_id: str,
        fencing_token: int,
        intent: EffectIntent,
        callback: Callable[[str], Any],
    ) -> Any:
        """Dispatch idempotently with fence checks before and after the call."""
        effect_id = self.prepare_external(task_id, worker_id, fencing_token, intent)
        with self._sessions() as session:
            self._require_current(session, task_id, worker_id, fencing_token)
            row = session.get(ContentTaskEffectRow, effect_id)
            if row is not None and row.state == "COMPLETED":
                return None
        value = callback(effect_id)
        with self._sessions.begin() as session:
            self._require_current(session, task_id, worker_id, fencing_token)
            row = session.get(ContentTaskEffectRow, effect_id)
            if row is None:
                raise RuntimeError("fenced effect intent disappeared")
            row.state = "COMPLETED"
            row.completed_at = datetime.now(UTC)
            row.attempt_count += 1
        return value

    @staticmethod
    def _require_current(session, task_id: str, worker_id: str, fencing_token: int) -> ContentTaskRow:
        row = session.scalar(select(ContentTaskRow).where(ContentTaskRow.task_id == task_id).with_for_update())
        expires_at = row.lease_expires_at if row is not None else None
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if (
            row is None
            or row.status != "RUNNING"
            or row.lease_owner != worker_id
            or row.fencing_token != fencing_token
            or expires_at is None
            or expires_at <= datetime.now(UTC)
        ):
            raise StaleTaskLease(f"content task lease is stale: {task_id}")
        return row

    @staticmethod
    def _intent(session, task_id: str, fencing_token: int, intent: EffectIntent) -> ContentTaskEffectRow:
        row = session.scalar(
            select(ContentTaskEffectRow)
            .where(ContentTaskEffectRow.task_id == task_id, ContentTaskEffectRow.effect_key == intent.effect_key)
            .with_for_update()
        )
        if row is None:
            row = ContentTaskEffectRow(
                effect_id="eff_" + hashlib.sha256(f"{task_id}|{intent.effect_key}".encode()).hexdigest(),
                task_id=task_id,
                effect_key=intent.effect_key,
                effect_kind=intent.effect_kind,
                payload_hash=intent.payload_hash,
                projection_payload=dict(intent.payload),
                fencing_token=fencing_token,
            )
            session.add(row)
            session.flush()
            return row
        if row.effect_kind != intent.effect_kind or row.payload_hash != intent.payload_hash:
            raise ValueError("fenced effect key already binds different immutable payload")
        # A recovered worker resumes a pending external intent using its own
        # current fence. A completed effect is never dispatched a second time.
        if row.state != "COMPLETED":
            row.fencing_token = fencing_token
        return row


__all__ = ["EffectIntent", "PostgresFencedEffectUnitOfWork"]

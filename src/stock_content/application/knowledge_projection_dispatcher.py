"""Durable, independently retryable Qdrant knowledge projection.

The content task's SQL publication is authoritative.  This dispatcher consumes
only its durable projection intents after that task has completed; an index
outage can therefore never change ingestion terminal state or Bundle output.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ContentTaskEffectRow, ContentTaskRow, KnowledgeUnitRow
from stock_content.adapters.qdrant import NullKnowledgeIndex
from stock_content.domain.models import KnowledgeUnit


@dataclass(frozen=True, slots=True)
class _DispatchClaim:
    """A worker's non-transferable projection-dispatch generation."""

    effect_id: str
    fencing_token: int


class KnowledgeProjectionDispatcher:
    """Lease and deliver ``KNOWLEDGE_INDEX`` intents with deterministic retry."""

    effect_kind = "KNOWLEDGE_INDEX"

    def __init__(
        self,
        sessions: sessionmaker,
        index: Any,
        *,
        max_attempts: int = 10,
        retry_base_seconds: int = 5,
    ) -> None:
        self._sessions = sessions
        self._index = index
        self._max_attempts = max(1, max_attempts)
        self._retry_base_seconds = max(1, retry_base_seconds)

    @property
    def configured(self) -> bool:
        return self._index is not None and not isinstance(self._index, NullKnowledgeIndex)

    def dispatch_due(self, worker_id: str, *, limit: int = 20, lease_seconds: int = 60) -> dict[str, int]:
        """Deliver due intents once each; failures remain durable and retryable."""
        if not self.configured:
            return {"dispatched": 0, "retried": 0, "dead_lettered": 0, "pending": self.pending_count()}
        dispatched = retried = dead_lettered = 0
        for claim in self._claim_due(worker_id, limit=limit, lease_seconds=lease_seconds):
            try:
                # Loading before the final fence is intentionally harmless:
                # it performs no external effect.  The dispatch method below
                # locks and reloads the intent immediately before index(), so
                # a worker paused here cannot write after a takeover.
                self._load_units(claim.effect_id)
            except Exception as error:  # authority/read failures are retryable
                outcome = self._retry(claim, worker_id, error)
            else:
                outcome = self._dispatch_claim(claim, worker_id)

            if outcome == "DISPATCHED":
                dispatched += 1
            elif outcome == "RETRIED":
                retried += 1
            elif outcome == "DEAD_LETTERED":
                dead_lettered += 1
            # A superseded worker is deliberately a no-op.  In particular it
            # must not overwrite a successor's completed receipt or retry.
        return {
            "dispatched": dispatched,
            "retried": retried,
            "dead_lettered": dead_lettered,
            "pending": self.pending_count(),
        }

    def pending_count(self) -> int:
        with self._sessions() as session:
            return len(
                session.scalars(
                    select(ContentTaskEffectRow.effect_id).where(
                        ContentTaskEffectRow.effect_kind == self.effect_kind,
                        ContentTaskEffectRow.state.in_(("PENDING", "DISPATCHING")),
                    )
                ).all()
            )

    def _claim_due(self, worker_id: str, *, limit: int, lease_seconds: int) -> list[_DispatchClaim]:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            due = or_(
                and_(
                    ContentTaskEffectRow.state == "PENDING",
                    or_(
                        ContentTaskEffectRow.next_attempt_at.is_(None),
                        ContentTaskEffectRow.next_attempt_at <= now,
                    ),
                ),
                and_(
                    ContentTaskEffectRow.state == "DISPATCHING",
                    ContentTaskEffectRow.dispatch_expires_at <= now,
                ),
            )
            rows = session.scalars(
                select(ContentTaskEffectRow)
                .where(ContentTaskEffectRow.effect_kind == self.effect_kind, due)
                .order_by(ContentTaskEffectRow.created_at, ContentTaskEffectRow.effect_id)
                .limit(max(1, limit))
                .with_for_update(skip_locked=True)
            ).all()
            for row in rows:
                row.state = "DISPATCHING"
                row.dispatch_owner = worker_id
                row.dispatch_expires_at = now + timedelta(seconds=max(1, lease_seconds))
                row.next_attempt_at = None
                row.attempt_count += 1
                # The intent's original token binds its creation to the
                # ingestion task.  Once dispatch is independent, this
                # monotonically increasing token fences dispatch ownership.
                row.fencing_token += 1
            return [_DispatchClaim(row.effect_id, row.fencing_token) for row in rows]

    def _load_units(self, effect_id: str | _DispatchClaim) -> list[KnowledgeUnit]:
        effect_id = effect_id.effect_id if isinstance(effect_id, _DispatchClaim) else effect_id
        with self._sessions() as session:
            effect = session.get(ContentTaskEffectRow, effect_id)
            return self._load_units_in_session(session, effect)

    @staticmethod
    def _load_units_in_session(session, effect: ContentTaskEffectRow | None) -> list[KnowledgeUnit]:
        if effect is None:
            raise RuntimeError("projection intent disappeared")
        # The immutable payload remains hash-bound by the fenced effect, but
        # SQL is the sole source for hydrated projection data.  The payload
        # holds only stable knowledge identifiers, never locators or raw
        # media, so an external index retry has no secret input.
        payload = getattr(effect, "projection_payload", None) or {}
        if not isinstance(payload, dict):
            raise RuntimeError("PROJECTION_INTENT_PAYLOAD_INVALID")
        knowledge_ids = tuple(str(value) for value in payload.get("knowledge_ids") or ())
        if knowledge_ids:
            rows = {
                row.knowledge_uid: row
                for row in session.scalars(
                    select(KnowledgeUnitRow).where(KnowledgeUnitRow.knowledge_uid.in_(knowledge_ids))
                ).all()
            }
            if set(rows) != set(knowledge_ids):
                raise RuntimeError("PROJECTION_SQL_AUTHORITY_INCOMPLETE")
            return [_to_unit(rows[knowledge_id]) for knowledge_id in knowledge_ids]
        # Migration-safe recovery for an old 034 intent that crashed after
        # its hash was stored but before this release had a payload column.
        # The task's sealed SQL result is still authority; if it cannot
        # identify a completed video, do not invent a projection.
        task = session.get(ContentTaskRow, effect.task_id)
        video_id = str((task.result or {}).get("video_id") or "") if task else ""
        if not video_id:
            raise RuntimeError("PROJECTION_SQL_AUTHORITY_INCOMPLETE")
        rows = session.scalars(
            select(KnowledgeUnitRow)
            .where(KnowledgeUnitRow.video_id == video_id)
            .order_by(KnowledgeUnitRow.knowledge_uid)
        ).all()
        return [_to_unit(row) for row in rows]

    def _dispatch_claim(self, claim: _DispatchClaim, worker_id: str) -> str:
        """Invoke the index only while holding the current dispatch fence.

        The external call is deliberately inside the effect-row transaction.
        A Postgres row lock makes a lease takeover wait instead of allowing a
        second worker to pass the external-effect seam.  A process crash after
        remote success but before commit is still at-least-once at the remote
        boundary; the stable effect key and deterministic Qdrant point IDs
        keep its business projection exactly-once.
        """
        try:
            with self._sessions.begin() as session:
                row = self._owned(session, claim, worker_id)
                # Reload under the same lock: no stale payload or units reach
                # the remote index after a predecessor was paused after load.
                units = self._load_units_in_session(session, row)
                self._index.index(units, idempotency_key=claim.effect_id)
                row.state = "COMPLETED"
                row.completed_at = datetime.now(UTC)
                row.dispatch_owner = None
                row.dispatch_expires_at = None
                row.last_error_code = None
            return "DISPATCHED"
        except _ProjectionDispatchLeaseLost:
            return "LOST"
        except Exception as error:  # external projection must not affect task state
            return self._retry(claim, worker_id, error)

    def _complete(self, effect_id: str | _DispatchClaim, worker_id: str) -> None:
        """Compatibility seam for receipt tests; dispatch_due uses a token."""
        with self._sessions.begin() as session:
            row = self._owned(session, effect_id, worker_id)
            row.state = "COMPLETED"
            row.completed_at = datetime.now(UTC)
            row.dispatch_owner = None
            row.dispatch_expires_at = None
            row.last_error_code = None

    def _retry(self, claim: _DispatchClaim, worker_id: str, error: Exception) -> str:
        now = datetime.now(UTC)
        try:
            with self._sessions.begin() as session:
                row = self._owned(session, claim, worker_id)
                row.dispatch_owner = None
                row.dispatch_expires_at = None
                row.last_error_code = type(error).__name__[:64]
                if row.attempt_count >= self._max_attempts:
                    row.state = "DEAD_LETTER"
                    return "DEAD_LETTERED"
                row.state = "PENDING"
                delay = min(3600, self._retry_base_seconds * (2 ** min(row.attempt_count - 1, 8)))
                row.next_attempt_at = now + timedelta(seconds=delay)
                return "RETRIED"
        except _ProjectionDispatchLeaseLost:
            return "LOST"

    @staticmethod
    def _owned(session, claim: _DispatchClaim | str, worker_id: str) -> ContentTaskEffectRow:
        effect_id = claim.effect_id if isinstance(claim, _DispatchClaim) else claim
        row = session.scalar(
            select(ContentTaskEffectRow)
            .where(ContentTaskEffectRow.effect_id == effect_id)
            .with_for_update()
        )
        now = datetime.now(UTC)
        if (
            row is None
            or row.state != "DISPATCHING"
            or row.dispatch_owner != worker_id
            or row.dispatch_expires_at is None
            or _utc(row.dispatch_expires_at) <= now
            or (isinstance(claim, _DispatchClaim) and row.fencing_token != claim.fencing_token)
        ):
            raise _ProjectionDispatchLeaseLost("PROJECTION_DISPATCH_LEASE_LOST")
        return row


class _ProjectionDispatchLeaseLost(RuntimeError):
    """A non-transferable dispatch generation was superseded."""


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _to_unit(row: KnowledgeUnitRow) -> KnowledgeUnit:
    return KnowledgeUnit(
        knowledge_uid=row.knowledge_uid,
        video_id=row.video_id,
        chapter_id=row.chapter_id,
        statement=row.statement,
        kind=row.kind,
        knowledge_kind=row.knowledge_kind,
        knowledge_version=row.knowledge_version,
        subject=row.subject,
        subject_key=row.subject_key,
        predicate_key=row.predicate_key,
        ticker=row.ticker,
        sentiment=row.sentiment,
        support_status=row.support_status,
        truth_status=row.truth_status,
        review_status=row.review_status,
        lifecycle_status=row.lifecycle_status,
        confidence=row.confidence,
        as_of=_utc(row.as_of),
        available_from=_utc(row.available_from),
        valid_from=_utc(row.valid_from) if row.valid_from else None,
        valid_to=_utc(row.valid_to) if row.valid_to else None,
        source_statement_hash=row.source_statement_hash,
        content_hash=row.content_hash,
        attributes=dict(row.attributes or {}),
        provenance=dict(row.provenance or {}),
    )


__all__ = ["KnowledgeProjectionDispatcher"]

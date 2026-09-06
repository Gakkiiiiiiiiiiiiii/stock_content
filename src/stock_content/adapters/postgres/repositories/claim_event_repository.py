from __future__ import annotations

from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ClaimStateEventRow, FinancialClaimRow
from stock_content.domain.claim_state_event import ClaimStateEvent, validate_event_chain


class ClaimStateEventRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def append(self, event: ClaimStateEvent) -> ClaimStateEvent:
        with self._sessions.begin() as session:
            self.append_in_session(session, event)
        return event

    def append_in_session(self, session, event: ClaimStateEvent) -> ClaimStateEvent:
        # Validate the content-addressed identity at the persistence boundary
        # as well as at construction time.  Loaded/tampered frozen dataclass
        # instances must not be able to insert an event with a stale hash.
        ClaimStateEvent(
            claim_id=event.claim_id,
            event_type=event.event_type,
            payload=dict(event.payload),
            known_from=event.known_from,
            business_valid_from=event.business_valid_from,
            business_valid_to=event.business_valid_to,
            known_to=event.known_to,
            source_available_from=event.source_available_from,
            previous_event_hash=event.previous_event_hash,
            event_id=event.event_id,
            event_hash=event.event_hash,
            legacy_history_incomplete=event.legacy_history_incomplete,
        )
        # A snapshot projection may append more than one event for a claim in
        # one transaction.  Flush first so the chain's terminal node is read
        # from the durable session state rather than an incidental
        # ``created_at`` ordering (which may tie on SQLite).
        session.flush()
        row = session.get(ClaimStateEventRow, event.event_id)
        if row is not None:
            def utc(value):
                return value if value is None or value.tzinfo else value.replace(tzinfo=UTC)

            stored = (row.claim_id, row.event_type, dict(row.payload or {}), utc(row.known_from), utc(row.known_to),
                      utc(row.business_valid_from), utc(row.business_valid_to), utc(row.source_available_from),
                      row.previous_event_hash, row.event_hash, row.legacy_history_incomplete)
            candidate = (event.claim_id, event.event_type, dict(event.payload), event.known_from, event.known_to,
                         event.business_valid_from, event.business_valid_to, event.source_available_from,
                         event.previous_event_hash, event.event_hash, event.legacy_history_incomplete)
            if stored != candidate:
                raise ValueError(f"claim state event {event.event_id} already stores different payload")
            return event
        tail = self._tail_in_session(session, event.claim_id)
        if tail is not None and event.previous_event_hash != tail.event_hash:
            raise ValueError("claim state event previous_event_hash does not match chain tail")
        session.add(ClaimStateEventRow(
                    claim_state_event_id=event.event_id,
                    claim_id=event.claim_id,
                    event_type=event.event_type,
                    business_valid_from=event.business_valid_from,
                    business_valid_to=event.business_valid_to,
                    known_from=event.known_from,
                    known_to=event.known_to,
                    source_available_from=event.source_available_from,
                    payload=dict(event.payload),
                    previous_event_hash=event.previous_event_hash,
                    event_hash=event.event_hash,
                    legacy_history_incomplete=event.legacy_history_incomplete,
                ))
        return event

    insert = append

    def list_for_claim(self, claim_id: str) -> list[ClaimStateEvent]:
        with self._sessions() as session:
            rows = session.scalars(
                select(ClaimStateEventRow)
                .where(ClaimStateEventRow.claim_id == claim_id)
            ).all()
        return list(validate_event_chain(self._events_from_rows(rows)))

    def _tail_in_session(self, session, claim_id: str) -> ClaimStateEvent | None:
        """Return the sole terminal event, never an arbitrary timestamp tie."""
        rows = session.scalars(
            select(ClaimStateEventRow)
            .where(ClaimStateEventRow.claim_id == claim_id)
            # Lock every row in this claim's chain.  A terminal-row-only lock
            # would not protect a chain whose ordering is encoded by hashes.
            .with_for_update()
        ).all()
        events = validate_event_chain(self._events_from_rows(rows))
        return events[-1] if events else None

    @staticmethod
    def _events_from_rows(rows) -> list[ClaimStateEvent]:
        def utc(value):
            return value if value is None or value.tzinfo else value.replace(tzinfo=UTC)

        return [
            ClaimStateEvent(
                claim_id=row.claim_id,
                event_type=row.event_type,
                payload=dict(row.payload or {}),
                known_from=utc(row.known_from),
                business_valid_from=utc(row.business_valid_from),
                business_valid_to=utc(row.business_valid_to),
                known_to=utc(row.known_to),
                source_available_from=utc(row.source_available_from),
                previous_event_hash=row.previous_event_hash,
                event_id=row.claim_state_event_id,
                event_hash=row.event_hash,
                legacy_history_incomplete=row.legacy_history_incomplete,
            )
            for row in rows
        ]

    def is_history_incomplete(self, claim_id: str) -> bool:
        """Read the one-way migration marker; never infer history from claims."""
        with self._sessions() as session:
            row = session.get(FinancialClaimRow, claim_id)
            return bool(row is not None and row.legacy_history_incomplete)


__all__ = ["ClaimStateEventRepository"]

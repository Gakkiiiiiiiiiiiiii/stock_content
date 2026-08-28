"""Append-only lifecycle repository with deterministic bitemporal reads."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import LifecycleEventLedgerRow
from stock_content.domain.lifecycle_event import (
    KnowledgeLifecycleEvent,
    LifecycleTargetType,
    select_lifecycle_event,
)


class LifecycleRepository:
    """Persist lifecycle events without update/delete operations."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def append(self, event: KnowledgeLifecycleEvent) -> KnowledgeLifecycleEvent:
        with self._sessions.begin() as session:
            self.append_in_session(session, event)
        return event

    def append_in_session(self, session, event: KnowledgeLifecycleEvent) -> KnowledgeLifecycleEvent:
        row = session.get(LifecycleEventLedgerRow, event.lifecycle_event_id)
        if row is not None:
            if not (
                row.target_type == event.target_type.value
                and row.target_id == event.target_id
                and row.from_status == event.from_status
                and row.to_status == event.to_status
                and _same_time(row.effective_at, event.effective_at)
                and _same_time(row.recorded_at, event.recorded_at)
                and row.reason_code == event.reason_code
                and row.policy_version == event.policy_version
                and row.supersedes_event_id == event.supersedes_event_id
            ):
                raise ValueError(f"lifecycle event id {event.lifecycle_event_id} already stores different payload")
            return event
        session.add(
            LifecycleEventLedgerRow(
                lifecycle_event_id=event.lifecycle_event_id,
                target_type=event.target_type.value,
                target_id=event.target_id,
                from_status=event.from_status,
                to_status=event.to_status,
                effective_at=event.effective_at,
                recorded_at=event.recorded_at,
                reason_code=event.reason_code,
                policy_version=event.policy_version,
                supersedes_event_id=event.supersedes_event_id,
            )
        )
        return event

    insert = append

    def get(self, lifecycle_event_id: str) -> KnowledgeLifecycleEvent | None:
        with self._sessions() as session:
            row = session.get(LifecycleEventLedgerRow, lifecycle_event_id)
        if row is None:
            return None
        return KnowledgeLifecycleEvent(
            lifecycle_event_id=row.lifecycle_event_id,
            target_type=LifecycleTargetType(row.target_type),
            target_id=row.target_id,
            from_status=row.from_status,
            to_status=row.to_status,
            effective_at=_utc(row.effective_at),
            recorded_at=_utc(row.recorded_at),
            reason_code=row.reason_code,
            policy_version=row.policy_version,
            supersedes_event_id=row.supersedes_event_id,
        )

    def select_as_of(
        self,
        *,
        target_type: LifecycleTargetType | str,
        target_id: str,
        business_as_of: datetime,
        knowledge_as_of: datetime,
    ) -> KnowledgeLifecycleEvent | None:
        with self._sessions() as session:
            rows = session.scalars(
                select(LifecycleEventLedgerRow)
                .where(
                    LifecycleEventLedgerRow.target_id == target_id,
                    LifecycleEventLedgerRow.target_type == (
                        target_type.value if isinstance(target_type, LifecycleTargetType) else str(target_type)
                    ),
                )
            ).all()
        events = [
            KnowledgeLifecycleEvent(
                lifecycle_event_id=row.lifecycle_event_id,
                target_type=LifecycleTargetType(row.target_type),
                target_id=row.target_id,
                from_status=row.from_status,
                to_status=row.to_status,
                effective_at=row.effective_at,
                recorded_at=row.recorded_at,
                reason_code=row.reason_code,
                policy_version=row.policy_version,
                supersedes_event_id=row.supersedes_event_id,
            )
            for row in rows
        ]
        return select_lifecycle_event(
            events,
            target_type=target_type,
            target_id=target_id,
            business_as_of=business_as_of,
            knowledge_as_of=knowledge_as_of,
        )


__all__ = ["LifecycleRepository"]


def _same_time(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return left == right
    if left.tzinfo is None:
        left = left.replace(tzinfo=UTC)
    if right.tzinfo is None:
        right = right.replace(tzinfo=left.tzinfo)
    return left == right


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)

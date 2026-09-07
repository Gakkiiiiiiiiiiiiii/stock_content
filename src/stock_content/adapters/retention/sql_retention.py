"""SQL retention audit adapter; it deliberately stores no physical locator."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import RetentionExecutionRow
from stock_content.domain.retention import RetentionExecution, RetentionExecutionState, Tombstone


class SqlRetentionExecutionRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def get(self, tombstone_id: str) -> Tombstone | None:
        execution = self.get_execution(tombstone_id)
        return execution.tombstone if execution else None

    def insert_immutable(self, tombstone: Tombstone) -> Tombstone:
        return self.prepare_execution(tombstone).tombstone

    def get_execution(self, tombstone_id: str) -> RetentionExecution | None:
        with self._sessions() as session:
            row = session.get(RetentionExecutionRow, tombstone_id)
            return _execution(row) if row else None

    def prepare_execution(self, tombstone: Tombstone) -> RetentionExecution:
        with self._sessions.begin() as session:
            row = session.get(RetentionExecutionRow, tombstone.tombstone_id)
            if row is None:
                by_artifact = session.scalar(
                    select(RetentionExecutionRow).where(RetentionExecutionRow.artifact_id == tombstone.artifact_id)
                )
                if by_artifact is not None:
                    existing = _execution(by_artifact)
                    if existing.tombstone != tombstone:
                        raise ValueError("retention artifact already binds different immutable tombstone")
                    return existing
                row = RetentionExecutionRow(
                    tombstone_id=tombstone.tombstone_id,
                    artifact_id=tombstone.artifact_id,
                    artifact_class=tombstone.artifact_class.value,
                    content_hash=tombstone.content_hash,
                    source_identity_hash=tombstone.source_identity_hash,
                    audit_lineage_id=tombstone.audit_lineage_id,
                    reason=tombstone.reason,
                    expired_at=tombstone.expired_at,
                )
                session.add(row)
                session.flush()
            return _execution(row)

    def mark_deleted(self, tombstone_id: str) -> RetentionExecution:
        with self._sessions.begin() as session:
            row = _required(session.get(RetentionExecutionRow, tombstone_id))
            if row.state != RetentionExecutionState.DELETED.value:
                row.state = RetentionExecutionState.DELETED.value
                row.attempt_count += 1
                row.last_error_code = None
                row.finalized_at = datetime.now(UTC)
            return _execution(row)

    def mark_failed(self, tombstone_id: str, error_code: str) -> RetentionExecution:
        if not error_code or len(error_code) > 64:
            raise ValueError("retention error code must be a bounded non-secret code")
        with self._sessions.begin() as session:
            row = _required(session.get(RetentionExecutionRow, tombstone_id))
            if row.state != RetentionExecutionState.DELETED.value:
                row.state = RetentionExecutionState.DELETE_FAILED.value
                row.attempt_count += 1
                row.last_error_code = error_code
            return _execution(row)


def _required(row: RetentionExecutionRow | None) -> RetentionExecutionRow:
    if row is None:
        raise KeyError("retention execution not found")
    return row


def _execution(row: RetentionExecutionRow) -> RetentionExecution:
    from stock_content.domain.retention import RetentionClass

    return RetentionExecution(
        tombstone=Tombstone(
            tombstone_id=row.tombstone_id,
            artifact_id=row.artifact_id,
            artifact_class=RetentionClass(row.artifact_class),
            content_hash=row.content_hash,
            source_identity_hash=row.source_identity_hash,
            audit_lineage_id=row.audit_lineage_id,
            reason=row.reason,
            expired_at=_utc(row.expired_at),
            recorded_at=_utc(row.created_at),
        ),
        state=RetentionExecutionState(row.state),
        attempt_count=row.attempt_count,
        last_error_code=row.last_error_code,
    )


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


__all__ = ["SqlRetentionExecutionRepository"]

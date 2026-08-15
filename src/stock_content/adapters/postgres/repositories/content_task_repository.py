from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ContentTaskRow
from stock_content.domain.models import ContentTask


class PostgresContentTaskRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    @staticmethod
    def _domain(row: ContentTaskRow) -> ContentTask:
        return ContentTask(
            task_id=row.task_id,
            source_type=row.source_type,
            source_ref=row.source_ref,
            status=row.status,
            stage=row.stage,
            progress=row.progress,
            retry_count=row.retry_count,
            max_retries=row.max_retries,
            error=row.error,
            options=dict(row.options or {}),
            result=dict(row.result or {}),
            checkpoint=dict(row.checkpoint or {}),
            input_hash=row.input_hash,
            idempotency_key=row.idempotency_key,
            trace_id=row.trace_id,
        )

    def create(self, task: ContentTask) -> ContentTask:
        with self._sessions.begin() as session:
            if task.idempotency_key:
                existing = session.scalar(
                    select(ContentTaskRow).where(ContentTaskRow.idempotency_key == task.idempotency_key)
                )
                if existing is not None:
                    return self._domain(existing)
            session.add(ContentTaskRow(**task.to_dict()))
        return task

    def get(self, task_id: str) -> ContentTask | None:
        with self._sessions() as session:
            row = session.get(ContentTaskRow, task_id)
            return self._domain(row) if row else None

    def claim_pending(self, worker_id: str, lease_seconds: int) -> ContentTask | None:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            query = (
                select(ContentTaskRow)
                .where(
                    ContentTaskRow.retry_count < ContentTaskRow.max_retries,
                    or_(
                        ContentTaskRow.status == "PENDING",
                        (ContentTaskRow.status == "RUNNING") & (ContentTaskRow.lease_expires_at < now),
                    ),
                )
                .order_by(ContentTaskRow.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            row = session.scalar(query)
            if row is None:
                return None
            row.status = "RUNNING"
            row.lease_owner = worker_id
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.updated_at = now
            session.flush()
            return self._domain(row)

    def update_progress(self, task_id: str, stage: str, progress: int) -> None:
        with self._sessions.begin() as session:
            row = session.get(ContentTaskRow, task_id)
            if row:
                row.stage = stage
                row.progress = max(0, min(progress, 100))

    def checkpoint(self, task_id: str, stage: str, checkpoint: dict, progress: int | None = None) -> None:
        with self._sessions.begin() as session:
            row = session.get(ContentTaskRow, task_id)
            if row:
                row.stage = stage
                row.checkpoint = {**(row.checkpoint or {}), stage: checkpoint}
                if progress is not None:
                    row.progress = max(0, min(progress, 100))

    def succeed(self, task_id: str, result: dict[str, Any]) -> None:
        with self._sessions.begin() as session:
            row = session.get(ContentTaskRow, task_id)
            if row:
                row.status = "SUCCEEDED"
                row.stage = "completed"
                row.progress = 100
                row.result = result
                row.checkpoint = {**(row.checkpoint or {}), "completed": {"at": datetime.now(UTC).isoformat()}}
                row.error = None
                row.lease_owner = None
                row.lease_expires_at = None

    def fail(self, task_id: str, stage: str, error: str) -> None:
        with self._sessions.begin() as session:
            row = session.get(ContentTaskRow, task_id)
            if row:
                row.retry_count += 1
                row.status = "FAILED" if row.retry_count >= row.max_retries else "PENDING"
                row.stage = stage
                row.error = error[:4000]
                row.lease_owner = None
                row.lease_expires_at = None

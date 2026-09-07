from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ContentTaskRow
from stock_content.domain.models import ContentTask
from stock_content.ports.repositories import IdempotencyConflict, StaleTaskLease


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
            request_hash=row.request_hash,
            task_kind=row.task_kind,
            source_platform=row.source_platform,
            canonical_source_ref=row.canonical_source_ref,
            credential_ref_hash=row.credential_ref_hash,
            locator_secret_hash=row.locator_secret_hash,
            source_identity_hash=row.source_identity_hash,
            trace_id=row.trace_id,
            lease_owner=row.lease_owner,
            lease_expires_at=row.lease_expires_at,
            fencing_token=row.fencing_token,
        )

    def create(self, task: ContentTask) -> ContentTask:
        if task.idempotency_key:
            with self._sessions() as session:
                existing = session.scalar(
                    select(ContentTaskRow).where(ContentTaskRow.idempotency_key == task.idempotency_key)
                )
                if existing is not None:
                    return self._reserve_existing(existing, task)
        try:
            with self._sessions.begin() as session:
                session.add(ContentTaskRow(**task.to_dict()))
                # Force the unique-key race inside this transaction, where it
                # can be recovered by loading the committed winner below.
                session.flush()
            return task
        except IntegrityError:
            if not task.idempotency_key:
                raise
        with self._sessions() as session:
            existing = session.scalar(
                select(ContentTaskRow).where(ContentTaskRow.idempotency_key == task.idempotency_key)
            )
            if existing is None:
                raise
            return self._reserve_existing(existing, task)

    def _reserve_existing(self, existing: ContentTaskRow, task: ContentTask) -> ContentTask:
        if existing.request_hash != task.request_hash:
            raise IdempotencyConflict("Idempotency-Key is already bound to a different request")
        return self._domain(existing)

    def get(self, task_id: str) -> ContentTask | None:
        with self._sessions() as session:
            row = session.get(ContentTaskRow, task_id)
            return self._domain(row) if row else None

    def claim_pending(self, worker_id: str, task_kind: str, lease_seconds: int) -> ContentTask | None:
        # Migration 030 deliberately marks ambiguous pending legacy work as
        # unresolved.  It can only be re-ingested, never claimed by asking for
        # that string directly.
        if task_kind == "legacy_unresolved":
            return None
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            query = (
                select(ContentTaskRow)
                .where(
                    ContentTaskRow.retry_count < ContentTaskRow.max_retries,
                    ContentTaskRow.task_kind == str(task_kind),
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
            row.fencing_token += 1
            row.updated_at = now
            session.flush()
            return self._domain(row)

    @staticmethod
    def _fenced_row(session, task_id: str, worker_id: str, fencing_token: int, now: datetime) -> ContentTaskRow:
        row = session.scalar(
            select(ContentTaskRow).where(ContentTaskRow.task_id == task_id).with_for_update()
        )
        expires_at = row.lease_expires_at if row is not None else None
        # SQLite does not round-trip timezone metadata for DateTime(timezone=True).
        # Its values are nevertheless written as UTC by this repository.
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if (
            row is None
            or row.status != "RUNNING"
            or row.lease_owner != worker_id
            or row.fencing_token != fencing_token
            or row.lease_expires_at is None
            or expires_at <= now
        ):
            raise StaleTaskLease(f"content task lease is stale: {task_id}")
        return row

    def renew_lease(self, task_id: str, worker_id: str, fencing_token: int, lease_seconds: int) -> ContentTask:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            row = self._fenced_row(session, task_id, worker_id, fencing_token, now)
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.updated_at = now
            session.flush()
            return self._domain(row)

    def update_progress(self, task_id: str, stage: str, progress: int, worker_id: str, fencing_token: int) -> None:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            row = self._fenced_row(session, task_id, worker_id, fencing_token, now)
            row.stage = stage
            row.progress = max(0, min(progress, 100))

    def checkpoint(
        self, task_id: str, stage: str, checkpoint: dict, worker_id: str, fencing_token: int,
        progress: int | None = None,
    ) -> None:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            row = self._fenced_row(session, task_id, worker_id, fencing_token, now)
            row.stage = stage
            row.checkpoint = {**(row.checkpoint or {}), stage: checkpoint}
            if progress is not None:
                row.progress = max(0, min(progress, 100))

    def commit_effect(self, task_id: str, result: dict[str, Any], worker_id: str, fencing_token: int) -> None:
        """Record the terminal effect only while the caller owns the lease.

        The row lock and lease predicate are the effect-commit port for the
        task queue.  Callers must use their publication/snapshot transaction
        before this final acknowledgement; a stale worker cannot publish a
        second terminal task effect.
        """
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            row = self._fenced_row(session, task_id, worker_id, fencing_token, now)
            row.status = "SUCCEEDED"
            row.stage = "completed"
            row.progress = 100
            row.result = result
            row.checkpoint = {**(row.checkpoint or {}), "completed": {"at": now.isoformat()}}
            row.error = None
            row.lease_owner = None
            row.lease_expires_at = None

    def fail(self, task_id: str, stage: str, error: str, worker_id: str, fencing_token: int) -> None:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            row = self._fenced_row(session, task_id, worker_id, fencing_token, now)
            row.retry_count += 1
            row.status = "FAILED" if row.retry_count >= row.max_retries else "PENDING"
            row.stage = stage
            row.error = error[:4000]
            row.lease_owner = None
            row.lease_expires_at = None

    def succeed_unleased(self, task_id: str, result: dict[str, Any]) -> None:
        """Terminal path for local replay records, which are never queued."""
        with self._sessions.begin() as session:
            row = session.get(ContentTaskRow, task_id)
            if row is None or row.task_kind != "replay" or row.status != "RUNNING":
                raise StaleTaskLease(f"unleased replay task is not active: {task_id}")
            row.status = "SUCCEEDED"
            row.stage = "completed"
            row.progress = 100
            row.result = result
            row.error = None

    def fail_unleased(self, task_id: str, stage: str, error: str) -> None:
        with self._sessions.begin() as session:
            row = session.get(ContentTaskRow, task_id)
            if row is None or row.task_kind != "replay" or row.status != "RUNNING":
                raise StaleTaskLease(f"unleased replay task is not active: {task_id}")
            row.status = "FAILED"
            row.stage = stage
            row.error = error[:4000]

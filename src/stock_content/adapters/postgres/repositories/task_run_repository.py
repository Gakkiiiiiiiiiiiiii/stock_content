from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import BackgroundTaskRunRow, OperatorActionAuditRow
from stock_content.domain.task_run import Checkpoint, OperatorAction, TaskRun, TaskRunState


class PostgresTaskRunRepository:
    """SQL adapter preserving fencing checks at the write boundary."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    @staticmethod
    def _domain(row: BackgroundTaskRunRow) -> TaskRun:
        return TaskRun(
            task_run_id=row.task_run_id,
            task_type=row.task_type,
            state=TaskRunState(row.state),
            owner=row.owner,
            lease_expires_at=row.lease_expires_at,
            fencing_token=row.fencing_token,
            attempt=row.attempt,
            checkpoints=tuple(
                Checkpoint(name, dict(value or {})) for name, value in (row.checkpoints or {}).items()
            ),
        )

    def create(self, task: TaskRun) -> TaskRun:
        with self._sessions.begin() as session:
            if session.get(BackgroundTaskRunRow, task.task_run_id) is None:
                session.add(BackgroundTaskRunRow(
                    task_run_id=task.task_run_id, task_type=task.task_type, state=task.state.value,
                    fencing_token=task.fencing_token, attempt=task.attempt, checkpoints={},
                ))
        return task

    def get(self, task_run_id: str) -> TaskRun | None:
        with self._sessions() as session:
            row = session.get(BackgroundTaskRunRow, task_run_id)
            return self._domain(row) if row else None

    def put(self, task: TaskRun) -> TaskRun:
        """Persist an already-fenced immutable state transition."""
        with self._sessions.begin() as session:
            row = session.scalar(
                select(BackgroundTaskRunRow).where(
                    BackgroundTaskRunRow.task_run_id == task.task_run_id,
                ).with_for_update()
            )
            if row is None:
                raise KeyError(task.task_run_id)
            if row.fencing_token != task.fencing_token:
                raise ValueError("stale fencing token")
            row.state = task.state.value
            row.owner = task.owner
            row.lease_expires_at = task.lease_expires_at
            row.attempt = task.attempt
            row.checkpoints = {item.name: item.payload for item in task.checkpoints}
        return task

    def acquire(
        self, task_run_id: str, owner: str, *, now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> TaskRun:
        now = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(BackgroundTaskRunRow).where(BackgroundTaskRunRow.task_run_id == task_run_id).with_for_update()
            )
            if row is None:
                raise KeyError(task_run_id)
            task = self._domain(row).acquire_lease(owner, now=now, ttl=ttl)
            row.state, row.owner = task.state.value, task.owner
            row.lease_expires_at, row.fencing_token = task.lease_expires_at, task.fencing_token
            row.updated_at = now
            return task

    def save_checkpoint(
        self, task_run_id: str, owner: str, fencing_token: int, checkpoint: Checkpoint, *,
        now: datetime | None = None,
    ) -> TaskRun:
        now = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(BackgroundTaskRunRow).where(
                    BackgroundTaskRunRow.task_run_id == task_run_id,
                ).with_for_update()
            )
            if row is None:
                raise KeyError(task_run_id)
            task = self._domain(row).save_checkpoint(owner, fencing_token, checkpoint, now=now)
            row.checkpoints = {item.name: item.payload for item in task.checkpoints}
            row.updated_at = now
            return task

    def renew(
        self,
        task_run_id: str,
        owner: str,
        fencing_token: int,
        *,
        now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> TaskRun:
        now = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(BackgroundTaskRunRow).where(
                    BackgroundTaskRunRow.task_run_id == task_run_id,
                ).with_for_update()
            )
            if row is None:
                raise KeyError(task_run_id)
            task = self._domain(row).renew_lease(owner, fencing_token, now=now, ttl=ttl)
            row.lease_expires_at = task.lease_expires_at
            row.updated_at = now
            return task

    def transition(
        self,
        task_run_id: str,
        state: TaskRunState | str,
        owner: str,
        fencing_token: int,
        *,
        now: datetime | None = None,
    ) -> TaskRun:
        now = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(BackgroundTaskRunRow).where(
                    BackgroundTaskRunRow.task_run_id == task_run_id,
                ).with_for_update()
            )
            if row is None:
                raise KeyError(task_run_id)
            task = self._domain(row).transition(state, owner, fencing_token, now=now)
            row.state = task.state.value
            row.owner = task.owner
            row.lease_expires_at = task.lease_expires_at
            row.attempt = task.attempt
            row.updated_at = now
            return task

    def operator_action(
        self, task_run_id: str, action: OperatorAction | str, *, actor: str, reason: str,
        request_id: str, now: datetime | None = None,
    ) -> TaskRun:
        now = now or datetime.now(UTC)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(BackgroundTaskRunRow).where(
                    BackgroundTaskRunRow.task_run_id == task_run_id,
                ).with_for_update()
            )
            if row is None:
                raise KeyError(task_run_id)
            task = self._domain(row).operator_action(action, actor, now=now)
            row.state = task.state.value
            row.owner = task.owner
            row.lease_expires_at = task.lease_expires_at
            row.attempt = task.attempt
            action_value = OperatorAction(action).value
            session.add(OperatorActionAuditRow(
                audit_id=f"audit_{uuid4().hex}", task_run_id=task_run_id, action=action_value, actor=actor,
                reason=reason, request_id=request_id, created_at=now,
            ))
            return task


__all__ = ["PostgresTaskRunRepository"]

"""Lease/fencing state machine for long-running operator tasks."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4


class TaskRunState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"


class OperatorAction(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    RETRY = "retry"
    INVALIDATE = "invalidate"
    REBUILD = "rebuild"


class LeaseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Checkpoint:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class TaskAudit:
    action: str
    at: datetime
    actor: str
    detail: str = ""
    fencing_token: int | None = None


@dataclass(frozen=True, slots=True)
class TaskRun:
    task_run_id: str
    task_type: str
    state: TaskRunState = TaskRunState.PENDING
    owner: str | None = None
    lease_expires_at: datetime | None = None
    fencing_token: int = 0
    checkpoints: tuple[Checkpoint, ...] = ()
    audit: tuple[TaskAudit, ...] = ()
    attempt: int = 0

    @classmethod
    def create(cls, task_type: str, task_run_id: str | None = None) -> "TaskRun":
        return cls(task_run_id or f"run_{uuid4().hex}", task_type)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    def _audit(
        self, action: str, actor: str, detail: str = "", *, token: int | None = None, now: datetime,
    ) -> "TaskRun":
        return replace(self, audit=self.audit + (TaskAudit(action, now, actor, detail, token),))

    def acquire_lease(
        self, owner: str, *, now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> "TaskRun":
        now = self._utc(now or datetime.now(UTC))
        if self.state in {TaskRunState.SUCCEEDED, TaskRunState.INVALIDATED}:
            raise LeaseError(f"cannot lease terminal task: {self.state}")
        # SQLite returns TIMESTAMP columns without tzinfo even when the
        # application persisted UTC-aware values.  Normalize both sides at
        # the domain boundary so lease takeover remains deterministic across
        # PostgreSQL and SQLite without weakening the expiry check.
        lease_expires_at = self._utc(self.lease_expires_at) if self.lease_expires_at else None
        if self.owner and lease_expires_at and lease_expires_at > now and self.owner != owner:
            raise LeaseError("task lease is held by another worker")
        updated = replace(
            self, state=TaskRunState.RUNNING, owner=owner, lease_expires_at=now + ttl,
            fencing_token=self.fencing_token + 1,
        )
        return updated._audit("lease_acquired", owner, token=updated.fencing_token, now=now)

    def renew_lease(
        self, owner: str, fencing_token: int, *, now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> "TaskRun":
        now = self._utc(now or datetime.now(UTC))
        self.require_lease(owner, fencing_token, now=now)
        return replace(self, lease_expires_at=now + ttl)._audit("lease_renewed", owner, token=fencing_token, now=now)

    def require_lease(self, owner: str, fencing_token: int, *, now: datetime | None = None) -> None:
        now = self._utc(now or datetime.now(UTC))
        if self.owner != owner or self.fencing_token != fencing_token:
            raise LeaseError("stale fencing token or owner")
        expires_at = self._utc(self.lease_expires_at) if self.lease_expires_at else None
        if expires_at is None or expires_at <= now:
            raise LeaseError("task lease has expired")

    def release_lease(self, owner: str, fencing_token: int, *, now: datetime | None = None) -> "TaskRun":
        now = self._utc(now or datetime.now(UTC))
        self.require_lease(owner, fencing_token, now=now)
        return replace(self, owner=None, lease_expires_at=None)._audit(
            "lease_released", owner, token=fencing_token, now=now,
        )

    def save_checkpoint(
        self, owner: str, fencing_token: int, checkpoint: Checkpoint, *, now: datetime | None = None,
    ) -> "TaskRun":
        now = self._utc(now or datetime.now(UTC))
        self.require_lease(owner, fencing_token, now=now)
        return replace(self, checkpoints=self.checkpoints + (checkpoint,))._audit(
            "checkpoint", owner, checkpoint.name, token=fencing_token, now=now,
        )

    def transition(
        self, state: TaskRunState | str, owner: str, fencing_token: int, *, now: datetime | None = None,
    ) -> "TaskRun":
        now = self._utc(now or datetime.now(UTC))
        self.require_lease(owner, fencing_token, now=now)
        target = TaskRunState(state)
        allowed = {
            TaskRunState.RUNNING: {
                TaskRunState.SUCCEEDED, TaskRunState.FAILED, TaskRunState.PAUSED, TaskRunState.RETRYING,
            },
            TaskRunState.RETRYING: {TaskRunState.RUNNING, TaskRunState.FAILED},
            TaskRunState.PAUSED: {TaskRunState.RUNNING, TaskRunState.INVALIDATED},
        }
        if target not in allowed.get(self.state, set()):
            raise ValueError(f"invalid task transition {self.state} -> {target}")
        # Terminal completion releases the worker lease while retaining the
        # monotonically increasing fencing token.  This permits a failed run
        # to be resumed by another worker, while stale owners still cannot
        # write because the token and owner are checked before transition.
        released = target in {TaskRunState.SUCCEEDED, TaskRunState.FAILED}
        updated = replace(
            self,
            state=target,
            owner=None if released else self.owner,
            lease_expires_at=None if released else self.lease_expires_at,
            attempt=self.attempt + (1 if target is TaskRunState.RETRYING else 0),
        )
        return updated._audit("state_changed", owner, target.value, token=fencing_token, now=now)

    def operator_action(self, action: OperatorAction | str, actor: str, *, now: datetime | None = None) -> "TaskRun":
        now = self._utc(now or datetime.now(UTC))
        action = OperatorAction(action)
        if action is OperatorAction.INVALIDATE:
            return replace(self, state=TaskRunState.INVALIDATED, owner=None, lease_expires_at=None)._audit(
                action.value, actor, now=now,
            )
        if action is OperatorAction.REBUILD:
            return replace(
                self, state=TaskRunState.RETRYING, owner=None, lease_expires_at=None, attempt=self.attempt + 1,
            )._audit(action.value, actor, now=now)
        target = {
            OperatorAction.PAUSE: TaskRunState.PAUSED,
            OperatorAction.RESUME: TaskRunState.RUNNING,
            OperatorAction.RETRY: TaskRunState.RETRYING,
        }[action]
        return replace(self, state=target)._audit(action.value, actor, now=now)


__all__ = ["Checkpoint", "LeaseError", "OperatorAction", "TaskAudit", "TaskRun", "TaskRunState"]

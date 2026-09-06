"""Application service and in-memory port implementation for task leases."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from stock_content.domain.task_run import Checkpoint, OperatorAction, TaskRun, TaskRunState


class TaskRunRepository(Protocol):
    def get(self, task_run_id: str) -> TaskRun | None: ...
    def put(self, task: TaskRun) -> TaskRun: ...

    # Durable repositories may implement these atomic operations so the
    # fencing check and write happen in one database transaction.  The
    # service retains the put-based fallback for the in-memory/test port.
    def create(self, task: TaskRun) -> TaskRun: ...
    def acquire(
        self, task_run_id: str, owner: str, *, now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> TaskRun: ...

    def save_checkpoint(
        self, task_run_id: str, owner: str, fencing_token: int, checkpoint: Checkpoint, *,
        now: datetime | None = None,
    ) -> TaskRun: ...

    def renew(
        self, task_run_id: str, owner: str, fencing_token: int, *, now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> TaskRun: ...

    def transition(
        self, task_run_id: str, state: TaskRunState | str, owner: str, fencing_token: int, *,
        now: datetime | None = None,
    ) -> TaskRun: ...


class InMemoryTaskRunRepository:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRun] = {}

    def get(self, task_run_id: str) -> TaskRun | None:
        return self._tasks.get(task_run_id)

    def put(self, task: TaskRun) -> TaskRun:
        self._tasks[task.task_run_id] = task
        return task


class TaskLeaseService:
    def __init__(self, repository: TaskRunRepository | None = None) -> None:
        self.repository = repository or InMemoryTaskRunRepository()

    def create(self, task_type: str, task_run_id: str | None = None) -> TaskRun:
        task = TaskRun.create(task_type, task_run_id)
        create = getattr(self.repository, "create", None)
        return create(task) if create else self.repository.put(task)

    def acquire(
        self, task_run_id: str, owner: str, *, now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> TaskRun:
        acquire = getattr(self.repository, "acquire", None)
        if acquire:
            return acquire(task_run_id, owner, now=now, ttl=ttl)
        task = self._get(task_run_id)
        return self.repository.put(task.acquire_lease(owner, now=now, ttl=ttl))

    def renew(
        self, task_run_id: str, owner: str, fencing_token: int, *, now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=5),
    ) -> TaskRun:
        renew = getattr(self.repository, "renew", None)
        if renew:
            return renew(task_run_id, owner, fencing_token, now=now, ttl=ttl)
        return self._save(task_run_id, lambda t: t.renew_lease(owner, fencing_token, now=now, ttl=ttl))

    def checkpoint(
        self, task_run_id: str, owner: str, fencing_token: int, checkpoint: Checkpoint, *, now: datetime | None = None,
    ) -> TaskRun:
        save_checkpoint = getattr(self.repository, "save_checkpoint", None)
        if save_checkpoint:
            return save_checkpoint(task_run_id, owner, fencing_token, checkpoint, now=now)
        return self._save(task_run_id, lambda t: t.save_checkpoint(owner, fencing_token, checkpoint, now=now))

    def transition(
        self, task_run_id: str, state: TaskRunState | str, owner: str, fencing_token: int, *,
        now: datetime | None = None,
    ) -> TaskRun:
        transition = getattr(self.repository, "transition", None)
        if transition:
            return transition(task_run_id, state, owner, fencing_token, now=now)
        return self._save(task_run_id, lambda t: t.transition(state, owner, fencing_token, now=now))

    def operator(
        self, task_run_id: str, action: OperatorAction | str, actor: str, *, reason: str = "",
        request_id: str = "", now: datetime | None = None,
    ) -> TaskRun:
        """Apply an operator action; reason/request_id are audit metadata."""
        operator_action = getattr(self.repository, "operator_action", None)
        if operator_action:
            return operator_action(
                task_run_id, action, actor=actor, reason=reason, request_id=request_id, now=now,
            )
        detail = f"reason={reason};request_id={request_id}"
        return self._save(task_run_id, lambda t: t.operator_action(action, actor, now=now)._audit(
            "operator_request", actor, detail, now=now,
        ))

    def _get(self, task_run_id: str) -> TaskRun:
        task = self.repository.get(task_run_id)
        if task is None:
            raise KeyError(task_run_id)
        return task

    def _save(self, task_run_id: str, update) -> TaskRun:
        return self.repository.put(update(self._get(task_run_id)))


__all__ = ["InMemoryTaskRunRepository", "TaskLeaseService", "TaskRunRepository"]

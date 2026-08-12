from __future__ import annotations

from uuid import uuid4

from stock_content.domain.models import ContentTask


class ContentApplication:
    """Application facade; media/storage adapters are injected in later phases."""

    def __init__(self) -> None:
        self._tasks: dict[str, ContentTask] = {}

    def enqueue(self, source_type: str, source_ref: str, options: dict | None = None) -> dict:
        task = ContentTask(task_id=uuid4().hex, source_type=source_type, source_ref=source_ref)
        task.result = {"options": options or {}}
        self._tasks[task.task_id] = task
        return {"task_id": task.task_id, "status": task.status, "stage": task.stage}

    def get_task(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return {"task_id": task.task_id, "status": task.status, "stage": task.stage, **task.result}

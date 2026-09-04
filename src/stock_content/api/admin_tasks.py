"""Admin task operations adapter; persistence wiring is intentionally separate."""
from __future__ import annotations

from fastapi import APIRouter

from stock_content.application.task_lease_service import TaskLeaseService


def create_admin_tasks_router(service: TaskLeaseService | None = None) -> APIRouter:
    task_service = service or TaskLeaseService()
    router = APIRouter(prefix="/admin/tasks", tags=["admin"])

    @router.post("")
    def create_task(task_type: str) -> dict[str, object]:
        task = task_service.create(task_type)
        return {"task_run_id": task.task_run_id, "state": task.state.value}

    @router.post("/{task_run_id}/{action}")
    def operator_action(
        task_run_id: str, action: str, actor: str, reason: str, request_id: str,
    ) -> dict[str, object]:
        task = task_service.operator(task_run_id, action, actor, reason=reason, request_id=request_id)
        return {"task_run_id": task.task_run_id, "action": action}

    return router


router = create_admin_tasks_router()
__all__ = ["create_admin_tasks_router", "router"]

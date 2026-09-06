from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.task_run_repository import PostgresTaskRunRepository
from stock_content.application.task_lease_service import TaskLeaseService
from stock_content.domain.task_run import Checkpoint, LeaseError


@pytest.mark.parametrize("task_type", ("verification", "lifecycle", "backfill"))
def test_expired_lease_takeover_fences_crashed_worker_and_preserves_one_effect(tmp_path, task_type):
    database = Database(f"sqlite:///{tmp_path / f'{task_type}.db'}")
    database.create_schema()
    service = TaskLeaseService(PostgresTaskRunRepository(database.session_factory))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    task = service.create(task_type, f"{task_type}:claim-1")

    first = service.acquire(task.task_run_id, "worker-a", now=start, ttl=timedelta(seconds=10))
    with pytest.raises(LeaseError, match="held by another worker"):
        service.acquire(task.task_run_id, "worker-b", now=start + timedelta(seconds=9))

    recovered = service.acquire(task.task_run_id, "worker-b", now=start + timedelta(seconds=11))
    assert recovered.fencing_token == first.fencing_token + 1

    with pytest.raises(LeaseError, match="stale fencing token or owner"):
        service.checkpoint(
            task.task_run_id,
            "worker-a",
            first.fencing_token,
            Checkpoint("stale-effect"),
            now=start + timedelta(seconds=11),
        )
    with pytest.raises(LeaseError, match="stale fencing token or owner"):
        service.transition(
            task.task_run_id,
            "SUCCEEDED",
            "worker-a",
            first.fencing_token,
            now=start + timedelta(seconds=11),
        )

    service.checkpoint(
        task.task_run_id,
        "worker-b",
        recovered.fencing_token,
        Checkpoint("effect", {"claim_id": "claim-1"}),
        now=start + timedelta(seconds=11),
    )
    completed = service.transition(
        task.task_run_id,
        "SUCCEEDED",
        "worker-b",
        recovered.fencing_token,
        now=start + timedelta(seconds=11),
    )
    restarted = TaskLeaseService(PostgresTaskRunRepository(database.session_factory))
    durable = restarted.repository.get(task.task_run_id)

    assert completed.state == "SUCCEEDED"
    assert durable is not None
    assert durable.state == completed.state
    assert durable.fencing_token == completed.fencing_token
    assert [checkpoint.name for checkpoint in durable.checkpoints] == ["effect"]

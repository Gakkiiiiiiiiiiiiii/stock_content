"""Real PostgreSQL idempotency coverage for legal migration replay requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from sqlalchemy import select

from stock_content.adapters.postgres.models import ContentTaskRow
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.application.replay.identity import (
    migration_replay_idempotency_key,
    migration_replay_request_identity,
)
from stock_content.application.replay_service import ReplayService
from stock_content.domain.models import ContentTask


def test_postgres_concurrent_identical_legal_migration_replay_creates_one_task(postgres_database):
    repository = PostgresContentTaskRepository(postgres_database.session_factory)
    request_identity = migration_replay_request_identity(
        "cs-concurrent-replay",
        "pipeline.v4.concurrent",
        {"transcript": "legal result-affecting override"},
        runtime_option_keys=ReplayService._RUNTIME_OPTIONS,
    )

    def create(_: int) -> ContentTask:
        return repository.create(ContentTask(
            task_id=f"pg-replay-{uuid4().hex}",
            source_type="fixture",
            source_ref="concurrent-replay",
            task_kind="replay",
            status="RUNNING",
            input_hash=request_identity,
            request_hash=request_identity,
            idempotency_key=migration_replay_idempotency_key(request_identity),
        ))

    with ThreadPoolExecutor(max_workers=8) as workers:
        returned = list(workers.map(create, range(8)))

    assert len({task.task_id for task in returned}) == 1
    with postgres_database.session_factory() as session:
        rows = list(session.scalars(select(ContentTaskRow)))
    assert len(rows) == 1
    assert rows[0].request_hash == request_identity

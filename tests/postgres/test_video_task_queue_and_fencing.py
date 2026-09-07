"""Real PostgreSQL race coverage for SC-E2E-P0-03 (opt-in)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event
from time import sleep

import pytest
from sqlalchemy import select

from stock_content.adapters.postgres.models import ContentTaskEffectRow, ContentTaskRow
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.application.fenced_effects import EffectIntent, PostgresFencedEffectUnitOfWork
from stock_content.application.knowledge_projection_dispatcher import KnowledgeProjectionDispatcher
from stock_content.domain.models import ContentTask
from stock_content.domain.worker_capability import TaskKind
from stock_content.ports.repositories import StaleTaskLease


def test_postgres_skip_locked_race_and_stale_worker_cannot_commit_effect(postgres_database):
    repository = PostgresContentTaskRepository(postgres_database.session_factory)
    repository.create(ContentTask(
        task_id="pg-video-task", source_type="bilibili", source_ref="BV1pg", task_kind="video_pipeline",
    ))

    def claim(worker_id: str):
        return repository.claim_pending(worker_id, TaskKind.VIDEO_PIPELINE.value, 60)

    with ThreadPoolExecutor(max_workers=2) as workers:
        claimed = list(workers.map(claim, ("video-a", "video-b")))
    winner = next(task for task in claimed if task is not None)
    assert sum(task is not None for task in claimed) == 1

    with postgres_database.session_factory.begin() as session:
        row = session.get(ContentTaskRow, winner.task_id)
        row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)

    recovered = repository.claim_pending("video-recovery", TaskKind.VIDEO_PIPELINE.value, 60)
    assert recovered is not None and recovered.fencing_token == winner.fencing_token + 1
    with pytest.raises(StaleTaskLease):
        repository.commit_effect(winner.task_id, {"content_snapshot_id": "old"}, "video-a", winner.fencing_token)
    repository.commit_effect(
        recovered.task_id, {"content_snapshot_id": "recovered"}, "video-recovery", recovered.fencing_token,
    )
    assert repository.get(recovered.task_id).result == {"content_snapshot_id": "recovered"}


def test_postgres_fenced_effect_transaction_rejects_takeover(postgres_database):
    repository = PostgresContentTaskRepository(postgres_database.session_factory)
    repository.create(ContentTask("pg-effect", "bilibili", "BV1pg-effect", task_kind="video_pipeline"))
    first = repository.claim_pending("video-a", TaskKind.VIDEO_PIPELINE.value, 60)
    assert first is not None
    effects = PostgresFencedEffectUnitOfWork(postgres_database.session_factory)
    intent = EffectIntent("publication:snapshot-1", "SNAPSHOT_PUBLICATION", {"snapshot_id": "snapshot-1"})
    effects.execute_sql(first.task_id, "video-a", first.fencing_token, intent, lambda _session: None)
    with postgres_database.session_factory.begin() as session:
        session.get(ContentTaskRow, first.task_id).lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    recovered = repository.claim_pending("video-b", TaskKind.VIDEO_PIPELINE.value, 60)
    assert recovered is not None
    with pytest.raises(StaleTaskLease):
        effects.execute_sql(first.task_id, "video-a", first.fencing_token, intent, lambda _session: None)


def test_postgres_projection_dispatch_holds_row_lock_across_external_effect(postgres_database):
    """The skip-locked claimant cannot pass an in-flight effect seam.

    This is intentionally PostgreSQL-gated: SQLite's lock model cannot prove
    that the row lock remains effective while the remote index call is in
    flight.  The worker that passed the final fence may finish after its wall
    clock lease expires because the row lock prevents a concurrent takeover.
    """
    repository = PostgresContentTaskRepository(postgres_database.session_factory)
    repository.create(ContentTask("pg-projection", "bilibili", "BV1pg-projection", task_kind="video_pipeline"))
    task = repository.claim_pending("ingest-worker", TaskKind.VIDEO_PIPELINE.value, 60)
    assert task is not None
    effects = PostgresFencedEffectUnitOfWork(postgres_database.session_factory)
    effects.prepare_external(
        task.task_id,
        "ingest-worker",
        task.fencing_token,
        EffectIntent("index:pg-lock", "KNOWLEDGE_INDEX", {"knowledge_ids": []}),
    )
    started, release = Event(), Event()
    calls: list[str] = []

    class BlockingIndex:
        def index(self, _units, *, idempotency_key=None):
            calls.append(str(idempotency_key))
            started.set()
            assert release.wait(timeout=10)

    dispatcher = KnowledgeProjectionDispatcher(postgres_database.session_factory, BlockingIndex())
    # This test targets the row lock, not knowledge hydration; the durable
    # intent is the object under test and the SQLite suite covers hydration.
    dispatcher._load_units = lambda _effect_id: []  # type: ignore[method-assign]  # noqa: SLF001
    dispatcher._load_units_in_session = lambda _session, _effect: []  # type: ignore[method-assign]  # noqa: SLF001
    claim = dispatcher._claim_due("old-worker", limit=1, lease_seconds=1)[0]  # noqa: SLF001
    with ThreadPoolExecutor(max_workers=1) as workers:
        active = workers.submit(dispatcher._dispatch_claim, claim, "old-worker")  # noqa: SLF001
        assert started.wait(timeout=5)
        sleep(1.1)
        # PostgreSQL SKIP LOCKED leaves the in-flight row alone even though
        # its wall-clock lease has elapsed; the new worker cannot call index.
        assert dispatcher.dispatch_due("new-worker") == {
            "dispatched": 0, "retried": 0, "dead_lettered": 0, "pending": 1,
        }
        release.set()
        assert active.result(timeout=10) == "DISPATCHED"

    assert len(calls) == 1
    with postgres_database.session_factory() as session:
        effect = session.scalar(
            select(ContentTaskEffectRow).where(ContentTaskEffectRow.effect_key == "index:pg-lock")
        )
        assert effect and effect.state == "COMPLETED"

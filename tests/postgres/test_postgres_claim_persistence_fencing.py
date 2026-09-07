"""Opt-in PostgreSQL takeover coverage for the pre-snapshot claim effect."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from stock_content.adapters.postgres.models import ContentTaskRow
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.application.fenced_effects import PostgresFencedEffectUnitOfWork
from stock_content.application.pipeline import PipelineContext, PipelineState
from stock_content.application.stages import ClaimPersistenceStage
from stock_content.domain.models import ContentTask
from stock_content.domain.worker_capability import TaskKind
from stock_content.ports.repositories import StaleTaskLease


class _Registry:
    evidence = SimpleNamespace(artifact_id="evidence-pg")
    claims = SimpleNamespace(artifact_id="claims-pg")

    @staticmethod
    def get(_slot: str):
        return None


class _Artifacts:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def put(self, _item) -> None:
        self.calls.append("put")

    def put_claim_members(self, _item) -> None:
        self.calls.append("members")


class _Claims:
    def __init__(self) -> None:
        self.calls = 0

    def save(self, _claim) -> None:
        self.calls += 1


def _context(task_id: str, worker: str, token: int) -> PipelineContext:
    context = PipelineContext(
        task_id=task_id,
        source={"type": "bilibili", "ref": "BV1pg"},
        state=PipelineState(claims=[SimpleNamespace(claim_id="claim-pg")]),
        worker_id=worker,
        fencing_token=token,
    )
    context.artifacts = _Registry()
    return context


def test_postgres_claim_persistence_takeover_writes_zero_stale_effects(postgres_database):
    tasks = PostgresContentTaskRepository(postgres_database.session_factory)
    tasks.create(ContentTask("pg-claim-persistence", "bilibili", "BV1pg", task_kind="video_pipeline"))
    first = tasks.claim_pending("worker-a", TaskKind.VIDEO_PIPELINE.value, 60)
    assert first is not None
    with postgres_database.session_factory.begin() as session:
        session.get(ContentTaskRow, first.task_id).lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    second = tasks.claim_pending("worker-b", TaskKind.VIDEO_PIPELINE.value, 60)
    assert second is not None

    artifacts, claims = _Artifacts(), _Claims()
    stage = ClaimPersistenceStage(claims, artifacts, PostgresFencedEffectUnitOfWork(postgres_database.session_factory))
    with pytest.raises(StaleTaskLease):
        stage.execute(_context(first.task_id, "worker-a", first.fencing_token))
    assert artifacts.calls == [] and claims.calls == 0

    stage.execute(_context(second.task_id, "worker-b", second.fencing_token))
    assert artifacts.calls == ["put", "put", "members"] and claims.calls == 1

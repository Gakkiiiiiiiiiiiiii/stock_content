from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentTaskRow
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.application.fenced_effects import PostgresFencedEffectUnitOfWork
from stock_content.application.pipeline import PipelineContext, PipelineState
from stock_content.application.stages import ClaimPersistenceStage
from stock_content.domain.models import ContentTask
from stock_content.domain.worker_capability import TaskKind
from stock_content.ports.repositories import StaleTaskLease


class _Registry:
    def __init__(self) -> None:
        self.evidence = SimpleNamespace(artifact_id="evidence-1")
        self.claims = SimpleNamespace(artifact_id="claims-1")

    def get(self, _slot: str):
        return None


class _Artifacts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def put(self, value) -> None:
        self.calls.append(("put", value))

    def put_claim_members(self, value) -> None:
        self.calls.append(("members", value))


class _Claims:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def save(self, claim) -> None:
        self.calls.append(claim.claim_id)


def _context(task_id: str, worker_id: str, fencing_token: int) -> PipelineContext:
    context = PipelineContext(
        task_id=task_id,
        source={"type": "bilibili", "ref": "BV1fence"},
        state=PipelineState(claims=[SimpleNamespace(claim_id="claim-1")]),
        worker_id=worker_id,
        fencing_token=fencing_token,
    )
    context.artifacts = _Registry()
    return context


def test_claim_persistence_fence_rejects_stale_attempt_and_dedupes_resume(tmp_path) -> None:
    """The actual stage writes zero pre-snapshot effects after lease takeover."""
    database = Database(f"sqlite:///{tmp_path / 'claim-fence.db'}")
    database.create_schema()
    tasks = PostgresContentTaskRepository(database.session_factory)
    tasks.create(ContentTask("claim-task", "bilibili", "BV1fence", task_kind=TaskKind.VIDEO_PIPELINE.value))
    first = tasks.claim_pending("worker-a", TaskKind.VIDEO_PIPELINE.value, 60)
    assert first is not None
    with database.session_factory.begin() as session:
        session.get(ContentTaskRow, first.task_id).lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    second = tasks.claim_pending("worker-b", TaskKind.VIDEO_PIPELINE.value, 60)
    assert second is not None and second.fencing_token == first.fencing_token + 1

    artifacts, claims = _Artifacts(), _Claims()
    stage = ClaimPersistenceStage(claims, artifacts, PostgresFencedEffectUnitOfWork(database.session_factory))
    with pytest.raises(StaleTaskLease):
        stage.execute(_context(first.task_id, "worker-a", first.fencing_token))
    assert artifacts.calls == [] and claims.calls == []

    valid = _context(second.task_id, "worker-b", second.fencing_token)
    stage.execute(valid)
    assert valid.state.claims_persisted is True
    assert [kind for kind, _value in artifacts.calls] == ["put", "put", "members"]
    assert claims.calls == ["claim-1"]

    # A crash after the durable effect receipt and before the next checkpoint
    # re-enters the stage with the same identity and commits no duplicate rows.
    resumed = _context(second.task_id, "worker-b", second.fencing_token)
    stage.execute(resumed)
    assert resumed.state.claims_persisted is True
    assert len(artifacts.calls) == 3 and claims.calls == ["claim-1"]
    database.engine.dispose()


def test_claim_persistence_refuses_partial_queue_fence_before_writing() -> None:
    artifacts, claims = _Artifacts(), _Claims()
    stage = ClaimPersistenceStage(claims, artifacts, fenced_effects=None)
    context = _context("claim-task", "worker-a", 1)
    with pytest.raises(RuntimeError, match="CLAIM_PERSISTENCE_FENCED_UOW_REQUIRED"):
        stage.execute(context)
    assert artifacts.calls == [] and claims.calls == []

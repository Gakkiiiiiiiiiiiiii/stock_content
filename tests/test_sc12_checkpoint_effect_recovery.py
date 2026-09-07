"""SC-E2E-P1-03 checkpoint sealing and fenced recovery regressions."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentStageCheckpointRow, ContentTaskRow
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stage_runner import StageResult, StageRunner
from stock_content.domain.artifacts import ArtifactRegistry, SourceArtifact
from stock_content.domain.checkpoint import CheckpointRecord, CheckpointValidationError, validate_resume
from stock_content.domain.models import ContentTask
from stock_content.ports.repositories import StaleTaskLease


class SealedStage:
    name = "sealed"
    required_inputs = ("source",)
    output_types = ()
    calls = 0

    def execute(self, context):
        type(self).calls += 1
        return StageResult(context=context)


def _claimed(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'checkpoint.db'}")
    database.create_schema()
    tasks = PostgresContentTaskRepository(database.session_factory)
    task = tasks.create(ContentTask(task_id="sc12-task", source_type="bilibili", source_ref="BV-sc12"))
    lease = tasks.claim_pending("worker-one", "video_pipeline", 60)
    assert lease is not None
    return database, tasks, task, lease


def _source() -> SourceArtifact:
    return SourceArtifact(
        artifact_id="source-sc12",
        artifact_type="source",
        source_type="bilibili",
        source_ref="BV-sc12",
        source_identity_hash="public-source-id",
        source_version_id="public-version-id",
        source_content_hash="raw-hash",
    )


def test_checkpoint_seals_hash_chain_identity_and_fence_without_locator(tmp_path):
    database, _tasks, task, lease = _claimed(tmp_path)
    artifacts = SqlArtifactRepository(database.session_factory)
    source = _source()
    artifacts.put(source)
    context = PipelineContext(
        task_id=task.task_id,
        source={"type": task.source_type, "ref": task.source_ref},
        artifacts=ArtifactRegistry(source=source),
        worker_id="worker-one",
        fencing_token=lease.fencing_token,
    )

    SealedStage.calls = 0
    StageRunner(SealedStage(), artifact_repository=artifacts, legacy_fallback=False).execute(context)
    record = context.checkpoints[-1]

    assert record.input_artifact_ids == (source.artifact_id,)
    assert record.input_hashes == (source.content_hash,)
    assert record.public_materialization_hash
    assert record.model_identity == {"asr": "faster-whisper", "asr_version": "1.0"}
    assert record.worker_id == "worker-one"
    assert record.fencing_token == lease.fencing_token
    assert record.state_checksum
    assert "signature" not in str(record.to_dict()).lower()
    validate_resume([record], {source.artifact_id: source})

    tampered = CheckpointRecord.from_dict({**record.to_dict(), "input_hashes": ["tampered"]})
    with pytest.raises(CheckpointValidationError, match="checksum|input artifact"):
        validate_resume([tampered], {source.artifact_id: source})


def test_stale_worker_cannot_persist_checkpoint_after_takeover(tmp_path):
    database, tasks, task, first = _claimed(tmp_path)
    artifacts = SqlArtifactRepository(database.session_factory)
    source = _source()
    artifacts.put(source)
    context = PipelineContext(
        task_id=task.task_id,
        source={"type": task.source_type, "ref": task.source_ref},
        artifacts=ArtifactRegistry(source=source),
        worker_id="worker-one",
        fencing_token=first.fencing_token,
    )
    with database.session_factory.begin() as session:
        row = session.get(ContentTaskRow, task.task_id)
        row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    second = tasks.claim_pending("worker-two", "video_pipeline", 60)
    assert second is not None and second.fencing_token == first.fencing_token + 1

    with pytest.raises(StaleTaskLease):
        StageRunner(SealedStage(), artifact_repository=artifacts, legacy_fallback=False).execute(context)
    with database.session_factory() as session:
        rows = session.scalars(
            select(ContentStageCheckpointRow).where(ContentStageCheckpointRow.task_id == task.task_id)
        ).all()
    assert rows == []

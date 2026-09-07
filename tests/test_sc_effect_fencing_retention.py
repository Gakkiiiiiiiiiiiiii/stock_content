from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ContentArtifactRow,
    ContentTaskRow,
    RetentionArtifactLocatorRow,
    SourceArtifactMetadataRow,
)
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.adapters.retention.in_memory_tombstones import InMemoryTombstoneRepository
from stock_content.adapters.retention.safe_filesystem import SafeFilesystemArtifactDeleter
from stock_content.adapters.retention.sql_candidates import SqlRetentionCandidateRepository
from stock_content.adapters.retention.sql_retention import SqlRetentionExecutionRepository
from stock_content.application.fenced_effects import EffectIntent, PostgresFencedEffectUnitOfWork
from stock_content.application.retention_scheduler import RetentionScheduler
from stock_content.application.retention_service import RetentionService
from stock_content.domain.models import ContentTask
from stock_content.domain.retention import (
    RetentionCandidate,
    RetentionClass,
    RetentionExecutionState,
    RetentionPolicy,
)
from stock_content.domain.worker_capability import TaskKind
from stock_content.ports.repositories import StaleTaskLease

NOW = datetime(2026, 9, 7, tzinfo=UTC)


def _candidate(*, legal_hold: bool = False) -> RetentionCandidate:
    return RetentionCandidate(
        "artifact-1", RetentionClass.RAW_MEDIA, "a" * 64, "b" * 64, "lineage-1", NOW - timedelta(days=8), legal_hold
    )


def _policy() -> RetentionPolicy:
    return RetentionPolicy({kind: 7 for kind in RetentionClass})


class _RecordingDeleter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def delete(self, artifact_id: str, *, idempotency_key: str) -> None:
        self.calls.append((artifact_id, idempotency_key))
        if self.fail:
            raise OSError("private-locator-must-not-persist")


def _task_repository(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'effects.db'}")
    database.create_schema()
    repository = PostgresContentTaskRepository(database.session_factory)
    repository.create(ContentTask("task-1", "bilibili", "BV1fence", task_kind="video_pipeline"))
    return database, repository


def test_fenced_sql_and_external_effects_reject_stale_worker_and_resume_once(tmp_path: Path) -> None:
    database, tasks = _task_repository(tmp_path)
    first = tasks.claim_pending("worker-a", TaskKind.VIDEO_PIPELINE.value, 60)
    assert first is not None
    effects = PostgresFencedEffectUnitOfWork(database.session_factory)
    sql_intent = EffectIntent("persist:video-1", "PERSIST_PROJECTION", {"video_id": "video-1"})
    effects.execute_sql(
        first.task_id,
        "worker-a",
        first.fencing_token,
        sql_intent,
        lambda session: setattr(session.get(ContentTaskRow, first.task_id), "result", {"video_id": "video-1"}),
    )

    # Takeover happens after an expired lease. The old fence cannot alter a
    # projection or dispatch an index intent after this point.
    with database.session_factory.begin() as session:
        session.get(ContentTaskRow, first.task_id).lease_expires_at = NOW - timedelta(seconds=1)
    second = tasks.claim_pending("worker-b", TaskKind.VIDEO_PIPELINE.value, 60)
    assert second is not None and second.fencing_token == first.fencing_token + 1
    with pytest.raises(StaleTaskLease):
        effects.execute_sql(
            first.task_id,
            "worker-a",
            first.fencing_token,
            EffectIntent("persist:stale", "PERSIST_PROJECTION", {"video_id": "stale"}),
            lambda session: setattr(session.get(ContentTaskRow, first.task_id), "result", {"video_id": "stale"}),
        )

    dispatched: list[str] = []
    external = EffectIntent("index:knowledge-1", "KNOWLEDGE_INDEX", {"snapshot_id": "snapshot-1"})
    with pytest.raises(StaleTaskLease):
        effects.dispatch_external(first.task_id, "worker-a", first.fencing_token, external, dispatched.append)
    assert dispatched == []
    effects.dispatch_external(second.task_id, "worker-b", second.fencing_token, external, dispatched.append)
    effects.dispatch_external(second.task_id, "worker-b", second.fencing_token, external, dispatched.append)
    assert len(dispatched) == 1
    with database.session_factory() as session:
        assert session.get(ContentTaskRow, first.task_id).result == {"video_id": "video-1"}
    database.engine.dispose()


def test_retention_executes_durable_tombstone_delete_failure_retry_and_legal_hold() -> None:
    repository = InMemoryTombstoneRepository()
    service = RetentionService(_policy(), repository)
    failing = _RecordingDeleter(fail=True)
    failed = service.execute(_candidate(), failing, now=NOW, dry_run=False)
    assert failed.action == "RETENTION_DELETE_FAILED"
    execution = repository.get_execution(failed.tombstone.tombstone_id)
    assert execution and execution.state is RetentionExecutionState.DELETE_FAILED
    assert "private-locator" not in str(execution.last_error_code)

    successful = _RecordingDeleter()
    retried = service.execute(_candidate(), successful, now=NOW, dry_run=False)
    again = service.execute(_candidate(), successful, now=NOW, dry_run=False)
    assert retried.action == again.action == "RETENTION_DELETED"
    assert len(successful.calls) == 1
    assert repository.get_execution(retried.tombstone.tombstone_id).state is RetentionExecutionState.DELETED
    held = service.execute(_candidate(legal_hold=True), successful, now=NOW, dry_run=False)
    assert held.action == "LEGAL_HOLD" and len(successful.calls) == 1


def test_retention_sql_state_survives_service_restart_and_safe_filesystem_rejects_traversal(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'retention.db'}")
    database.create_schema()
    candidate = _candidate()
    root = tmp_path / "private"
    root.mkdir()
    target = root / "artifact.bin"
    target.write_bytes(b"fixture only")
    service = RetentionService(_policy(), SqlRetentionExecutionRepository(database.session_factory))
    deleted = service.execute(
        candidate,
        SafeFilesystemArtifactDeleter(root, {candidate.artifact_id: "artifact.bin"}),
        now=NOW,
        dry_run=False,
    )
    assert deleted.action == "RETENTION_DELETED" and not target.exists()
    restarted = RetentionService(_policy(), SqlRetentionExecutionRepository(database.session_factory))
    assert restarted.plan(candidate, now=NOW, dry_run=False).tombstone == deleted.tombstone
    assert (
        restarted.execute(
            candidate,
            SafeFilesystemArtifactDeleter(root, {candidate.artifact_id: "artifact.bin"}),
            now=NOW,
            dry_run=False,
        ).action
        == "RETENTION_DELETED"
    )
    unsafe = restarted.execute(
        RetentionCandidate(
            "artifact-2", RetentionClass.RAW_MEDIA, "c" * 64, "d" * 64, "lineage-2", NOW - timedelta(days=8)
        ),
        SafeFilesystemArtifactDeleter(root, {"artifact-2": "../outside.bin"}),
        now=NOW,
        dry_run=False,
    )
    assert unsafe.action == "RETENTION_DELETE_FAILED"
    assert not (tmp_path / "outside.bin").exists()
    database.engine.dispose()


def test_retention_scheduler_uses_only_db_private_locator_mapping_and_honors_hold(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'retention-scheduler.db'}")
    database.create_schema()
    root = tmp_path / "private-root"
    root.mkdir()
    target = root / "artifact.bin"
    target.write_bytes(b"fixture-only")
    with database.session_factory.begin() as session:
        session.add(
            ContentArtifactRow(
                artifact_id="artifact-scheduled",
                artifact_type="media",
                schema_version="artifact.v1",
                producer_stage="test",
                producer_version="1",
                content_hash="d" * 64,
                parent_artifact_ids=[],
                payload={},
                created_at=NOW - timedelta(days=8),
            )
        )
        session.add(
                SourceArtifactMetadataRow(
                artifact_id="artifact-scheduled",
                source_policy_version="source-policy.v1",
                retention_class="raw_media",
                access_classification="PRIVATE",
                    source_content_hash="e" * 64,
                    source_identity_hash="f" * 64,
                content_size=1,
                mime_type="application/octet-stream",
            )
        )
        session.add(
            RetentionArtifactLocatorRow(
                artifact_id="artifact-scheduled", private_root_id="fixture-root", relative_locator="artifact.bin"
            )
        )
    scheduler = RetentionScheduler(
        RetentionService(_policy(), SqlRetentionExecutionRepository(database.session_factory)),
        SqlRetentionCandidateRepository(database.session_factory, private_root_id="fixture-root"),
        private_root=root,
    )
    assert scheduler.sweep(dry_run=False) == ("RETENTION_DELETED",)
    assert not target.exists()
    # A durable tombstone prevents a restart from producing a second delete.
    assert scheduler.sweep(dry_run=False) == ("RETENTION_DELETED",)
    with database.session_factory.begin() as session:
        session.get(RetentionArtifactLocatorRow, "artifact-scheduled").legal_hold = True
    assert scheduler.sweep(dry_run=False) == ("LEGAL_HOLD",)
    database.engine.dispose()

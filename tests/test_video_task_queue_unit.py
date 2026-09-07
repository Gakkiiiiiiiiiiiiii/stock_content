from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentTaskRow
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.application.service import _assert_checkpoint_has_no_signed_url
from stock_content.domain.models import ContentTask
from stock_content.domain.worker_capability import TaskKind, WorkerProfile, require_capability
from stock_content.ports.repositories import StaleTaskLease


def _repository(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'task-fencing.db'}")
    database.create_schema()
    return database, PostgresContentTaskRepository(database.session_factory)


def _task(task_id: str = "video-task") -> ContentTask:
    return ContentTask(task_id=task_id, source_type="bilibili", source_ref="BV1fence", task_kind="video_pipeline")


def test_video_capability_is_exclusive_and_claim_is_kind_filtered(tmp_path):
    database, repository = _repository(tmp_path)
    repository.create(_task())
    repository.create(ContentTask(
        task_id="legacy", source_type="legacy", source_ref="unknown", task_kind="legacy_unresolved",
    ))

    with pytest.raises(ValueError, match="cannot execute"):
        require_capability(WorkerProfile.CORE, TaskKind.VIDEO_PIPELINE)
    require_capability(WorkerProfile.VIDEO, TaskKind.VIDEO_PIPELINE)

    assert repository.claim_pending("core-1", TaskKind.CORE.value, 60) is None
    claimed = repository.claim_pending("video-1", TaskKind.VIDEO_PIPELINE.value, 60)
    assert claimed is not None
    assert claimed.task_id == "video-task"
    assert claimed.fencing_token == 1
    assert repository.claim_pending("video-2", TaskKind.VIDEO_PIPELINE.value, 60) is None
    assert repository.claim_pending("video-2", "legacy_unresolved", 60) is None
    database.engine.dispose()


def test_expired_lease_fences_every_mutation_and_allows_one_terminal_effect(tmp_path):
    database, repository = _repository(tmp_path)
    repository.create(_task())
    first = repository.claim_pending("video-1", TaskKind.VIDEO_PIPELINE.value, 60)
    assert first is not None

    with database.session_factory.begin() as session:
        row = session.get(ContentTaskRow, first.task_id)
        row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)

    second = repository.claim_pending("video-2", TaskKind.VIDEO_PIPELINE.value, 60)
    assert second is not None and second.fencing_token == first.fencing_token + 1

    for write in (
        lambda: repository.renew_lease(first.task_id, "video-1", first.fencing_token, 60),
        lambda: repository.update_progress(first.task_id, "asr", 50, "video-1", first.fencing_token),
        lambda: repository.checkpoint(first.task_id, "asr", {"artifact_hash": "safe"}, "video-1", first.fencing_token),
        lambda: repository.fail(first.task_id, "asr", "stale", "video-1", first.fencing_token),
        lambda: repository.commit_effect(
            first.task_id, {"content_snapshot_id": "stale"}, "video-1", first.fencing_token,
        ),
    ):
        with pytest.raises(StaleTaskLease):
            write()

    repository.checkpoint(second.task_id, "asr", {"artifact_hash": "fresh"}, "video-2", second.fencing_token, 50)
    repository.commit_effect(second.task_id, {"content_snapshot_id": "snapshot-1"}, "video-2", second.fencing_token)
    with pytest.raises(StaleTaskLease):
        repository.commit_effect(second.task_id, {"content_snapshot_id": "snapshot-2"}, "video-2", second.fencing_token)

    with database.session_factory() as session:
        row = session.scalar(select(ContentTaskRow).where(ContentTaskRow.task_id == first.task_id))
        assert row.status == "SUCCEEDED"
        assert row.result == {"content_snapshot_id": "snapshot-1"}
        assert row.checkpoint["asr"]["artifact_hash"] == "fresh"
    database.engine.dispose()


def test_fencing_migration_and_video_image_are_explicit_without_media_dependencies():
    root = Path(__file__).parents[1]
    migration = (root / "migrations" / "031_content_task_fencing.sql").read_text(encoding="utf-8")
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    image = (root / "docker" / "Dockerfile.video").read_text(encoding="utf-8")

    assert "fencing_token" in migration
    assert "stock-content-video-worker" in compose
    assert "CONTENT_WORKER_QUEUE: video_pipeline" in compose
    assert "faster-whisper" not in image and "yt-dlp" not in image


def test_checkpoint_rejects_ephemeral_signed_urls():
    with pytest.raises(ValueError, match="EPHEMERAL_SIGNED_URL"):
        _assert_checkpoint_has_no_signed_url({"media": "https://cdn.example/video?signature=temporary"})

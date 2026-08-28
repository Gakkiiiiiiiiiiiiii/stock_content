from pathlib import Path

import pytest
from sqlalchemy import inspect

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ClaimOccurrenceRow, SemanticSegmentRow
from stock_content.adapters.postgres.repositories.semantic_segment_repository import SemanticSegmentRepository
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import SemanticSegmentationStage
from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.models import VideoAsset
from stock_content.domain.semantic_segment import (
    build_semantic_segment_artifact,
    materialize_semantic_segments,
    semantic_segment_id,
)


def _transcript() -> TranscriptArtifact:
    return TranscriptArtifact(
        artifact_id="transcript-contract",
        artifact_type="transcript",
        media_artifact_id="media-contract",
        asr_model="fixture",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(
                segment_index=index,
                start_seconds=float(index),
                end_seconds=float(index + 1),
                text=f"segment {index}",
                raw_text=f"segment {index}",
                media_artifact_id="media-contract",
                asr_model="fixture",
                asr_model_version="1",
            )
            for index in range(2)
        ],
    )


def test_semantic_segment_id_is_stable_content_identity_within_schema_limit():
    base = semantic_segment_id("transcript", "start", "end")
    assert base == semantic_segment_id("transcript", "start", "end")
    assert len(base) == 64
    assert base.startswith("semseg_")
    assert len({
        base,
        semantic_segment_id("transcript-2", "start", "end"),
        semantic_segment_id("transcript", "start-2", "end"),
        semantic_segment_id("transcript", "start", "end-2"),
        semantic_segment_id("transcript", "start", "end", "semantic-segment.v2"),
    }) == 5


def test_semantic_domain_and_repository_carry_authoritative_video_id(tmp_path):
    transcript = _transcript()
    segments = materialize_semantic_segments(transcript, ())
    assert "video_id" not in segments[0].__dataclass_fields__
    artifact = build_semantic_segment_artifact(transcript, ())
    assert "video_id" not in artifact.__dataclass_fields__
    assert "video_id" not in artifact.segments[0].__dataclass_fields__

    database = Database(f"sqlite:///{tmp_path / 'semantic-contract.db'}")
    database.create_schema()
    repository = SemanticSegmentRepository(database.session_factory)
    repository.save(artifact, video_id="video-contract")
    stored = repository.get(segments[0].semantic_segment_id)
    assert stored is not None
    assert "video_id" not in stored.__dataclass_fields__
    with database.session_factory() as session:
        row = session.get(SemanticSegmentRow, segments[0].semantic_segment_id)
        assert row is not None and row.video_id == "video-contract"

    no_video_artifact = build_semantic_segment_artifact(transcript, ())
    with pytest.raises(ValueError, match="authoritative video_id"):
        repository.save(no_video_artifact)


class _CapturingSemanticRepository:
    def __init__(self):
        self.video_ids = []

    def save(self, artifact, *, video_id=None):
        self.video_ids.append(video_id)


def test_semantic_stage_passes_current_video_identity_and_fails_closed_without_it():
    repository = _CapturingSemanticRepository()
    context = PipelineContext(
        task_id="semantic-video-contract",
        source={"type": "fixture", "ref": "semantic-video-contract"},
        options={"offline_fixture": True},
    )
    context.artifacts.transcript = _transcript()
    context.state.video = VideoAsset(
        video_id="authoritative-video",
        source_type="fixture",
        source_ref="semantic-video-contract",
        title="fixture",
    )
    SemanticSegmentationStage(repository=repository).execute(context)
    assert repository.video_ids == ["authoritative-video"]

    missing_video = PipelineContext(
        task_id="semantic-video-missing",
        source={"type": "fixture", "ref": "semantic-video-missing"},
        options={"offline_fixture": True},
    )
    missing_video.artifacts.transcript = _transcript()
    with pytest.raises(ValueError, match="authoritative current context.state.video.video_id"):
        SemanticSegmentationStage(repository=repository).execute(missing_video)


def test_semantic_orm_and_migrations_match_final_contract():
    database = Database("sqlite://")
    database.create_schema()
    columns = {item["name"]: item for item in inspect(database.engine).get_columns("semantic_segment")}
    assert columns["semantic_segment_id"]["type"].length == 64
    assert columns["video_id"]["type"].length == 64
    assert columns["video_id"]["nullable"] is False
    assert columns["subject"]["type"].length == 255
    assert columns["model_id"]["type"].length == 160
    assert columns["prompt_version"]["type"].length == 80

    occurrence_columns = {
        item["name"]: item for item in inspect(database.engine).get_columns("claim_occurrence")
    }
    assert occurrence_columns["semantic_segment_id"]["type"].length == 64

    model_checks = {constraint.sqltext.text for constraint in SemanticSegmentRow.__table__.constraints
                    if constraint.__class__.__name__ == "CheckConstraint"}
    occurrence_checks = {constraint.sqltext.text for constraint in ClaimOccurrenceRow.__table__.constraints
                         if constraint.__class__.__name__ == "CheckConstraint"}
    assert "start_segment_index <= end_segment_index" in model_checks
    assert "start_ms <= end_ms" in model_checks
    assert "available_from >= ingested_at" in occurrence_checks
    assert "available_from >= extraction_completed_at" in occurrence_checks
    assert "available_from >= snapshot_committed_at" in occurrence_checks

    migration_root = Path(__file__).parents[1] / "migrations"
    semantic_sql = (migration_root / "017_semantic_segments.sql").read_text(encoding="utf-8")
    occurrence_sql = (migration_root / "019_claim_occurrence_final.sql").read_text(encoding="utf-8")
    assert "semantic_segment_id varchar(64) PRIMARY KEY" in semantic_sql
    assert "video_id varchar(64) NOT NULL" in semantic_sql
    assert "subject varchar(255)" in semantic_sql
    assert "model_id varchar(160)" in semantic_sql
    assert "prompt_version varchar(80)" in semantic_sql
    assert "semantic_segment_id varchar(64) NOT NULL" in occurrence_sql
    assert "CHECK (available_from >= ingested_at)" in occurrence_sql

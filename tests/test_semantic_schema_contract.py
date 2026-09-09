from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ClaimOccurrenceRow, SemanticSegmentRow
from stock_content.adapters.postgres.repositories.semantic_segment_repository import SemanticSegmentRepository
from stock_content.application.pipeline import PipelineContext
from stock_content.application.replay.identity import migration_derivation_namespace
from stock_content.application.stages import SemanticSegmentationStage
from stock_content.domain.artifacts import (
    EvidenceArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    artifact_id_of,
    canonical_json,
    content_hash_of,
    deserialize_artifact,
    serialize_artifact,
)
from stock_content.domain.claim_occurrence import ClaimOccurrence, knowledge_uid_for_occurrence
from stock_content.domain.models import VideoAsset
from stock_content.domain.semantic_segment import (
    build_semantic_segment_artifact,
    materialize_semantic_segments,
    semantic_segment_id,
)
from stock_content.domain.temporal_semantics import OccurrenceTimes


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


def test_migration_namespace_is_stable_and_isolates_immutable_derived_rows(tmp_path):
    transcript = _transcript()
    v4_namespace = migration_derivation_namespace("cs-parent", "pipeline.v4")
    assert v4_namespace == migration_derivation_namespace("cs-parent", "pipeline.v4")
    assert v4_namespace != migration_derivation_namespace("cs-parent", "pipeline.v5")

    legacy = build_semantic_segment_artifact(transcript, (), model_id="legacy")
    unnamespaced_v4 = build_semantic_segment_artifact(transcript, (), model_id="pipeline.v4")
    v4 = build_semantic_segment_artifact(transcript, (), model_id="pipeline.v4", identity_seed=v4_namespace)
    v4_repeat = build_semantic_segment_artifact(
        transcript, (), model_id="pipeline.v4", identity_seed=v4_namespace
    )
    v5 = build_semantic_segment_artifact(
        transcript,
        (),
        model_id="pipeline.v5",
        identity_seed=migration_derivation_namespace("cs-parent", "pipeline.v5"),
    )
    legacy, unnamespaced_v4, v4, v4_repeat, v5 = (
        replace(item, artifact_id=artifact_id_of(item))
        for item in (legacy, unnamespaced_v4, v4, v4_repeat, v5)
    )
    assert legacy.segments[0].semantic_segment_id == unnamespaced_v4.segments[0].semantic_segment_id
    assert v4.segments[0].semantic_segment_id == v4_repeat.segments[0].semantic_segment_id
    assert len({legacy.segments[0].semantic_segment_id, v4.segments[0].semantic_segment_id,
                v5.segments[0].semantic_segment_id}) == 3

    database = Database(f"sqlite:///{tmp_path / 'migration-namespace.db'}")
    database.create_schema()
    repository = SemanticSegmentRepository(database.session_factory)
    repository.save(legacy, video_id="video-contract")
    with pytest.raises(ValueError, match="already stores different payload"):
        repository.save(unnamespaced_v4, video_id="video-contract")
    repository.save(v4, video_id="video-contract")
    repository.save(v4_repeat, video_id="video-contract")
    repository.save(v5, video_id="video-contract")

    old_evidence = EvidenceArtifact(
        artifact_id="evidence-pending", artifact_type="evidence", parent_artifact_ids=(legacy.artifact_id,)
    )
    migrated_evidence = EvidenceArtifact(
        artifact_id="evidence-pending", artifact_type="evidence", parent_artifact_ids=(v4.artifact_id,)
    )
    assert artifact_id_of(old_evidence) != artifact_id_of(migrated_evidence)
    clock = datetime(2025, 1, 1, tzinfo=UTC)
    times = OccurrenceTimes(
        ingested_at=clock, extraction_completed_at=clock, snapshot_committed_at=clock, available_from=clock
    )
    old_occurrence = ClaimOccurrence(
        claim_id="claim-shared", source_artifact_id="source", transcript_artifact_id=transcript.artifact_id,
        semantic_segment_id=legacy.segments[0].semantic_segment_id, evidence_refs=["evidence-shared"], times=times,
    )
    migrated_occurrence = ClaimOccurrence(
        claim_id="claim-shared", source_artifact_id="source", transcript_artifact_id=transcript.artifact_id,
        semantic_segment_id=v4.segments[0].semantic_segment_id, evidence_refs=["evidence-shared"], times=times,
    )
    assert old_occurrence.occurrence_id != migrated_occurrence.occurrence_id
    assert knowledge_uid_for_occurrence(old_occurrence.occurrence_id) != knowledge_uid_for_occurrence(
        migrated_occurrence.occurrence_id
    )


def test_semantic_artifact_empty_namespace_retains_legacy_canonical_bytes(tmp_path):
    """A pre-namespace row must survive strict integrity verification unchanged."""
    legacy_identity = {
        "artifact_type": "semantic_segments",
        "schema_version": "artifact.v1",
        "producer_stage": "semantic_segmentation",
        "producer_version": "4.043",
        "parent_artifact_ids": ["transcript-legacy"],
        "transcript_artifact_id": "transcript-legacy",
        "segments": [
            {
                "semantic_segment_id": "semseg_legacy",
                "segment_index": 0,
                "start_segment_index": 0,
                "end_segment_index": 0,
                "start_segment_id": "seg-0",
                "end_segment_id": "seg-0",
                "start_ms": 0,
                "end_ms": 1_000,
                "topic": None,
                "subject": None,
                "segment_type": "ANALYSIS",
                "confidence": None,
            }
        ],
        "model_id": "semantic.fixture",
        "prompt_version": "v1",
        "segmentation_schema_version": "semantic-segment.v1",
    }
    legacy_payload = {
        "artifact_id": "semantic_segments-350flegacy",
        "created_at": datetime(2025, 1, 1, tzinfo=UTC),
        "content_hash": content_hash_of(legacy_identity),
        **legacy_identity,
    }

    restored = deserialize_artifact(legacy_payload)
    assert restored.segments[0].derivation_namespace == ""
    assert canonical_json(serialize_artifact(restored)) == canonical_json(legacy_payload)
    assert restored.content_hash == legacy_payload["content_hash"]

    database = Database(f"sqlite:///{tmp_path / 'legacy-semantic-artifact.db'}")
    database.create_schema()
    from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository

    repository = SqlArtifactRepository(database.session_factory)
    repository.put(restored)
    assert repository.get(restored.artifact_id).content_hash == legacy_payload["content_hash"]


def test_nonempty_semantic_namespace_participates_in_artifact_identity():
    transcript = _transcript()
    legacy = build_semantic_segment_artifact(transcript, (), model_id="semantic.fixture")
    reprocess = build_semantic_segment_artifact(
        transcript,
        (),
        model_id="semantic.fixture",
        identity_seed="",
    )
    migration = build_semantic_segment_artifact(
        transcript,
        (),
        model_id="semantic.fixture",
        identity_seed="migration-namespace",
    )

    assert "derivation_namespace" not in serialize_artifact(legacy)["segments"][0]
    assert serialize_artifact(reprocess) == serialize_artifact(legacy)
    assert reprocess.content_hash == legacy.content_hash
    assert serialize_artifact(migration)["segments"][0]["derivation_namespace"] == "migration-namespace"
    assert artifact_id_of(legacy) != artifact_id_of(migration)
    assert legacy.content_hash != migration.content_hash


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

    migration = PipelineContext(
        task_id="semantic-migration",
        source={"type": "fixture", "ref": "semantic-migration"},
        options={"offline_fixture": True, "replay_derived_identity_seed": "migration-test"},
    )
    migration.artifacts.transcript = _transcript()
    migration.state.video = VideoAsset(
        video_id="authoritative-video", source_type="fixture", source_ref="semantic-migration", title="fixture"
    )
    SemanticSegmentationStage(repository=repository).execute(migration)
    assert (
        migration.state.semantic_segments[0].semantic_segment_id
        != context.state.semantic_segments[0].semantic_segment_id
    )

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
    assert columns["derivation_namespace"]["type"].length == 48
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
    migration_sql = (migration_root / "039_migration_replay_derivation_namespace.sql").read_text(encoding="utf-8")
    assert "derivation_namespace varchar(48) NOT NULL DEFAULT ''" in migration_sql
    assert "transcript_artifact_id, derivation_namespace, segment_index" in migration_sql

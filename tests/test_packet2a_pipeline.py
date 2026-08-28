from dataclasses import replace
from datetime import UTC, date, datetime, timezone

import pytest
from sqlalchemy import func, select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ClaimOccurrenceRow,
    ContentSnapshotRow,
    LifecycleEventLedgerRow,
)
from stock_content.adapters.postgres.repositories import SqlArtifactRepository, SqlSnapshotStore
from stock_content.adapters.postgres.repositories.claim_occurrence_repository import ClaimOccurrenceRepository
from stock_content.adapters.postgres.repositories.lifecycle_repository import LifecycleRepository
from stock_content.application.pipeline import CORE_METRIC_KEYS, PipelineContext
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.stages import (
    AtomicClaimExtractionStage,
    BuildVideoStage,
    ClaimCanonicalizationStage,
    ClaimOccurrencePersistenceStage,
    EvidenceGroundingStage,
    LifecycleProjectionStage,
    SemanticContextStage,
    SemanticSegmentationStage,
    TemporalNormalizationStage,
)
from stock_content.domain.artifacts import (
    ClaimOccurrenceArtifact,
    KnowledgeArtifact,
    SourceArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    VerificationArtifact,
    artifact_id_of,
)
from stock_content.domain.atomic_claim_extractor import AtomicClaimExtractor
from stock_content.domain.claim_draft import ClaimOccurrenceDraft, TemporalExpressionDraft
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.knowledge_projection_builder import KnowledgeProjectionBuilder
from stock_content.domain.lifecycle_event import KnowledgeLifecycleEvent, LifecycleTargetType
from stock_content.domain.models import VideoAsset
from stock_content.domain.semantic_segmenter import SemanticSegmenter
from stock_content.domain.temporal_semantics import OccurrenceTimes


def _transcript(count=3):
    return TranscriptArtifact(
        artifact_id="transcript-fixture",
        artifact_type="transcript",
        media_artifact_id="media-fixture",
        asr_model="fixture",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(
                segment_index=i,
                start_seconds=i,
                end_seconds=i + 1,
                text=f"segment {i}",
                raw_text=f"segment {i}",
                media_artifact_id="media-fixture",
                asr_model="fixture",
                asr_model_version="1",
            )
            for i in range(count)
        ],
    )


class _Gateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def available(self):
        return True

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_semantic_segmenter_short_call_and_repair_fail_closed():
    gateway = _Gateway([{"content": '{"unexpected": []}'}, {"content": '{"boundaries": []}'}])
    result = SemanticSegmenter(gateway).segment(_transcript())
    assert len(gateway.calls) == 2
    assert len(result.segments) == 1
    bad = _Gateway([{"content": '{"unexpected": []}'}, {"content": '{"unexpected": []}'}])
    with pytest.raises(ValueError, match="after one repair"):
        SemanticSegmenter(bad).segment(_transcript())


def test_build_video_preserves_authoritative_resolved_metadata_and_times():
    context = PipelineContext(
        task_id="resolved-metadata",
        source={"type": "fixture", "ref": "one"},
        options={
            "metadata": {
                "title": "resolved title",
                "author": "source author",
                "canonical_url": "https://example.test/video/one",
                # Naive timestamps retain the existing UTC interpretation.
                "published_at": "2025-07-01T12:00:00",
                "source_version": "resolver-v7",
                "resolved_at": "2025-07-01T12:01:00+08:00",
                "provider_field": "preserved",
            }
        },
    )
    context.state.metadata = dict(context.options["metadata"])
    context.state.transcript = "source text"

    BuildVideoStage().execute(context)

    video = context.state.video
    assert video.canonical_url == "https://example.test/video/one"
    assert video.published_at == datetime(2025, 7, 1, 12, tzinfo=UTC)
    assert video.source_version == "resolver-v7"
    assert video.resolved_at == datetime(2025, 7, 1, 4, 1, tzinfo=UTC)
    assert video.metadata["provider_field"] == "preserved"

    context.state.metadata["published_at"] = "not-a-timestamp"
    context.state.video = None
    with pytest.raises(ValueError, match="published_at"):
        BuildVideoStage().execute(context)


def test_projection_is_occurrence_scoped_and_keeps_occurrence_time_lineage():
    now = datetime(2025, 7, 2, tzinfo=UTC)
    claim = FinancialClaim(
        claim_type="FINANCIAL_METRIC",
        subject_type="EQUITY",
        subject_id="600519.SH",
        ticker="600519",
        predicate="revenue",
        value=10,
        evidence_refs=["e1"],
        source_support_status="SUPPORTED",
        source_confidence=0.9,
        extractor_confidence=0.9,
    )

    def occurrence(semantic_segment_id: str, evidence_ref: str) -> ClaimOccurrence:
        return ClaimOccurrence(
            claim_id=claim.claim_id,
            source_artifact_id="source-1",
            transcript_artifact_id="transcript-1",
            semantic_segment_id=semantic_segment_id,
            evidence_refs=[evidence_ref],
            times=OccurrenceTimes(
                asserted_at=now,
                source_published_at=now,
                ingested_at=now,
                extraction_completed_at=now,
                snapshot_committed_at=now,
                available_from=now,
            ),
        )

    first = occurrence("semantic-1", "e1")
    second = occurrence("semantic-2", "e2")
    builder = KnowledgeProjectionBuilder()
    first_projection = builder.build(claim, first)
    second_projection = builder.build(claim, second)
    repeated_projection = builder.build(claim, first)

    assert first_projection["knowledge_uid"] == first.occurrence_id
    assert second_projection["knowledge_uid"] == second.occurrence_id
    assert first_projection["knowledge_uid"] != second_projection["knowledge_uid"]
    assert repeated_projection["knowledge_uid"] == first_projection["knowledge_uid"]
    assert first_projection["attributes"]["claim_id"] == claim.claim_id
    assert first_projection["attributes"]["semantic_segment_id"] == "semantic-1"
    assert first_projection["attributes"]["asserted_at"] == now.isoformat()
    assert first_projection["attributes"]["source_published_at"] == now.isoformat()


def test_atomic_extractor_accepts_zero_claims_and_prompt_is_atomic():
    gateway = _Gateway([{"content": '{"claims": []}'}])
    extractor = AtomicClaimExtractor(gateway)
    drafts = extractor.extract({"semantic_segment_id": "seg", "transcript_text": "nothing"})
    assert drafts == []
    assert "atomic" in gateway.calls[0]["prompt"].lower()
    assert "another temporal model" in gateway.calls[0]["prompt"]


def test_packet2a_stages_form_empty_artifact_chain(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'packet2a.db'}")
    database.create_schema()
    occurrence_repo = ClaimOccurrenceRepository(database.session_factory)
    lifecycle_repo = LifecycleRepository(database.session_factory)
    context = PipelineContext(
        task_id="packet2a",
        source={"type": "fixture", "ref": "one"},
        options={"as_of": "2026-01-01T00:00:00Z"},
    )
    context.artifacts.transcript = _transcript()
    SemanticSegmentationStage().execute(context)
    assert set(CORE_METRIC_KEYS) <= set(context.runtime.metrics)
    assert context.runtime.metrics["semantic_segments_per_video"] == 1.0
    SemanticContextStage().execute(context)
    AtomicClaimExtractionStage().execute(context)
    EvidenceGroundingStage().execute(context)
    TemporalNormalizationStage().execute(context)
    ClaimCanonicalizationStage().execute(context)
    ClaimOccurrencePersistenceStage(occurrence_repo).execute(context)
    context.artifacts.verification = VerificationArtifact(
        artifact_id="verification-fixture",
        artifact_type="verification",
        claim_artifact_id="",
        results=[],
    )
    LifecycleProjectionStage(lifecycle_repo).execute(context)
    assert context.artifacts.semantic_segments is not None
    assert context.artifacts.occurrences is not None
    assert context.artifacts.lifecycle is not None
    assert context.artifacts.occurrences.parent_artifact_ids
    assert context.artifacts.lifecycle.parent_artifact_ids == (
        context.artifacts.occurrences.artifact_id,
        context.artifacts.verification.artifact_id,
    )


def test_packet2a_materializes_final_artifact_chain_and_raw_temporal_provenance():
    context = PipelineContext(
        task_id="packet2a-chain",
        source={"type": "fixture", "ref": "one"},
        options={"offline_fixture": True, "as_of": "2026-01-01T00:00:00Z"},
    )
    context.state.metadata = {"published_at": "2025-07-01T12:00:00+08:00"}
    original = _transcript(1)
    segment = replace(original.segments[0], segment_id="", text="2026Q2 segment 0", raw_text="2026Q2 segment 0")
    context.artifacts.transcript = replace(original, segments=[segment], content_hash="")
    SemanticSegmentationStage().execute(context)
    SemanticContextStage().execute(context)
    segment = context.state.semantic_segments[0]
    context.state.claim_drafts = [
        ClaimOccurrenceDraft(
            semantic_segment_id=segment.semantic_segment_id,
            knowledge_kind="OPINION",
            claim_type="FINANCIAL_METRIC",
            subject_key="600519.SH",
            predicate_key="revenue",
            conclusion="2026Q2 segment 0",
            value=10,
            evidence_segment_indices=[0],
            temporal_expressions=[
                TemporalExpressionDraft(
                    role="REPORTING_PERIOD",
                    raw_expression="2026Q2",
                    evidence_segment_indices=[0],
                    confidence=0.9,
                )
            ],
            extraction_confidence=0.9,
        )
    ]
    EvidenceGroundingStage().execute(context)
    TemporalNormalizationStage(normalization_version="temporal-normalization.final.7").execute(context)
    ClaimCanonicalizationStage().execute(context)
    ClaimOccurrencePersistenceStage().execute(context)
    assert context.artifacts.claims.parent_artifact_ids == (context.artifacts.occurrences.artifact_id,)
    occurrence = context.state.occurrences[0]
    assert occurrence.times.source_published_at == datetime(2025, 7, 1, 4, tzinfo=UTC)
    built_occurrences = context.artifacts.occurrences
    reordered_occurrences = ClaimOccurrenceArtifact(
        artifact_id="occurrences-reordered",
        artifact_type="occurrences",
        schema_version=built_occurrences.schema_version,
        producer_stage=built_occurrences.producer_stage,
        producer_version=built_occurrences.producer_version,
        semantic_segment_artifact_id=built_occurrences.semantic_segment_artifact_id,
        evidence_artifact_id=built_occurrences.evidence_artifact_id,
        occurrence_ids=[
            *reversed(built_occurrences.occurrence_ids),
            *built_occurrences.occurrence_ids,
        ],
        parent_artifact_ids=built_occurrences.parent_artifact_ids,
    )
    assert artifact_id_of(reordered_occurrences) == artifact_id_of(built_occurrences)
    context.state.video = VideoAsset(
        video_id="fixture",
        source_type="fixture",
        source_ref="one",
        title="fixture",
        published_at=datetime(2025, 7, 3, tzinfo=UTC),
    )
    context.options["source_published_at"] = "2025-07-04T00:00:00+00:00"
    ClaimOccurrencePersistenceStage().execute(context)
    assert context.state.occurrences[0].times.source_published_at == datetime(2025, 7, 4, tzinfo=UTC)
    context.options.pop("source_published_at")
    ClaimOccurrencePersistenceStage().execute(context)
    assert context.state.occurrences[0].times.source_published_at == datetime(2025, 7, 3, tzinfo=UTC)
    assert occurrence.raw_temporal_expressions[0]["raw_expression"] == "2026Q2"
    assert occurrence.raw_temporal_expressions[0]["grounded_evidence_refs"]
    context.artifacts.verification = VerificationArtifact(
        artifact_id="verification-chain",
        artifact_type="verification",
        claim_artifact_id=context.artifacts.claims.artifact_id,
        results=[],
    )
    from stock_content.domain.models import KnowledgeUnit

    context.state.knowledge = [
        KnowledgeUnit(
            knowledge_uid=context.state.claims[0].claim_id,
            video_id="fixture",
            chapter_id=None,
            statement="营收增长 10%",
        )
    ]
    context.artifacts.knowledge = KnowledgeArtifact(
        artifact_id="knowledge-initial",
        artifact_type="knowledge",
        verification_artifact_id=context.artifacts.verification.artifact_id,
        knowledge_units=[context.state.claims[0].claim_id],
        parent_artifact_ids=(context.artifacts.verification.artifact_id,),
    )
    LifecycleProjectionStage().execute(context)
    assert context.artifacts.knowledge.parent_artifact_ids == (context.artifacts.lifecycle.artifact_id,)
    assert context.state.knowledge[0].attributes["lifecycle_artifact_id"] == context.artifacts.lifecycle.artifact_id


def test_temporal_stage_uses_publish_anchor_and_claim_semantics():
    context = PipelineContext(
        task_id="temporal-stage",
        source={"type": "fixture", "ref": "one"},
        options={"offline_fixture": True, "as_of": "2026-01-01T00:00:00Z"},
    )
    original = _transcript(1)
    segment_item = replace(
        original.segments[0], segment_id="", text="计划2026Q2 segment 0", raw_text="计划2026Q2 segment 0"
    )
    context.artifacts.transcript = replace(original, segments=[segment_item], content_hash="")
    SemanticSegmentationStage().execute(context)
    SemanticContextStage().execute(context)
    segment = context.state.semantic_segments[0]
    context.state.claim_drafts = [
        ClaimOccurrenceDraft(
            semantic_segment_id=segment.semantic_segment_id,
            knowledge_kind="FACT",
            claim_type="FINANCIAL_METRIC",
            subject_key="600519.SH",
            predicate_key="revenue",
            conclusion="计划2026Q2 segment 0",
            evidence_segment_indices=[0],
            temporal_expressions=[
                TemporalExpressionDraft(
                    role="REPORTING_PERIOD",
                    raw_expression="2026Q2",
                    anchor="SOURCE_PUBLISH_TIME",
                    evidence_segment_indices=[0],
                    confidence=0.9,
                )
            ],
            extraction_confidence=0.9,
        )
    ]
    context.state.video = VideoAsset(
        video_id="fixture",
        source_type="fixture",
        source_ref="one",
        title="fixture",
        published_at=datetime(2025, 7, 1, tzinfo=timezone.utc),
    )
    EvidenceGroundingStage().execute(context)
    TemporalNormalizationStage().execute(context)
    binding = context.state.temporal_bindings[0]
    assert binding.assertion_status.value == "PLANNED"
    assert binding.metric_temporal_nature.value == "DURATION"
    assert binding.start_date == date(2026, 4, 1)


@pytest.mark.parametrize("failure_point", ["snapshot", "occurrence", "lifecycle"])
def test_snapshot_bundle_rolls_back_all_ledgers_on_any_failure(tmp_path, monkeypatch, failure_point):
    """Snapshot publication is one SQL unit, including both dependent ledgers."""
    database = Database(f"sqlite:///{tmp_path / f'bundle-{failure_point}.db'}")
    database.create_schema()
    SqlArtifactRepository(database.session_factory).put(
        SourceArtifact(
            artifact_id="source-atomic",
            artifact_type="source",
            source_type="fixture",
            source_ref="atomic",
            source_content_hash="content-atomic",
        )
    )
    now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    occurrence = ClaimOccurrence(
        claim_id="claim-atomic",
        source_artifact_id="source-atomic",
        transcript_artifact_id="transcript-atomic",
        semantic_segment_id="segment-atomic",
        times=OccurrenceTimes(
            ingested_at=now,
            extraction_completed_at=now,
            snapshot_committed_at=now,
            available_from=now,
        ),
    )
    event = KnowledgeLifecycleEvent(
        target_type=LifecycleTargetType.OCCURRENCE,
        target_id=occurrence.occurrence_id,
        from_status=None,
        to_status="ACTIVE",
        effective_at=now,
        recorded_at=now,
        reason_code="INITIAL_EXTRACTION",
        policy_version="lifecycle.v1",
    )

    if failure_point == "snapshot":
        def fail_snapshot(*args, **kwargs):
            raise RuntimeError("snapshot write failed")

        monkeypatch.setattr(SqlSnapshotStore, "_save_in_session", fail_snapshot)
    elif failure_point == "occurrence":
        def fail_occurrence(*args, **kwargs):
            raise RuntimeError("occurrence write failed")

        monkeypatch.setattr(ClaimOccurrenceRepository, "save_in_session", fail_occurrence)
    else:
        def fail_lifecycle(*args, **kwargs):
            raise RuntimeError("lifecycle write failed")

        monkeypatch.setattr(LifecycleRepository, "append_in_session", fail_lifecycle)

    service = SnapshotService(SqlSnapshotStore(database.session_factory))
    with pytest.raises(RuntimeError, match="write failed"):
        service.record_bundle_from_artifacts(
            source_type="fixture",
            source_ref="atomic",
            source_content_hash="content-atomic",
            artifact_ids={"source": "source-atomic"},
            source_artifact_id="source-atomic",
            code_sha="sha-atomic",
            config_hash="cfg-atomic",
            occurrences=(occurrence,),
            lifecycle_events=(event,),
        )

    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ContentSnapshotRow)) == 0
        assert session.scalar(select(func.count()).select_from(ClaimOccurrenceRow)) == 0
        assert session.scalar(select(func.count()).select_from(LifecycleEventLedgerRow)) == 0

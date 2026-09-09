from __future__ import annotations

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.claim_occurrence_repository import ClaimOccurrenceRepository
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import (
    ClaimCanonicalizationStage,
    ClaimOccurrencePersistenceStage,
    EvidenceGroundingStage,
    SemanticSegmentationStage,
)
from stock_content.domain.artifacts import (
    FrameArtifact,
    OCRArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    TranscriptVisualCrosscheckArtifact,
    TranscriptVisualCrosscheckRecord,
    VisionArtifact,
)
from stock_content.domain.claim_draft import ClaimOccurrenceDraft, VisualEvidenceAnchor
from stock_content.domain.knowledge_semantics import bundle_v2_semantics


def _transcript() -> TranscriptArtifact:
    return TranscriptArtifact(
        artifact_id="transcript-semantic",
        artifact_type="transcript",
        media_artifact_id="media",
        asr_model="fixture",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(
                segment_index=0,
                start_seconds=10,
                end_seconds=12,
                text="信息基础设施投资预计到2030年达到三万亿元。",
                raw_text="信息基础设施投资预计到2030年达到三万亿元。",
                media_artifact_id="media",
                asr_model="fixture",
                asr_model_version="1",
            )
        ],
    )


def test_semantic_envelope_strips_presentation_framing_and_keeps_unknown_year_unknown():
    envelope = bundle_v2_semantics(
        statement="课程提出：截至8月末黄金储备增加。",
        claim_type="OPINION",
        temporal_expressions=[{"raw_expression": "8月末"}],
    )
    assert envelope["primary_domain"] == "CENTRAL_BANK_GOLD_RESERVES"
    assert envelope["claim_nature"] == "OPINION"
    assert envelope["attribution"] == {"attributed": True, "source_label": "source_speaker"}
    assert envelope["detail"]["explanation"] == "截至8月末黄金储备增加。"
    assert envelope["temporal"] == {
        "kind": "AS_OF",
        "start": None,
        "end": None,
        "as_of": None,
        "rule": None,
        "label": "8月末",
        "precision": "YEAR_UNSPECIFIED",
        "explicitly_unknown": False,
    }
    assert envelope["external_truth_status"] == "NOT_CHECKED"


def test_semantic_envelope_normalizes_a_bare_forecast_year_to_a_utc_target_interval():
    envelope = bundle_v2_semantics(
        statement="信息基础设施投资预计到2030年达到三万亿元。",
        claim_type="FORECAST",
        supplied={
            "temporal": {
                "kind": "FORECAST_TARGET",
                "start": "2030",
                "end": None,
                "as_of": None,
                "rule": None,
                "label": "2030",
                "precision": "YEAR",
                "explicitly_unknown": False,
            }
        },
    )
    assert envelope["temporal"] == {
        "kind": "FORECAST_TARGET",
        "start": "2030-01-01T00:00:00Z",
        "end": "2030-12-31T23:59:59.999999Z",
        "as_of": None,
        "rule": None,
        "label": "2030",
        "precision": "YEAR",
        "explicitly_unknown": False,
    }


def test_semantic_envelope_does_not_invent_a_year_for_an_as_of_month_end():
    envelope = bundle_v2_semantics(
        statement="截至8月末黄金储备增加。",
        claim_type="OPINION",
        supplied={
            "temporal": {
                "kind": "AS_OF",
                "start": None,
                "end": None,
                "as_of": "8月末",
                "rule": None,
                "label": None,
                "precision": "MONTH",
                "explicitly_unknown": False,
            }
        },
    )
    assert envelope["temporal"] == {
        "kind": "AS_OF",
        "start": None,
        "end": None,
        "as_of": None,
        "rule": None,
        "label": "8月末",
        "precision": "YEAR_UNSPECIFIED",
        "explicitly_unknown": False,
    }


def test_producer_persists_occurrence_owned_frame_ocr_and_vision_evidence(tmp_path):
    context = PipelineContext(
        task_id="semantic-envelope",
        source={"type": "fixture", "ref": "video"},
        options={"offline_fixture": True, "as_of": "2026-01-01T00:00:00Z"},
    )
    context.artifacts.transcript = _transcript()
    SemanticSegmentationStage().execute(context)
    semantic = context.state.semantic_segments[0]
    frame = FrameArtifact(
        artifact_id="frame-artifact",
        artifact_type="frame",
        media_artifact_id="media",
        frame_id="frame-1",
        timestamp_ms=11_000,
        image_hash="image-hash",
        storage_ref="fixture",
        semantic_segment_ids=(semantic.semantic_segment_id,),
    )
    ocr = OCRArtifact(
        artifact_id="ocr-artifact",
        artifact_type="ocr",
        frame_artifact_id=frame.artifact_id,
        frame_id="frame-1",
        timestamp_ms=11_000,
        image_hash="image-hash",
        text="2030年 三万亿元",
        bbox=[0, 0, 10, 10],
        confidence_score=0.99,
        engine="paddleocr",
        engine_version="3.7",
        requested_device="gpu:0",
        actual_device="gpu:0",
    )
    vision = VisionArtifact(
        artifact_id="vision-artifact",
        artifact_type="vision",
        frame_artifact_id=frame.artifact_id,
        frame_id="frame-1",
        timestamp_ms=11_000,
        image_hash="image-hash",
        label="政策图表",
        labels=["政策图表"],
        confidence_score=0.8,
        model_name="terra",
        model_version="test-v1",
    )
    context.artifacts.frames = [frame]
    context.artifacts.ocr = [ocr]
    context.artifacts.vision = [vision]
    context.artifacts.transcript_visual_crosscheck = TranscriptVisualCrosscheckArtifact(
        artifact_id="crosscheck",
        artifact_type="transcript_visual_crosscheck",
        transcript_artifact_id="transcript-semantic",
        semantic_segment_artifact_id=context.artifacts.semantic_segments.artifact_id,
        crosscheck_version="test",
        relations=(
            TranscriptVisualCrosscheckRecord.from_dict(
                {
                    "frame_id": "frame-1",
                    "frame_artifact_id": "frame-artifact",
                    "timestamp_ms": 11_000,
            "relation": "CONTRADICTS",
            "semantic_segment_ids": [semantic.semantic_segment_id],
            "mismatches": {"NUMBER": {"transcript": ["三万亿元"], "visual": ["三点八万亿元"]}},
                }
            ),
        ),
        eligible_frame_ids=("frame-1",),
    )
    context.state.claim_drafts = [
        ClaimOccurrenceDraft(
            semantic_segment_id=semantic.semantic_segment_id,
            knowledge_kind="POLICY",
            claim_type="FORECAST",
            subject_type="THEME",
            subject_key="INFORMATION_INFRASTRUCTURE",
            predicate_key="investment_target",
            conclusion="信息基础设施投资预计到2030年达到三万亿元",
            value="三万亿元",
            evidence_segment_indices=[0],
            extraction_confidence=0.9,
            verbatim_quote="信息基础设施投资预计到2030年达到三万亿元",
            normalized_statement="信息基础设施投资预计到2030年达到三万亿元",
            grounding_status="GROUNDED",
            claim_schema_version="claim.atomic.v1",
            legacy_grounding_incomplete=False,
            bundle_v2={"detail": {"mechanism": "投资目标对应信息基础设施建设需求。"}},
            visual_anchors=[
                VisualEvidenceAnchor(
                    frame_id="frame-1",
                    timestamp_ms=11_000,
                    bbox=(0, 0, 10, 10),
                    ocr_text="2030年 三万亿元",
                    model_id="paddleocr",
                    model_version="3.7",
                    confidence=0.99,
                ),
                VisualEvidenceAnchor(
                    frame_id="frame-1",
                    timestamp_ms=11_000,
                    bbox=(0, 0, 10, 10),
                    visual_label="政策图表",
                    model_id="terra",
                    model_version="test-v1",
                    confidence=0.8,
                    support_type="LABEL",
                ),
            ],
        )
    ]
    EvidenceGroundingStage().execute(context)
    ClaimCanonicalizationStage().execute(context)
    ClaimOccurrencePersistenceStage().execute(context)

    occurrence = context.state.occurrences[0]
    assert len(occurrence.secondary_evidence_refs) == 3
    assert occurrence.provenance["bundle_v2"]["primary_domain"] == "INFORMATION_INFRASTRUCTURE_POLICY"
    assert occurrence.provenance["bundle_v2"]["claim_nature"] == "FORECAST"
    assert occurrence.provenance["bundle_v2"]["attribution"]["attributed"] is True
    assert occurrence.provenance["bundle_v2"]["occurrence_review"] == {
        "status": "HUMAN_REVIEW_REQUIRED",
        "reason_codes": ["ASR_OCR_NUMERIC_CONFLICT"],
    }

    database = Database(f"sqlite:///{tmp_path / 'semantic-evidence.db'}")
    database.create_schema()
    repository = ClaimOccurrenceRepository(database.session_factory)
    repository.save(occurrence)
    restored = repository.get(occurrence.occurrence_id)
    assert restored is not None
    assert set(restored.secondary_evidence_refs) == set(occurrence.secondary_evidence_refs)

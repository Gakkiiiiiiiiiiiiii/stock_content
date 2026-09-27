from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

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
from stock_content.domain.claim_canonicalizer import ClaimCanonicalizer
from stock_content.domain.claim_draft import ClaimOccurrenceDraft, VisualEvidenceAnchor
from stock_content.domain.claims import claim_id_of
from stock_content.domain.knowledge_bundle_v2 import validate_v2_item
from stock_content.domain.knowledge_projection_builder import KnowledgeProjectionBuilder
from stock_content.domain.knowledge_semantics import (
    TrustedExternalVerification,
    atomic_statement,
    bundle_v2_semantics,
)


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


@pytest.mark.parametrize(
    "framed",
    [
        "讲者认为市场风险上升。", "讲者称市场风险上升。", "讲者表示市场风险上升。",
        "视频认为市场风险上升。", "视频展示：市场风险上升。", "课程认为市场风险上升。",
        "课程提出：市场风险上升。", "课程设置：市场风险上升。", "老师表示市场风险上升。",
        "主讲人指出市场风险上升。",
    ],
)
def test_atomic_statement_removes_only_source_attribution_framing(framed):
    assert atomic_statement(framed) == "市场风险上升。"
    assert atomic_statement("讲者所在公司利润增长。") == "讲者所在公司利润增长。"
    assert atomic_statement("视频展示市场风险上升。") == "市场风险上升。"
    assert atomic_statement("视频展示：讲者认为市场风险上升。") == "市场风险上升。"
    assert atomic_statement("据讲者介绍，市场风险上升。") == "市场风险上升。"
    assert atomic_statement("课程设置影响收益。") == "课程设置影响收益。"


def test_semantic_envelope_rejects_model_claimed_external_truth_without_independent_record():
    envelope = bundle_v2_semantics(
        statement="公司2025年收入增长12%。",
        claim_type="FINANCIAL_METRIC",
        supplied={"source_grade": "PRIMARY", "external_truth_status": "EXTERNALLY_VERIFIED"},
    )
    assert envelope["source_grade"] == "SOURCE_ASSERTION"
    assert envelope["external_truth_status"] == "NOT_CHECKED"
    forged = bundle_v2_semantics(
        statement="公司2025年收入增长12%。",
        claim_type="FINANCIAL_METRIC",
        supplied={
            "source_grade": "PRIMARY", "external_truth_status": "EXTERNALLY_VERIFIED",
            "external_verification": {"status": "MATCH", "provider": "official-filing", "source_id": "filing-1"},
        },
    )
    assert forged["source_grade"] == "SOURCE_ASSERTION"
    assert forged["external_truth_status"] == "NOT_CHECKED"
    verified = bundle_v2_semantics(
        statement="公司2025年收入增长12%。",
        claim_type="FINANCIAL_METRIC",
        supplied={"source_grade": "PRIMARY", "external_truth_status": "EXTERNALLY_VERIFIED"},
        trusted_verification=TrustedExternalVerification(
            verification_artifact_id="verification-artifact-1",
            claim_id="claim-1",
            verification_id="verification-result-1",
        ),
    )
    assert verified["source_grade"] == "PRIMARY"
    assert verified["external_truth_status"] == "EXTERNALLY_VERIFIED"
    assert verified["trusted_verification"] == {
        "verification_artifact_id": "verification-artifact-1",
        "claim_id": "claim-1",
        "verification_id": "verification-result-1",
    }


def test_canonicalizer_keeps_the_full_bundle_v2_content_addressed():
    draft = ClaimOccurrenceDraft(
        semantic_segment_id="semantic-1", knowledge_kind="CLAIM", claim_type="FINANCIAL_METRIC",
        subject_type="EQUITY", subject_key="600000", predicate_key="revenue_growth",
        conclusion="公司收入增长12%", normalized_statement="公司收入增长12%",
        grounding_status="GROUNDED", claim_schema_version="claim.atomic.v1",
        legacy_grounding_incomplete=False, extraction_confidence=0.9,
    )
    canonicalizer = ClaimCanonicalizer()
    base = canonicalizer.canonicalize(draft)
    repeated = canonicalizer.canonicalize(draft)

    assert base.claim_id == claim_id_of(base)
    assert repeated.claim_id == base.claim_id
    # This fixture freezes the pre-admission/full-bundle canonical algorithm.
    # Verification belongs to occurrence provenance and must not create a
    # first-writer-wins variant of this canonical record.
    assert base.claim_id == "claim-dabe5622f4a20945a3985488"
    assert "trusted_verification" not in base.bundle_v2
    assert base.bundle_v2["source_grade"] != "PRIMARY"
    assert base.bundle_v2["external_truth_status"] != "EXTERNALLY_VERIFIED"


def test_canonical_stage_never_persists_verification_admission_on_canonical_claim():
    draft = ClaimOccurrenceDraft(
        semantic_segment_id="semantic-1", knowledge_kind="CLAIM", claim_type="FINANCIAL_METRIC",
        subject_type="EQUITY", subject_key="600000", predicate_key="revenue_growth",
        conclusion="公司收入增长12%", normalized_statement="公司收入增长12%",
        grounding_status="GROUNDED", claim_schema_version="claim.atomic.v1",
        legacy_grounding_incomplete=False, extraction_confidence=0.9,
    )

    def canonical_claim(token=None):
        context = PipelineContext(task_id="canonical-token", source={"type": "fixture", "ref": "video"})
        context.state.claim_drafts = [draft]
        context.state.temporal_bindings_by_draft = {0: []}
        context.state.trusted_external_verifications = {0: token} if token else {}
        ClaimCanonicalizationStage().execute(context)
        return context.state.claims[0]

    base = canonical_claim()
    token = TrustedExternalVerification(
        verification_artifact_id="verification-artifact-1",
        claim_id=base.claim_id,
        verification_id="verification-result-1",
    )
    verified_first = canonical_claim(token)
    base_after = canonical_claim()

    assert base_after.claim_id == verified_first.claim_id == base.claim_id
    assert base_after.bundle_v2 == verified_first.bundle_v2 == base.bundle_v2
    assert "trusted_verification" not in verified_first.bundle_v2
    assert verified_first.bundle_v2["external_truth_status"] != "EXTERNALLY_VERIFIED"


def test_occurrence_verification_token_requires_its_canonical_claim_identity():
    def persist(token_claim_id: str):
        context = PipelineContext(
            task_id="occurrence-token", source={"type": "fixture", "ref": "video"},
            options={"offline_fixture": True},
        )
        context.artifacts.transcript = _transcript()
        SemanticSegmentationStage().execute(context)
        semantic_id = context.state.semantic_segments[0].semantic_segment_id
        context.state.claim_drafts = [ClaimOccurrenceDraft(
            semantic_segment_id=semantic_id, knowledge_kind="CLAIM", claim_type="FORECAST",
            subject_type="THEME", subject_key="INFRA", predicate_key="investment_target",
            conclusion="信息基础设施投资预计到2030年达到三万亿元。",
            normalized_statement="信息基础设施投资预计到2030年达到三万亿元。",
            verbatim_quote="信息基础设施投资预计到2030年达到三万亿元。",
            evidence_segment_indices=[0], grounding_status="GROUNDED",
            claim_schema_version="claim.atomic.v1", legacy_grounding_incomplete=False,
            extraction_confidence=0.9,
        )]
        EvidenceGroundingStage().execute(context)
        ClaimCanonicalizationStage().execute(context)
        claim = context.state.claims[0]
        context.state.trusted_external_verifications = {0: TrustedExternalVerification(
            verification_artifact_id="verification-artifact", claim_id=token_claim_id,
            verification_id="verification-result",
        )}
        ClaimOccurrencePersistenceStage().execute(context)
        return claim, context.state.occurrences[0]

    baseline_claim, _ = persist("wrong-claim")
    claim, wrong = persist("wrong-claim")
    _, correct = persist(claim.claim_id)

    assert claim.claim_id == baseline_claim.claim_id
    assert wrong.provenance["bundle_v2"]["external_truth_status"] == "NOT_CHECKED"
    assert "trusted_verification" not in wrong.provenance["bundle_v2"]
    projected_semantic = KnowledgeProjectionBuilder().build(claim, wrong)["attributes"]["bundle_v2"]
    assert projected_semantic["external_truth_status"] == "NOT_CHECKED"
    assert correct.provenance["bundle_v2"]["external_truth_status"] == "EXTERNALLY_VERIFIED"
    assert correct.provenance["bundle_v2"]["trusted_verification"]["claim_id"] == claim.claim_id


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
        evidence_window_ids=("window-1",),
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
            "evidence_window_ids": ["window-1"],
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
    context.state.claim_evidence_window_ids = {0: ("window-1",)}
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
    packet = occurrence.provenance["visual_evidence"]
    assert packet["knowledge_id"] == occurrence.occurrence_id
    assert packet["status"] == "HUMAN_REVIEW_REQUIRED"
    assert packet["windows"][0]["frames"][0]["image_hash"] == "image-hash"
    assert packet["windows"][0]["frames"][0]["ocr"][0]["summary"] == "2030年 三万亿元"
    assert packet["windows"][0]["frames"][0]["vision"][0]["summary"] == "政策图表"
    assert all(value.startswith("sha256:") and len(value) == 71 for value in (
        packet["windows"][0]["frames"][0]["frame_artifact_hash"],
        packet["windows"][0]["frames"][0]["ocr"][0]["artifact_hash"],
        packet["windows"][0]["frames"][0]["vision"][0]["artifact_hash"],
    ))
    # This packet came from the real stage/artifact graph, not a hand-made
    # public payload.  It must satisfy the reusable v2 item validator.
    v2_item = {
            "knowledge_id": occurrence.occurrence_id,
            "claim_id": occurrence.claim_id,
            "occurrence_id": occurrence.occurrence_id,
            "statement": occurrence.normalized_statement,
            "subject": {"type": "THEME", "key": "INFORMATION_INFRASTRUCTURE"},
            "predicate": "investment_target",
            "object": {"value": "三万亿元", "unit": None},
            "primary_domain": "INFORMATION_INFRASTRUCTURE_POLICY",
            "claim_nature": "FORECAST",
            "attribution": {"attributed": True, "source_label": "source_speaker"},
            "source_grade": "SOURCE_ASSERTION",
            "detail": {"mechanism": "投资目标对应信息基础设施建设需求。"},
            "temporal": occurrence.provenance["bundle_v2"]["temporal"],
            "evidence": [{
                "evidence_id": "transcript-evidence", "ownership": "PRIMARY", "modality": "transcript",
                "artifact_id": "transcript-semantic", "artifact_hash": "sha256:" + "a" * 64,
                    "locator": {
                        "segment_id": "segment-1", "frame_id": None,
                        "start_ms": 10_000, "end_ms": 12_000, "bbox": None,
                    },
                "content": occurrence.normalized_statement,
            }],
            "occurrence_review": occurrence.provenance["bundle_v2"]["occurrence_review"],
            "support_status": "SOURCE_SUPPORTED",
            "lifecycle_status": "EXTRACTED",
            "verification": {"status": "NOT_CHECKED", "reason_codes": []},
            "external_truth_status": "NOT_CHECKED",
            "grounding_status": "GROUNDED",
            "claim_schema_version": "claim.atomic.v1",
            "legacy_grounding_incomplete": False,
            "visual_evidence": packet,
    }
    validate_v2_item(v2_item, minimum_support_status="SOURCE_SUPPORTED")
    schema = json.loads((Path(__file__).parents[1] / "contracts" / "content-knowledge-bundle.v2.json").read_text())
    item_schema = {"$defs": schema["$defs"], "$ref": "#/$defs/item"}
    public_item = {
        key: value for key, value in v2_item.items()
        if key not in {"claim_schema_version", "legacy_grounding_incomplete"}
    }
    # Review-blocked occurrences are kept by the authority for quality only;
    # the public schema admits only active items.  The sealed packet bytes are
    # unchanged for this schema probe.
    public_item["lifecycle_status"] = "ACTIVE"
    assert not list(Draft202012Validator(item_schema).iter_errors(public_item))

    database = Database(f"sqlite:///{tmp_path / 'semantic-evidence.db'}")
    database.create_schema()
    repository = ClaimOccurrenceRepository(database.session_factory)
    repository.save(occurrence)
    restored = repository.get(occurrence.occurrence_id)
    assert restored is not None
    assert set(restored.secondary_evidence_refs) == set(occurrence.secondary_evidence_refs)
    assert restored.provenance["visual_evidence"] == occurrence.provenance["visual_evidence"]

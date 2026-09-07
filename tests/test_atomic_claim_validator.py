from types import SimpleNamespace

from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import AtomicClaimValidationStage
from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.atomic_claim_validator import (
    AtomicClaimDraftValidator,
    ClaimRejectionCode,
    RestrictedAtomicClaimRepair,
)
from stock_content.domain.claim_draft import ClaimOccurrenceDraft
from stock_content.domain.semantic_segment import materialize_semantic_segments


def _transcript():
    return TranscriptArtifact(
        artifact_id="authoritative-transcript",
        artifact_type="transcript",
        media_artifact_id="media",
        asr_model="fixture",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(
                segment_index=0,
                start_seconds=0,
                end_seconds=1,
                text="600519营收增长10%，目标期间为2025年第三季度。",
                raw_text="600519营收增长10%，目标期间为2025年第三季度。",
                media_artifact_id="media",
                asr_model="fixture",
                asr_model_version="1",
            ),
            TranscriptSegmentItem(
                segment_index=1,
                start_seconds=1,
                end_seconds=2,
                text="条件是需求回暖，证伪条件是需求下降。",
                raw_text="条件是需求回暖，证伪条件是需求下降。",
                media_artifact_id="media",
                asr_model="fixture",
                asr_model_version="1",
            ),
        ],
    )


def _payload(segment_id, **changes):
    value = {
        "semantic_segment_id": segment_id,
        "verbatim_quote": "600519营收增长10%",
        "normalized_statement": "600519营收增长10%",
        "subject": {"subject_type": "EQUITY", "subject_key": "600519"},
        "predicate": "营收增长",
        "object": {"text": "10%", "value": "10%"},
        "sentiment": "BULLISH",
        "polarity": "ASSERTS",
        "assertion_tense": "PRESENT",
        "evidence_segment_indices": [0],
        "condition_evidence_segment_indices": [],
        "invalidation_evidence_segment_indices": [],
        "temporal_expressions": [],
        "visual_anchors": [],
        "extraction_confidence": 0.9,
    }
    value.update(changes)
    return value


def _validate(payload, status="PASS"):
    transcript = _transcript()
    segments = materialize_semantic_segments(transcript, [])
    return AtomicClaimDraftValidator().validate_payloads(
        {"claims": [payload]}, transcript, segments, transcript_quality_status=status
    )


def test_atomic_validator_accepts_only_quote_grounded_single_fact_with_separate_temporal_condition_and_invalidation():
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    payload = _payload(
        segment.semantic_segment_id,
        condition_text="条件是需求回暖",
        condition_evidence_segment_indices=[1],
        invalidation_text="证伪条件是需求下降",
        invalidation_evidence_segment_indices=[1],
        temporal_expressions=[
            {
                "raw_expression": "2025年第三季度",
                "target_period": "2025年第三季度",
                "pit_meaning": "REPORTING_PERIOD",
                "evidence_segment_indices": [0],
                "confidence": 0.8,
            }
        ],
    )
    result = AtomicClaimDraftValidator().validate_payloads({"claims": [payload]}, transcript, [segment])
    assert len(result.accepted) == 1 and not result.rejected


def test_atomic_validator_rejects_schema_coordinates_quote_atomicity_and_hard_fact_addition():
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    validator = AtomicClaimDraftValidator()
    cases = [
        ("{bad", ClaimRejectionCode.MALFORMED_JSON),
        (
            {"claims": [_payload(segment.semantic_segment_id, extraction_confidence=float("nan"))]},
            ClaimRejectionCode.SCHEMA_INVALID,
        ),
        (
            {"claims": [_payload(segment.semantic_segment_id, evidence_segment_indices=[9])]},
            ClaimRejectionCode.EVIDENCE_COORDINATE_INVALID,
        ),
        (
            {"claims": [_payload(segment.semantic_segment_id, verbatim_quote="营收增长11%")]},
            ClaimRejectionCode.QUOTE_NOT_VERBATIM,
        ),
        (
            {"claims": [_payload(segment.semantic_segment_id, normalized_statement="600519营收增长11%")]},
            ClaimRejectionCode.HARD_FACT_MISMATCH,
        ),
        (
            {"claims": [_payload(segment.semantic_segment_id, normalized_statement="600519营收增长10%；利润增长5%")]},
            ClaimRejectionCode.NOT_ATOMIC,
        ),
    ]
    for payload, reason in cases:
        result = validator.validate_payloads(payload, transcript, [segment])
        assert not result.accepted and result.rejected[0].reason_code is reason


def test_atomic_validator_rejects_subject_polarity_temporal_and_inferred_visual_financial_support():
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    validator = AtomicClaimDraftValidator()
    cases = [
        (_payload(segment.semantic_segment_id, subject={"subject_key": "000001"}), ClaimRejectionCode.SUBJECT_MISMATCH),
        (
            _payload(segment.semantic_segment_id, normalized_statement="600519营收下降10%"),
            ClaimRejectionCode.POLARITY_MISMATCH,
        ),
        (
            _payload(
                segment.semantic_segment_id,
                temporal_expressions=[{"raw_expression": "2026年", "evidence_segment_indices": [0], "confidence": 0.9}],
            ),
            ClaimRejectionCode.TEMPORAL_NOT_GROUNDED,
        ),
        (
            _payload(
                segment.semantic_segment_id,
                visual_anchors=[
                    {
                        "frame_id": "f-1",
                        "timestamp_ms": 10,
                        "bbox": [0, 0, 1, 1],
                        "visual_label": "chart",
                        "model_id": "vision",
                        "model_version": "1",
                        "confidence": 0.8,
                        "support_type": "INFERRED_VISUAL",
                    }
                ],
            ),
            ClaimRejectionCode.INFERRED_VISUAL_HIGH_RISK,
        ),
    ]
    for payload, reason in cases:
        result = validator.validate_payloads({"claims": [payload]}, transcript, [segment])
        assert result.rejected[0].reason_code is reason


def test_atomic_validator_retains_conflicts_with_stable_group_and_quality_stage_fails_closed():
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    first = _payload(segment.semantic_segment_id)
    second = _payload(segment.semantic_segment_id, sentiment="BEARISH", polarity="DENIES")
    validator = AtomicClaimDraftValidator()
    result = validator.validate_payloads({"claims": [second, first]}, transcript, [segment])
    again = validator.validate_payloads({"claims": [first, second]}, transcript, [segment])
    assert len(result.accepted) == 2
    assert {item.contradiction_group_id for item in result.accepted} == {
        item.contradiction_group_id for item in again.accepted
    }
    assert result.accepted[0].contradiction_group_id is not None
    assert (
        validator.validate_payloads(
            {"claims": [first]}, transcript, [segment], transcript_quality_status="NEEDS_REVIEW"
        )
        .rejected[0]
        .reason_code
        is ClaimRejectionCode.TRANSCRIPT_QUALITY_NOT_PASS
    )

    context = PipelineContext(
        task_id="atomic-stage",
        source={"type": "fixture", "ref": "one"},
        options={"atomic_claim_payload": {"claims": [first]}},
    )
    context.artifacts.transcript = transcript
    context.state["semantic_segments"] = [segment]
    context.state.transcript_quality_report = SimpleNamespace(quality_status="NEEDS_REVIEW")
    AtomicClaimValidationStage().execute(context)
    assert not context.state["validated_atomic_claims"]
    assert context.state["atomic_claim_rejections"][0].reason_code is ClaimRejectionCode.TRANSCRIPT_QUALITY_NOT_PASS


def test_repair_can_only_delete_coordinates_or_choose_real_quote_substring():
    repaired = RestrictedAtomicClaimRepair().apply(
        {"verbatim_quote": "bad", "evidence_segment_indices": [0, 1]},
        [
            {"kind": "DELETE_COORDINATE", "field": "evidence_segment_indices", "index": 1},
            {"kind": "REPLACE_QUOTE", "quote": "营收增长10%"},
        ],
        "600519营收增长10%",
    )
    assert repaired == {"verbatim_quote": "营收增长10%", "evidence_segment_indices": [0]}


def test_actual_extractor_draft_is_validated_before_formal_projection():
    """Normal extraction has no client atomic payload escape hatch."""
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    context = PipelineContext(task_id="atomic-stage", source={"type": "fixture", "ref": "one"})
    context.artifacts.transcript = transcript
    context.state.semantic_segments = [segment]
    context.state.transcript_quality_report = SimpleNamespace(quality_status="PASS")
    context.state.claim_drafts = [
        ClaimOccurrenceDraft(
            semantic_segment_id=segment.semantic_segment_id,
            knowledge_kind="EARNINGS",
            claim_type="FINANCIAL_METRIC",
            subject_type="EQUITY",
            subject_key="600519",
            predicate_key="营收增长",
            conclusion="600519营收增长10%",
            value="10%",
            sentiment="BULLISH",
            evidence_segment_indices=[0],
            extraction_confidence=0.9,
        )
    ]

    AtomicClaimValidationStage().execute(context)

    assert len(context.state.validated_atomic_claims) == 1
    assert context.state.claim_drafts[0].grounding_status == "GROUNDED"
    assert context.state.claim_drafts[0].claim_schema_version == "claim.atomic.v1"
    assert context.state.claim_drafts[0].verbatim_quote == "600519营收增长10%，目标期间为2025年第三季度。"


def test_invalid_actual_extractor_draft_is_review_only_and_not_formal():
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    context = PipelineContext(task_id="atomic-stage", source={"type": "fixture", "ref": "one"})
    context.artifacts.transcript = transcript
    context.state.semantic_segments = [segment]
    context.state.transcript_quality_report = SimpleNamespace(quality_status="PASS")
    context.state.claim_drafts = [
        ClaimOccurrenceDraft(
            semantic_segment_id=segment.semantic_segment_id,
            knowledge_kind="EARNINGS",
            claim_type="FINANCIAL_METRIC",
            subject_type="EQUITY",
            subject_key="600519",
            predicate_key="营收增长",
            conclusion="600519营收增长11%",
            value="11%",
            sentiment="BULLISH",
            evidence_segment_indices=[0],
            extraction_confidence=0.9,
            # An extractor cannot promote itself by setting these values.
            grounding_status="GROUNDED",
            claim_schema_version="claim.atomic.v1",
            legacy_grounding_incomplete=False,
        )
    ]

    AtomicClaimValidationStage().execute(context)

    assert context.state.claim_drafts == []
    assert context.state.atomic_claim_rejections[0].reason_code is ClaimRejectionCode.HARD_FACT_MISMATCH

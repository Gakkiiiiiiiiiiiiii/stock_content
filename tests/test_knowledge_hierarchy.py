from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.atomic_claim_validator import AtomicClaimDraftValidator
from stock_content.domain.claim_draft import ClaimOccurrenceDraft
from stock_content.domain.claim_draft_grounder import ClaimDraftGrounder
from stock_content.domain.knowledge_bundle_v2 import validate_v2_item
from stock_content.domain.knowledge_evidence_window import (
    KnowledgeEvidenceWindowPlanner,
    evidence_segment_indices_for_draft,
)
from stock_content.domain.knowledge_hierarchy import (
    ChapterThesisIdentifier,
    attach_thesis_hierarchy,
    materialize_thesis_claim_drafts,
)
from stock_content.domain.knowledge_semantics import (
    bundle_v2_semantics,
    normalize_detail_text,
    starts_with_detail_source_framing,
)
from stock_content.domain.semantic_segment import SemanticSegment

FIXTURE = Path(__file__).parent / "fixtures" / "sep19_storage_thesis.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _transcript(payload):
    start, end = payload["segment_range"]
    text = payload["source_text"]
    segments = [
        TranscriptSegmentItem(
            segment_index=index,
            start_seconds=(index - start) * 1.25,
            end_seconds=(index - start + 1) * 1.25,
            text=text.get(str(index), "同主题论证延续"),
            raw_text=text.get(str(index), "同主题论证延续"),
        )
        for index in range(start, end + 1)
    ]
    return TranscriptArtifact(
        artifact_id="transcript-storage",
        artifact_type="transcript",
        media_artifact_id="media-storage",
        segments=segments,
        asr_model="fixture",
        asr_model_version="v1",
    )


def _semantic(transcript):
    first, last = transcript.segments[0], transcript.segments[-1]
    return SemanticSegment(
        semantic_segment_id="semantic-storage",
        transcript_artifact_id=transcript.artifact_id,
        segment_index=0,
        start_segment_index=first.segment_index,
        end_segment_index=last.segment_index,
        start_segment_id=first.segment_id,
        end_segment_id=last.segment_id,
        start_ms=first.start_ms,
        end_ms=last.end_ms,
        topic="存储景气",
        subject="存储",
    )


def _split_semantics(transcript):
    boundary = 596
    first = transcript.segments[0]
    middle = next(item for item in transcript.segments if item.segment_index == boundary)
    following = next(item for item in transcript.segments if item.segment_index == boundary + 1)
    last = transcript.segments[-1]
    return [
        SemanticSegment(
            semantic_segment_id="semantic-storage-proposal",
            transcript_artifact_id=transcript.artifact_id,
            segment_index=0,
            start_segment_index=first.segment_index,
            end_segment_index=middle.segment_index,
            start_segment_id=first.segment_id,
            end_segment_id=middle.segment_id,
            start_ms=first.start_ms,
            end_ms=middle.end_ms,
            topic="存储需求",
            subject="内存",
        ),
        SemanticSegment(
            semantic_segment_id="semantic-storage-conclusion",
            transcript_artifact_id=transcript.artifact_id,
            segment_index=1,
            start_segment_index=following.segment_index,
            end_segment_index=last.segment_index,
            start_segment_id=following.segment_id,
            end_segment_id=last.segment_id,
            start_ms=following.start_ms,
            end_ms=last.end_ms,
            topic="存储价格",
            subject="内存",
        ),
    ]


def _draft(*, statement, indices, claim_type, nature):
    return ClaimOccurrenceDraft(
        semantic_segment_id="semantic-storage",
        knowledge_kind="CLAIM",
        claim_type=claim_type,
        subject_type="THEME",
        subject_key="存储",
        subject_name="存储",
        predicate_key="依赖" if nature == "CAUSAL_THESIS" else "价格预测",
        conclusion=statement,
        normalized_statement=statement,
        verbatim_quote=statement,
        evidence_segment_indices=indices,
        sentiment="BULLISH",
        extraction_confidence=0.9,
        bundle_v2={"claim_nature": nature},
    )


def test_segments_570_to_625_form_one_storage_thesis_with_child_roles():
    payload = _fixture()
    transcript = _transcript(payload)
    thesis = ChapterThesisIdentifier().identify(
        transcript,
        [_semantic(transcript)],
        [],
        fixture_theses=[payload["thesis"]],
        offline_fixture=True,
    )[0]
    children = attach_thesis_hierarchy(
        [
            _draft(
                statement="容纳模型所需的内存空间与内存带宽是模型推理的关键条件。",
                indices=[597, 598, 599, 600, 601, 602],
                claim_type="INFERENCE",
                nature="CAUSAL_THESIS",
            ),
            _draft(
                statement="NAND Flash价格在2026年第三季度和第四季度仍有上涨预期。",
                indices=[608, 609, 610, 611, 612],
                claim_type="FORECAST",
                nature="FORECAST",
            ),
        ],
        [thesis],
    )

    assert thesis.knowledge_title == "AI需求支撑存储景气延续"
    assert thesis.knowledge_role == "THESIS"
    assert thesis.evidence_segment_indices == tuple(range(570, 626))
    assert thesis.detailed_explanation.startswith("AI训练")
    assert [item.bundle_v2["knowledge_role"] for item in children] == ["MECHANISM", "EVIDENCE"]
    assert {item.bundle_v2["parent_knowledge_id"] for item in children} == {thesis.thesis_id}

    parent = materialize_thesis_claim_drafts([thesis])[0]
    windows = KnowledgeEvidenceWindowPlanner(padding_ms=0).plan_claim_drafts(
        transcript, [parent, *children]
    )
    parent_window, mechanism_window, evidence_window = windows
    assert parent_window.transcript_segment_ids == tuple(item.segment_id for item in transcript.segments)
    assert parent_window.sampling_strategy == "THESIS_PHASES"
    assert len(parent_window.transcript_anchor_ms) == 3
    assert mechanism_window.transcript_segment_ids == tuple(
        transcript.segments[index - 570].segment_id for index in range(597, 603)
    )
    assert evidence_window.transcript_segment_ids == tuple(
        transcript.segments[index - 570].segment_id for index in range(608, 613)
    )
    assert len(windows) == 3


def test_fresh_thesis_uses_model_selected_title_and_transcript_indices():
    segments = [
        TranscriptSegmentItem(segment_index=11, start_seconds=0, end_seconds=1, text="示例材料需求增长。"),
        TranscriptSegmentItem(segment_index=12, start_seconds=1, end_seconds=2, text="示例材料产能受到约束。"),
        TranscriptSegmentItem(segment_index=13, start_seconds=2, end_seconds=3, text="示例材料价格可能继续上行。"),
    ]
    transcript = TranscriptArtifact(
        artifact_id="transcript-generic", artifact_type="transcript", media_artifact_id="media-generic",
        segments=segments, asr_model="fixture", asr_model_version="v1",
    )
    response = {"theses": [{
        "knowledge_title": "示例材料供需趋紧",
        "atomic_statement": "示例材料需求增长。",
        "subject_type": "THEME", "subject_key": "示例材料", "subject_name": "示例材料",
        "predicate": "需求增长", "claim_type": "INFERENCE", "sentiment": "BULLISH",
        "attribution": "示例转录", "detailed_explanation": "需求增长且产能受限，价格存在上行可能。",
        "proposal_segment_indices": [11], "argument_segment_indices": [12],
        "conclusion_segment_indices": [13],
    }]}

    class Gateway:
        def available(self):
            return True

        def complete(self, **kwargs):
            assert "示例材料" in kwargs["prompt"]
            return {"content": json.dumps(response, ensure_ascii=False), "model": "gpt-6-sol"}

    theses = ChapterThesisIdentifier(Gateway(), model_id="gpt-6-sol").identify(
        transcript, [_semantic(transcript)], [SimpleNamespace(start_seconds=0, end_seconds=3)]
    )
    assert len(theses) == 1
    assert theses[0].knowledge_title == "示例材料供需趋紧"
    assert theses[0].evidence_segment_indices == (11, 12, 13)
    assert materialize_thesis_claim_drafts(theses)[0].extraction_model_id == "gpt-6-sol"
    with pytest.raises(ValueError, match="offline tests"):
        ChapterThesisIdentifier().identify(
            transcript, [_semantic(transcript)], [], fixture_theses=response["theses"]
        )


def test_storage_thesis_is_materialized_and_validated_across_semantic_segments():
    payload = _fixture()
    transcript = _transcript(payload)
    semantics = _split_semantics(transcript)
    thesis = ChapterThesisIdentifier().identify(
        transcript,
        semantics,
        [],
        fixture_theses=[payload["thesis"]],
        offline_fixture=True,
    )[0]
    parent = materialize_thesis_claim_drafts([thesis])[0]

    assert parent.bundle_v2["knowledge_role"] == "THESIS"
    assert parent.bundle_v2["hierarchy_node_id"] == thesis.thesis_id
    assert parent.evidence_segment_indices == list(range(570, 626))

    result = AtomicClaimDraftValidator().validate_payloads(
        {"claims": [{
            "semantic_segment_id": parent.semantic_segment_id,
            "claim_type": parent.claim_type,
            "knowledge_kind": parent.knowledge_kind,
            "verbatim_quote": parent.verbatim_quote,
            "normalized_statement": parent.normalized_statement,
            "subject": {
                "subject_type": parent.subject_type,
                "subject_key": parent.subject_key,
                "subject_name": parent.subject_name,
            },
            "predicate": parent.predicate_key,
            "object": None,
            "condition_text": None,
            "invalidation_text": None,
            "sentiment": parent.sentiment,
            "polarity": "ASSERTS",
            "assertion_tense": "UNKNOWN",
            "evidence_segment_indices": parent.evidence_segment_indices,
            "condition_evidence_segment_indices": [],
            "invalidation_evidence_segment_indices": [],
            "temporal_expressions": [],
            "visual_anchors": [],
            "bundle_v2": parent.bundle_v2,
            "extraction_confidence": parent.extraction_confidence,
        }]},
        transcript,
        semantics,
        transcript_quality_status="PASS",
    )
    assert len(result.accepted) == 1
    assert not result.rejected

    grounded = ClaimDraftGrounder().ground(parent, transcript, semantics[0])
    assert len(grounded.evidences) == 56


def test_only_parent_thesis_expands_to_full_rhetorical_scope():
    payload = _fixture()
    transcript = _transcript(payload)
    thesis = ChapterThesisIdentifier().identify(
        transcript,
        [_semantic(transcript)],
        [],
        fixture_theses=[payload["thesis"]],
        offline_fixture=True,
    )[0]
    parent = materialize_thesis_claim_drafts([thesis])[0]
    child = attach_thesis_hierarchy(
        [
            _draft(
                statement="容纳模型所需的内存空间与内存带宽是模型推理的关键条件。",
                indices=[597, 598],
                claim_type="INFERENCE",
                nature="CAUSAL_THESIS",
            )
        ],
        [thesis],
    )[0]
    available = set(range(570, 626))

    assert evidence_segment_indices_for_draft(parent, available) == tuple(range(570, 626))
    assert evidence_segment_indices_for_draft(child, available) == (597, 598)


def test_adjacent_forecast_is_child_but_does_not_expand_parent_scope():
    payload = _fixture()
    transcript = _transcript(payload)
    thesis = ChapterThesisIdentifier().identify(
        transcript,
        [_semantic(transcript)],
        [],
        fixture_theses=[payload["thesis"]],
        offline_fixture=True,
    )[0]
    forecast = _draft(
        statement="涨幅是在18到23左右",
        indices=[627, 628, 629, 630],
        claim_type="FORECAST",
        nature="INDUSTRY_CYCLE_FORECAST",
    )

    child = attach_thesis_hierarchy([forecast], [thesis])[0]

    assert child.bundle_v2["knowledge_role"] == "EVIDENCE"
    assert child.bundle_v2["parent_knowledge_id"] == thesis.thesis_id
    assert child.evidence_segment_indices == [627, 628, 629, 630]


@pytest.mark.parametrize(
    ("nature", "statement", "role"),
    [
        ("DRUG_DISCOVERY_WORKFLOW", "它亲自去合成这个药物的分子", "MECHANISM"),
        ("DRUG_MODALITY_MECHANISM", "这样子的话它就更加稳定", "MECHANISM"),
        ("BIOMEDICAL_DATA_ECONOMICS", "AI制药被数据卡住", "MECHANISM"),
        ("INDUSTRY_CYCLE_FORECAST", "涨幅是在18到23左右", "EVIDENCE"),
    ],
)
def test_domain_specific_natures_receive_semantic_child_roles(nature, statement, role):
    draft = _draft(
        statement=statement,
        indices=[600],
        claim_type="INFERENCE",
        nature=nature,
    )
    assert attach_thesis_hierarchy(
        [draft],
        [ChapterThesisIdentifier().identify(
            _transcript(_fixture()),
            [_semantic(_transcript(_fixture()))],
            [],
            fixture_theses=[_fixture()["thesis"]],
            offline_fixture=True,
        )[0]],
    )[0].bundle_v2["knowledge_role"] == role


@pytest.mark.parametrize(
    "value",
    [
        "讲述者先说明存储需求仍在增长。",
        "视频展示：存储价格预测仍在上行。",
        "节目中提到，存储价格短期难以回落。",
        "口播认为存储需求继续增长。",
        "转录文本指出存储价格仍有支撑。",
    ],
)
def test_detailed_explanation_moves_source_boilerplate_out_of_content(value):
    assert starts_with_detail_source_framing(value)
    normalized = normalize_detail_text(value)
    assert normalized
    assert not starts_with_detail_source_framing(normalized)


def test_semantic_envelope_normalizes_detail_but_keeps_attribution_separate():
    envelope = bundle_v2_semantics(
        statement="存储价格短期难以快速回落。",
        claim_type="INFERENCE",
        supplied={
            "detail": {"explanation": "讲述者先说明AI推理依赖内存容量与带宽。"},
            "attribution": {"attributed": True, "source_label": "source_speaker"},
            "knowledge_role": "MECHANISM",
            "parent_knowledge_id": "thesis-storage",
        },
    )
    assert envelope["detail"]["explanation"] == "AI推理依赖内存容量与带宽。"
    assert envelope["attribution"]["source_label"] == "source_speaker"
    assert envelope["knowledge_role"] == "MECHANISM"


def test_v2_rejects_unseparated_source_framing_and_orphan_child():
    item = {
        "knowledge_id": "knowledge-1",
        "claim_id": "claim-1",
        "occurrence_id": "knowledge-1",
        "statement": "存储价格短期难以回落。",
        "subject": {"type": "THEME", "key": "存储"},
        "predicate": "价格趋势",
        "object": {"value": None, "unit": None},
        "primary_domain": "UNKNOWN",
        "claim_nature": "OPINION",
        "attribution": {"attributed": True, "source_label": "source_speaker"},
        "source_grade": "SOURCE_ASSERTION",
        "detail": {"explanation": "讲述者认为存储需求仍有支撑。"},
        "temporal": {
            "kind": "UNKNOWN", "start": None, "end": None, "as_of": None,
            "rule": None, "label": None, "precision": "UNKNOWN", "explicitly_unknown": True,
        },
        "evidence": [{
            "evidence_id": "evidence-1", "ownership": "PRIMARY", "modality": "transcript",
            "artifact_id": "transcript-1", "artifact_hash": "sha256:" + "a" * 64,
            "locator": {"segment_id": "segment-1", "frame_id": None, "start_ms": 1, "end_ms": 2, "bbox": None},
            "content": "存储价格短期难以回落。",
        }],
        "occurrence_review": {"status": "NOT_REQUIRED", "reason_codes": []},
        "support_status": "SOURCE_SUPPORTED",
        "lifecycle_status": "ACTIVE",
        "verification": {"status": "NOT_CHECKED", "reason_codes": []},
        "external_truth_status": "NOT_CHECKED",
        "grounding_status": "GROUNDED",
        "claim_schema_version": "claim.atomic.v1",
        "legacy_grounding_incomplete": False,
        "knowledge_role": "MECHANISM",
    }
    with pytest.raises(ValueError, match="SOURCE_FRAMING_IN_KNOWLEDGE_DETAIL"):
        validate_v2_item(item, minimum_support_status="SOURCE_LOCATED")
    item["detail"] = {"explanation": "AI推理依赖内存容量与带宽。"}
    with pytest.raises(ValueError, match="PARENT_KNOWLEDGE_REQUIRED"):
        validate_v2_item(item, minimum_support_status="SOURCE_LOCATED")

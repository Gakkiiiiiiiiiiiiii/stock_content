"""Fail-closed checks for local, non-production coherent-knowledge previews."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_local_coherent_knowledge_preview import corrected_transcript, validate_extraction  # noqa: E402
from build_local_coherent_knowledge_preview_v2 import (  # noqa: E402
    previously_reviewed_plan,
    reviewed_equities,
    validate_group_plan,
)


def test_audio_correction_requires_matching_source_and_media() -> None:
    source = {
        "media": {"video_sha256": "media"},
        "segments": [{"segment_index": 0, "text": "原话"}],
    }
    review = {
        "source_transcript_sha256": "transcript",
        "media_sha256": "media",
        "corrections": [{
            "segment_index": 0,
            "source_text": "原话",
            "corrected_text": "更正",
            "decision": "TWO_ASR_MODELS_AGREE",
        }],
    }
    corrected = corrected_transcript(source, "transcript", "media", review)
    assert corrected["segments"][0]["text"] == "更正"
    assert source["segments"][0]["text"] == "原话"
    with pytest.raises(ValueError, match="does not match"):
        corrected_transcript(source, "wrong", "media", review)


def test_group_plan_covers_each_topic_once() -> None:
    plan = {
        "groups": [{"topic_indices": [1, 2], "focus": "完整论点"}],
        "excluded_topic_indices": [0],
    }
    assert len(validate_group_plan(plan, 3)) == 1
    plan["excluded_topic_indices"] = [0, 2]
    with pytest.raises(ValueError, match="exactly once"):
        validate_group_plan(plan, 3)
    sparse = {
        "groups": [{"topic_indices": [0, 2], "focus": "同一论点跨过一句插话"}],
        "excluded_topic_indices": [1],
    }
    assert validate_group_plan(sparse, 3)[0]["topic_indices"] == [0, 2]


def test_citation_paraphrase_is_replaced_with_exact_row_and_audited() -> None:
    packet = {
        "topics": [{"start_segment_index": 0, "end_segment_index": 1}],
        "unresolved_entity_windows": [],
    }
    transcript = {"segments": [{"text": "市场出现反弹"}, {"text": "但成交量仍需观察"}]}
    card = {
        "topic_indices": [0],
        "knowledge_title": "反弹需看成交量",
        "atomic_statement": "讲者认为市场反弹还需成交量确认。",
        "detailed_explanation": "反弹已经出现，但量能是否跟上仍是观察条件。",
        "primary_domain": "市场走势",
        "subject": "市场",
        "claim_nature": "OPINION",
        "evidence": [{"segment_index": 1, "quote": "量能还需观察"}],
        "applicability": None,
        "risks": None,
        "invalidation_conditions": None,
        "business_time_note": None,
        "spoken_stock_names": [],
        "spoken_stock_codes": [],
    }
    candidate = {"knowledge": [card], "excluded_topic_indices": []}
    validate_extraction(candidate, packet, transcript)
    assert card["atomic_statement"] == "市场反弹还需成交量确认。"
    assert card["evidence"][0]["quote"] == "但成交量仍需观察"
    assert candidate["citation_repairs"][0]["segment_index"] == 1


def test_unspoken_stock_code_fails_closed() -> None:
    packet = {
        "topics": [{"start_segment_index": 0, "end_segment_index": 0}],
        "unresolved_entity_windows": [],
    }
    transcript = {"segments": [{"text": "这家公司值得继续观察"}]}
    card = {
        "topic_indices": [0],
        "knowledge_title": "公司观察",
        "atomic_statement": "公司后续仍需观察。",
        "detailed_explanation": "材料只提供了笼统的公司观察，没有股票代码。",
        "primary_domain": "公司",
        "subject": "公司",
        "claim_nature": "OPINION",
        "evidence": [{"segment_index": 0, "quote": "这家公司值得继续观察"}],
        "applicability": None,
        "risks": None,
        "invalidation_conditions": None,
        "business_time_note": None,
        "spoken_stock_names": [],
        "spoken_stock_codes": ["123456"],
    }
    with pytest.raises(ValueError, match="Unspoken stock code"):
        validate_extraction({"knowledge": [card], "excluded_topic_indices": []}, packet, transcript)


def test_structured_time_conflicts_and_unresolved_items_are_coordinate_checked() -> None:
    packet = {
        "topics": [{"start_segment_index": 0, "end_segment_index": 1}],
        "unresolved_entity_windows": [],
    }
    transcript = {"segments": [{"text": "预计十月底完成"}, {"text": "金额单位没有说清楚"}]}
    card = {
        "topic_indices": [0], "knowledge_title": "时间与单位待核",
        "atomic_statement": "预计十月底完成，但金额单位未明确。",
        "detailed_explanation": "时间是预测节点，金额口径仍待确认。", "primary_domain": "项目",
        "subject": "进度", "claim_nature": "FORECAST",
        "evidence": [{"segment_index": 0, "quote": "预计十月底完成"}],
        "applicability": None, "risks": None, "invalidation_conditions": None,
        "business_time": {
            "as_of": None, "precision": "MONTH", "kind": "FORECAST",
            "expressions": [{
                "raw_text": "十月底", "normalized": None, "role": "FORECAST_END", "segment_indices": [0],
            }],
            "note": "年份未给出",
        },
        "conflicts": [],
        "unresolved_items": [{
            "kind": "UNIT", "raw_text": "金额单位", "segment_indices": [1],
            "reason": "口播未给出币种或量级", "status": "UNRESOLVED", "resolution": None,
        }],
        "spoken_stock_names": [], "spoken_stock_codes": [],
    }
    assert validate_extraction(
        {"knowledge": [card], "excluded_topic_indices": []}, packet, transcript, structured=True
    ) == [card]
    card["unresolved_items"][0]["segment_indices"] = [2]
    with pytest.raises(ValueError, match="unresolved item"):
        validate_extraction(
            {"knowledge": [card], "excluded_topic_indices": []}, packet, transcript, structured=True
        )


def test_reviewed_equity_requires_same_source_and_untampered_frame(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"reviewed chart")
    frame_hash = hashlib.sha256(frame.read_bytes()).hexdigest()
    review = {
        "schema_version": "local-equity-frame-review.v1",
        "status": "LOCAL_REVIEW_ONLY_NOT_SEALED_OR_PERSISTED",
        "requested_model": "gpt-6-sol", "audit": {"passed": True},
        "topic_map_sha256": "map", "source_transcript_sha256": "source",
        "media_sha256": "media", "audio_review_sha256": "audio",
        "mentions": [{"visual_evidence": [{"relative_path": "frame.jpg", "image_sha256": frame_hash}]}],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    loaded, _ = reviewed_equities(
        path, map_hash="map", source_hash="source", media_hash="media", audio_review_hash="audio"
    )
    assert len(loaded) == 1
    with pytest.raises(ValueError, match="provenance"):
        reviewed_equities(
            path, map_hash="other", source_hash="source", media_hash="media", audio_review_hash="audio"
        )
    frame.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        reviewed_equities(
            path, map_hash="map", source_hash="source", media_hash="media", audio_review_hash="audio"
        )


def test_reused_grouping_requires_audited_matching_source(tmp_path: Path) -> None:
    prior = {
        "schema_version": "local-coherent-knowledge.v1", "audit": {"passed": True},
        "topic_map_sha256": "map", "source_transcript_sha256": "source", "media_sha256": "media",
        "knowledge": [{"topic_indices": [0], "knowledge_title": "一条完整观点"}],
        "excluded_topic_indices": [1],
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior), encoding="utf-8")
    plan, _ = previously_reviewed_plan(
        path, map_hash="map", source_hash="source", media_hash="media", topic_count=2
    )
    assert plan["groups"][0]["topic_indices"] == [0]
    with pytest.raises(ValueError, match="provenance"):
        previously_reviewed_plan(
            path, map_hash="map", source_hash="other", media_hash="media", topic_count=2
        )

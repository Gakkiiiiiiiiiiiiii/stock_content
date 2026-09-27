"""Fail-closed checks for local, non-production coherent-knowledge previews."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_local_coherent_knowledge_preview import corrected_transcript, validate_extraction  # noqa: E402
from build_local_coherent_knowledge_preview_v2 import validate_group_plan  # noqa: E402


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

"""Cross-topic and empty-asset regression for generic video equity links."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stock_content.domain.knowledge_equity_links import link_equity_mentions


def _video():
    cards = [
        {"knowledge_id": "cloud", "topic_indices": [0]},
        {"knowledge_id": "gold", "topic_indices": [1]},
    ]
    stages = [
        {"stage_id": "T01", "topic_index": 0, "start_segment_index": 0,
         "end_segment_index": 1, "start_ms": 0, "end_ms": 2000},
        {"stage_id": "T02", "topic_index": 1, "start_segment_index": 2,
         "end_segment_index": 3, "start_ms": 2001, "end_ms": 4000},
    ]
    rows = [{"text": value} for value in ("云计算订单", "观察设备公司", "黄金走势", "观察金价")]
    return cards, stages, rows


def _mention(*, stage_id="T01", segment_index=1, tier="FOCUSED_CHART_SPOKEN"):
    return {
        "entity_id": "equity-1", "name": "示例设备", "code": "123456",
        "evidence_tier": tier,
        "identity_status": (
            "CONFIRMED_IN_VIDEO" if tier == "FOCUSED_CHART_SPOKEN"
            else "CONFIRMED_ON_SCREEN_SPEECH_LINK_UNRESOLVED"
        ),
        "stage_ids": [stage_id],
        "transcript_evidence": [{"segment_index": segment_index, "text": "观察设备公司"}],
        "visual_evidence": [{"timestamp_ms": 1500, "image_sha256": "a" * 64}],
    }


def test_other_video_links_only_evidence_scoped_topic() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "unresolved_entity_in_range": True,
        "reason_codes": ["ENTITY_NAME_UNRESOLVED"],
        "unresolved_items": [{
            "kind": "ENTITY", "raw_text": "设备公司", "segment_indices": [1],
            "reason": "ASR 名称歧义", "status": "UNRESOLVED", "resolution": None,
        }],
    })
    linked, audit = link_equity_mentions(cards, stages, rows, [_mention()])
    mention = linked[0]["equity_mentions"][0]
    assert mention["raw_spoken_mentions"] == [{"segment_index": 1, "text": "观察设备公司"}]
    assert mention["canonical_identity"] == {
        "name": "示例设备", "code": "123456", "market": None,
        "code_status": "CONFIRMED_IN_VIDEO", "identity_status": "CONFIRMED_IN_VIDEO",
    }
    assert mention["speech_link_status"] == "SPOKEN_AND_DISPLAYED_CONFIRMED"
    assert mention["visual_evidence"][0]["frame_id"] == f"frame_{'a' * 24}"
    assert linked[0]["unresolved_items"][0]["status"] == "RESOLVED_BY_VISUAL"
    assert linked[0]["unresolved_entity_in_range"] is False
    assert linked[1]["equity_mentions"] == []
    assert audit["linked_mention_count"] == 1


def test_audio_corrected_visual_link_records_cross_modal_resolution() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "unresolved_entity_in_range": True,
        "reason_codes": ["ENTITY_NAME_UNRESOLVED"],
        "unresolved_items": [{
            "kind": "ENTITY", "raw_text": "设备公司", "segment_indices": [1],
            "reason": "ASR 名称歧义", "status": "UNRESOLVED", "resolution": None,
        }],
    })
    mention = _mention()
    mention["raw_entity_text"] = "设备公司"
    mention["asr_correction_supported"] = True

    linked, _ = link_equity_mentions(cards, stages, rows, [mention])

    assert linked[0]["unresolved_items"][0]["status"] == "RESOLVED_BY_CROSS_MODAL"


def test_other_video_does_not_join_across_topic_boundary() -> None:
    cards, stages, rows = _video()
    linked, audit = link_equity_mentions(cards, stages, rows, [_mention(stage_id="T02")])
    assert all(not card["equity_mentions"] for card in linked)
    assert audit["unlinked"] == [{"entity_id": "equity-1", "reason": "NO_EVIDENCE_SCOPED_KNOWLEDGE"}]


def test_visual_only_frame_cannot_override_conflicting_speech_scope() -> None:
    cards, stages, rows = _video()
    mention = _mention(stage_id="T02", tier="FOCUSED_CHART_VISUAL_ONLY")
    mention["visual_evidence"][0]["timestamp_ms"] = 2500
    linked, audit = link_equity_mentions(cards, stages, rows, [mention])
    assert all(not card["equity_mentions"] for card in linked)
    assert audit["unlinked"][0]["reason"] == "NO_EVIDENCE_SCOPED_KNOWLEDGE"


def test_visual_only_frame_may_immediately_follow_matching_speech_scope() -> None:
    cards, stages, rows = _video()
    mention = _mention(stage_id="T01", tier="FOCUSED_CHART_VISUAL_ONLY")
    mention["visual_evidence"][0]["timestamp_ms"] = 2500
    linked, audit = link_equity_mentions(cards, stages, rows, [mention])
    assert linked[0]["equity_mentions"][0]["canonical_identity"]["name"] == "示例设备"
    assert linked[1]["equity_mentions"] == []
    assert audit["linked_mention_count"] == 1


def test_other_video_without_stocks_keeps_empty_cards() -> None:
    cards, stages, rows = _video()
    linked, audit = link_equity_mentions(cards, stages, rows, [])
    assert all(not card["equity_mentions"] for card in linked)
    assert audit["reviewed_mention_count"] == audit["linked_mention_count"] == 0


def test_displayed_only_never_upgrades_spoken_identity() -> None:
    cards, stages, rows = _video()
    cards[0]["unresolved_entity_in_range"] = True
    cards[0]["reason_codes"] = ["ENTITY_NAME_UNRESOLVED"]
    cards[0]["unresolved_items"] = [{
        "kind": "ENTITY", "raw_text": "设备公司", "segment_indices": [1],
        "reason": "口播关联待核", "status": "UNRESOLVED", "resolution": None,
    }]
    linked, _ = link_equity_mentions(
        cards, stages, rows, [_mention(tier="FOCUSED_CHART_VISUAL_ONLY")]
    )
    assert linked[0]["equity_mentions"][0]["link_status"] == "DISPLAYED_ONLY_SPEECH_LINK_UNRESOLVED"
    assert linked[0]["equity_mentions"][0]["speech_link_status"] == "DISPLAYED_ONLY_SPEECH_LINK_UNRESOLVED"
    assert linked[0]["unresolved_items"][0]["status"] == "UNRESOLVED"
    assert linked[0]["unresolved_entity_in_range"] is True
    assert linked[0]["equity_mentions"][0]["recommendation_status"] == "NOT_A_RECOMMENDATION"


def test_slide_spoken_entity_supports_hk_market_without_inventing_code() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "unresolved_entity_in_range": True,
        "reason_codes": ["ENTITY_NAME_UNRESOLVED"],
        "unresolved_items": [{
            "kind": "ENTITY", "raw_text": "硬席智能", "segment_indices": [1],
            "reason": "ASR 名称歧义", "status": "UNRESOLVED", "resolution": None,
        }],
    })
    mention = {
        **_mention(tier="SLIDE_ENTITY_SPOKEN"),
        "name": "英矽智能",
        "code": None,
        "market": "HK",
        "code_status": "NOT_VISIBLE_IN_VIDEO",
        "identity_status": "CONFIRMED_IN_VIDEO",
    }
    linked, audit = link_equity_mentions(cards, stages, rows, [mention])
    projected = linked[0]["equity_mentions"][0]
    assert projected["canonical_identity"] == {
        "name": "英矽智能",
        "code": None,
        "market": "HK",
        "code_status": "NOT_VISIBLE_IN_VIDEO",
        "identity_status": "CONFIRMED_IN_VIDEO",
    }
    assert projected["raw_spoken_mentions"] == [{"segment_index": 1, "text": "观察设备公司"}]
    assert linked[0]["unresolved_items"][0]["status"] == "RESOLVED_BY_VISUAL"
    assert linked[0]["unresolved_items"][0]["resolution"] == "英矽智能（港股）"
    assert audit["linked_mention_count"] == 1


@pytest.mark.skipif(not os.getenv("OTHER_VIDEO_TRANSCRIPT"), reason="other-video transcript not supplied")
def test_different_local_video_rejects_cross_topic_and_accepts_no_assets() -> None:
    """Run against an independently stored video without embedding its contents."""
    transcript = json.loads(Path(os.environ["OTHER_VIDEO_TRANSCRIPT"]).read_text(encoding="utf-8"))
    rows = transcript["segments"]
    first_index, second_index = len(rows) // 3, (len(rows) * 2) // 3
    first, second = rows[first_index], rows[second_index]
    cards = [{"knowledge_id": "first", "topic_indices": [0]},
             {"knowledge_id": "second", "topic_indices": [1]}]
    stages = [
        {"stage_id": "T01", "topic_index": 0, "start_segment_index": first_index,
         "end_segment_index": first_index, "start_ms": round(first["start_seconds"] * 1000),
         "end_ms": round(first["end_seconds"] * 1000)},
        {"stage_id": "T02", "topic_index": 1, "start_segment_index": second_index,
         "end_segment_index": second_index, "start_ms": round(second["start_seconds"] * 1000),
         "end_ms": round(second["end_seconds"] * 1000)},
    ]
    no_assets, empty_audit = link_equity_mentions(cards, stages, rows, [])
    assert not any(card["equity_mentions"] for card in no_assets)
    assert empty_audit["linked_mention_count"] == 0
    wrong_scope = {
        "entity_id": "negative-probe", "name": "反例标的", "code": "123456",
        "evidence_tier": "FOCUSED_CHART_SPOKEN", "identity_status": "CONFIRMED_IN_VIDEO",
        "stage_ids": ["T02"],
        "transcript_evidence": [{"segment_index": first_index, "text": first["text"]}],
        "visual_evidence": [{"timestamp_ms": round(first["start_seconds"] * 1000),
                             "image_sha256": "a" * 64}],
    }
    linked, audit = link_equity_mentions(cards, stages, rows, [wrong_scope])
    assert not any(card["equity_mentions"] for card in linked)
    assert audit["unlinked"][0]["reason"] == "NO_EVIDENCE_SCOPED_KNOWLEDGE"

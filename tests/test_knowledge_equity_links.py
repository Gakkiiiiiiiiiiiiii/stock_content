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
    linked, audit = link_equity_mentions(cards, stages, rows, [_mention()])
    assert linked[0]["equity_mentions"][0]["code"] == "123456"
    assert linked[1]["equity_mentions"] == []
    assert audit["linked_mention_count"] == 1


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


def test_other_video_without_stocks_keeps_empty_cards() -> None:
    cards, stages, rows = _video()
    linked, audit = link_equity_mentions(cards, stages, rows, [])
    assert all(not card["equity_mentions"] for card in linked)
    assert audit["reviewed_mention_count"] == audit["linked_mention_count"] == 0


def test_displayed_only_never_upgrades_spoken_identity() -> None:
    cards, stages, rows = _video()
    linked, _ = link_equity_mentions(
        cards, stages, rows, [_mention(tier="FOCUSED_CHART_VISUAL_ONLY")]
    )
    assert linked[0]["equity_mentions"][0]["link_status"] == "DISPLAYED_ONLY_SPEECH_LINK_UNRESOLVED"
    assert linked[0]["equity_mentions"][0]["recommendation_status"] == "NOT_A_RECOMMENDATION"


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

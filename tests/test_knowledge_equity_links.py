"""Cross-topic and empty-asset regression for generic video equity links."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stock_content.domain.knowledge_equity_links import (
    link_equity_mentions,
    validate_confirmed_entity_prose_consistency,
)


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


def test_confirmed_spoken_entity_reconciles_only_identity_prose_and_preserves_raw_speech() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "knowledge_title": "身份未核实标的的指标曲线评价",
        "atomic_statement": "对身份未核实的标的，指标曲线被评价为漂亮。",
        "detailed_explanation": (
            "曲线评价属于主观判断。本卡口播名称为“设备公司”，"
            "仅凭本卡内容无法确认公司身份或股票代码。"
        ),
        "subject": "口播名称为“设备公司”、身份未核实的标的",
        "applicability": "仅适用于对该身份未核实标的的主观判断。",
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

    card = linked[0]
    assert card["knowledge_title"] == "示例设备（123456）的指标曲线评价"
    assert card["atomic_statement"] == "对示例设备（123456），指标曲线被评价为漂亮。"
    assert card["detailed_explanation"] == (
        "曲线评价属于主观判断。本卡原始口播为“设备公司”，"
        "同期画面确认其对应示例设备（123456）。"
    )
    assert card["subject"] == "示例设备（123456）"
    assert card["applicability"] == "仅适用于对示例设备（123456）的主观判断。"
    assert card["equity_mentions"][0]["raw_spoken_mentions"] == [
        {"segment_index": 1, "text": "观察设备公司"}
    ]
    assert card["entity_prose_reconciliation"] == [{
        "entity_id": "equity-1",
        "canonical_name": "示例设备",
        "raw_spoken_text": "设备公司",
        "segment_indices": [1],
        "changed_fields": [
            "knowledge_title", "atomic_statement", "detailed_explanation",
            "subject", "applicability",
        ],
        "policy": "EXACT_SPOKEN_COORDINATE_AND_CONFIRMED_DISPLAY_ONLY",
    }]


def test_confirmed_entity_does_not_rewrite_unresolved_item_at_other_spoken_coordinate() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "knowledge_title": "身份未核实标的的指标曲线评价",
        "atomic_statement": "对身份未核实的标的进行观察。",
        "detailed_explanation": "本卡仍无法确认公司身份。",
        "unresolved_entity_in_range": True,
        "reason_codes": ["ENTITY_NAME_UNRESOLVED"],
        "unresolved_items": [{
            "kind": "ENTITY", "raw_text": "另一家公司", "segment_indices": [0],
            "reason": "不同口播坐标", "status": "UNRESOLVED", "resolution": None,
        }],
    })

    linked, _ = link_equity_mentions(cards, stages, rows, [_mention()])

    assert linked[0]["knowledge_title"] == cards[0]["knowledge_title"]
    assert linked[0]["unresolved_items"][0]["status"] == "UNRESOLVED"
    assert "entity_prose_reconciliation" not in linked[0]


def test_confirmed_entity_rewrites_its_scoped_clause_without_touching_other_unresolved_entity() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "knowledge_title": "转录名称未确认的科技股标的表现比较",
        "atomic_statement": (
            "某名称转录为“设备公司”、具体身份未确认的标的表现更强，"
            "另一名称转录为“另一家公司”、身份同样未确认的标的表现较弱。"
        ),
        "detailed_explanation": "视频比较了两个身份未确认的标的。",
        "unresolved_entity_in_range": True,
        "reason_codes": ["ENTITY_NAME_UNRESOLVED"],
        "unresolved_items": [
            {
                "kind": "ENTITY", "raw_text": "设备公司", "segment_indices": [1],
                "reason": "ASR 名称歧义", "status": "UNRESOLVED", "resolution": None,
            },
            {
                "kind": "ENTITY", "raw_text": "另一家公司", "segment_indices": [0],
                "reason": "仍待确认", "status": "UNRESOLVED", "resolution": None,
            },
        ],
    })
    mention = _mention()
    mention["raw_entity_text"] = "设备公司"
    mention["asr_correction_supported"] = True

    linked, _ = link_equity_mentions(cards, stages, rows, [mention])

    assert linked[0]["knowledge_title"] == "示例设备（123456）表现比较"
    assert linked[0]["atomic_statement"] == (
        "示例设备（123456）表现更强，"
        "另一名称转录为“另一家公司”、身份同样未确认的标的表现较弱。"
    )
    assert linked[0]["detailed_explanation"] == (
        "视频比较了示例设备（123456）与另一个身份未确认的标的。"
    )
    assert linked[0]["unresolved_items"][1]["status"] == "UNRESOLVED"


def test_confirmed_entity_rewrites_identity_only_qualifiers_without_changing_thesis() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "knowledge_title": "基础工具类标的的弹性判断",
        "atomic_statement": (
            "视频提到一个转录作“设备公司”的标的可能更有弹性；"
            "该标的身份及部分术语未获核实。"
        ),
        "detailed_explanation": (
            "弹性判断属于视频观点；“设备公司”也不能据此确认为某家上市公司。"
        ),
        "subject": "转录作“设备公司”的标的（公司身份未确认）",
        "applicability": "具体证券身份和压力位未确认。",
        "unresolved_entity_in_range": True,
        "reason_codes": ["ENTITY_NAME_UNRESOLVED"],
        "unresolved_items": [{
            "kind": "ENTITY", "raw_text": "设备公司", "segment_indices": [1],
            "reason": "ASR 名称歧义", "status": "UNRESOLVED", "resolution": None,
        }],
    })
    mention = _mention()
    mention["raw_entity_text"] = "设备公司"

    linked, _ = link_equity_mentions(cards, stages, rows, [mention])

    assert linked[0]["atomic_statement"] == (
        "视频提到示例设备（123456）可能更有弹性；"
        "该标的身份已由同期画面确认，部分术语仍未获核实。"
    )
    assert linked[0]["detailed_explanation"] == (
        "弹性判断属于视频观点；同期画面已确认其规范身份为示例设备（123456）。"
    )
    assert linked[0]["subject"] == "示例设备（123456）"
    assert linked[0]["applicability"] == (
        "具体证券身份已由同期画面确认，压力位仍未确认。"
    )


def test_consistency_gate_rejects_stale_identity_prose_for_confirmed_same_coordinate() -> None:
    card = {
        "knowledge_id": "stale",
        "topic_indices": [0],
        "knowledge_title": "身份未核实标的的走势判断",
        "atomic_statement": "走势判断。",
        "detailed_explanation": "本卡口播名称为“设备公司”，无法确认公司身份。",
        "unresolved_items": [{
            "kind": "ENTITY", "raw_text": "设备公司", "segment_indices": [1],
            "status": "RESOLVED_BY_CROSS_MODAL", "resolution": "示例设备（123456）",
        }],
        "equity_mentions": [{
            "entity_id": "equity-1",
            "name": "示例设备",
            "code": "123456",
            "canonical_identity": {"name": "示例设备", "code": "123456", "market": None},
            "speech_link_status": "SPOKEN_AND_DISPLAYED_CONFIRMED",
            "raw_spoken_mentions": [{"segment_index": 1, "text": "观察设备公司"}],
        }],
    }

    with pytest.raises(ValueError, match="Confirmed entity identity remains unresolved in prose"):
        validate_confirmed_entity_prose_consistency([card])


def test_spoken_entity_resolves_overbroad_legacy_item_when_raw_name_is_literal_subset() -> None:
    cards, stages, rows = _video()
    cards[0].update({
        "unresolved_entity_in_range": True,
        "reason_codes": ["ENTITY_NAME_UNRESOLVED"],
        "unresolved_items": [{
            "kind": "ENTITY", "raw_text": "观察设备公司，随后继续讲走势", "segment_indices": [1],
            "reason": "旧版范围过大", "status": "UNRESOLVED", "resolution": None,
        }],
    })
    mention = _mention()
    mention["raw_entity_text"] = "观察设备公司"

    linked, _ = link_equity_mentions(cards, stages, rows, [mention])

    assert linked[0]["unresolved_items"][0]["status"] == "RESOLVED_BY_VISUAL"


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


def test_visual_only_declared_stage_allows_audited_chart_tail_without_spoken_upgrade() -> None:
    cards, stages, rows = _video()
    mention = _mention(stage_id="T01", tier="FOCUSED_CHART_VISUAL_ONLY")
    mention["transcript_evidence"] = []
    mention["visual_evidence"][0]["timestamp_ms"] = 11_500

    linked, audit = link_equity_mentions(cards, stages, rows, [mention])

    projected = linked[0]["equity_mentions"][0]
    assert projected["speech_link_status"] == "DISPLAYED_ONLY_SPEECH_LINK_UNRESOLVED"
    assert projected["transcript_evidence"] == []
    assert projected["visual_evidence"][0]["frame_scope_status"] == "POST_TOPIC_CORROBORATION"
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

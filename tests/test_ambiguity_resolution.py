import pytest

from stock_content.domain.ambiguity_resolution import (
    ambiguity_item_id,
    card_has_high_risk_ambiguity,
    granular_unresolved_items,
    partition_review_items,
    validate_triage_decision,
)


def test_granular_items_drop_inherited_superset_window() -> None:
    items = [
        {"kind": "ENTITY", "raw_text": "小德字", "segment_indices": [517], "status": "UNRESOLVED"},
        {
            "kind": "ENTITY",
            "raw_text": "包含小德字的整段上下文",
            "segment_indices": list(range(515, 525)),
            "status": "UNRESOLVED",
        },
        {"kind": "TERM", "raw_text": "两小音量", "segment_indices": [519], "status": "UNRESOLVED"},
    ]

    selected = granular_unresolved_items(items)

    assert [(item["kind"], item["segment_indices"]) for item in selected] == [
        ("ENTITY", [517]),
        ("TERM", [519]),
    ]


def test_partition_keeps_resolutions_out_of_pending_list() -> None:
    pending, resolved = partition_review_items([
        {"status": "UNRESOLVED"},
        {"status": "RESOLVED_BY_AUDIO"},
        {"status": "RESOLVED_BY_CROSS_MODAL"},
        {"status": "RESOLVED_BY_VIDEO_CONTEXT"},
    ])

    assert [item["status"] for item in pending] == ["UNRESOLVED"]
    assert [item["status"] for item in resolved] == [
        "RESOLVED_BY_AUDIO",
        "RESOLVED_BY_CROSS_MODAL",
        "RESOLVED_BY_VIDEO_CONTEXT",
    ]


def test_ambiguity_item_identity_is_card_scoped_and_coordinate_stable() -> None:
    item = {"kind": "ENTITY", "raw_text": "通負", "segment_indices": [4, 3, 4]}

    assert ambiguity_item_id("K01", item) == ambiguity_item_id(
        "K01", {**item, "segment_indices": [3, 4]}
    )
    assert ambiguity_item_id("K01", item) != ambiguity_item_id("K02", item)


def test_high_risk_card_excludes_term_and_date_only_items() -> None:
    assert not card_has_high_risk_ambiguity(
        {"unresolved_items": [{"kind": "TERM", "status": "UNRESOLVED"}]}
    )
    assert card_has_high_risk_ambiguity(
        {"unresolved_items": [{"kind": "UNIT", "status": "UNRESOLVED"}]}
    )


def test_triage_requires_entity_type_and_limits_video_context_to_dates() -> None:
    source = {
        "item_id": "ambiguity-1",
        "knowledge_id": "K01",
        "kind": "ENTITY",
        "raw_text": "小德字",
        "segment_indices": [7],
    }
    decision = validate_triage_decision(
        {
            "item_id": "ambiguity-1",
            "action": "VISUAL_REVIEW_REQUIRED",
            "corrected_kind": None,
            "entity_type": "EQUITY",
            "candidate_text": None,
            "resolution": None,
            "reason": "口播疑似股票简称，需要同期画面确认。",
        },
        source,
    )
    assert decision["entity_type"] == "EQUITY"

    with pytest.raises(ValueError, match="entity type"):
        validate_triage_decision({**decision, "entity_type": None}, source)

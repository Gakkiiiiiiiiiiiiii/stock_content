from stock_content.domain.ambiguity_resolution import (
    granular_unresolved_items,
    partition_review_items,
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

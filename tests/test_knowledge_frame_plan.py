from dataclasses import replace

from stock_content.domain.knowledge_evidence_window import HighSignal, KnowledgeEvidenceWindow
from stock_content.domain.knowledge_frame_plan import (
    HIGH_SIGNAL,
    KNOWLEDGE_CENTER,
    KNOWLEDGE_NEARBY,
    KnowledgeFramePlanner,
    evidence_window_id,
    frame_id_for,
)


def _window(
    semantic_id: str,
    center_ms: int,
    *,
    high_signal: bool = False,
) -> KnowledgeEvidenceWindow:
    return KnowledgeEvidenceWindow(
        semantic_segment_id=semantic_id,
        start_ms=max(0, center_ms - 2_000),
        end_ms=center_ms + 2_000,
        center_ms=center_ms,
        transcript_segment_ids=(f"ts-{semantic_id}",),
        high_signals=(HighSignal("NUMERIC", "numeric_value", (f"ts-{semantic_id}",)),) if high_signal else (),
    )


def test_targeted_plan_adds_center_nearby_and_dense_high_signal_offsets_deterministically():
    normal = _window("semantic-normal", 5_000)
    high = _window("semantic-high", 15_000, high_signal=True)
    planner = KnowledgeFramePlanner(max_frames_per_media=20)

    first = planner.plan([high, normal], media_duration_ms=20_000)
    second = planner.plan([normal, high], media_duration_ms=20_000)

    assert first == second
    assert [(item.timestamp_ms, item.extraction_reason) for item in first] == [
        (4_000, KNOWLEDGE_NEARBY),
        (5_000, KNOWLEDGE_CENTER),
        (6_000, KNOWLEDGE_NEARBY),
        (12_000, HIGH_SIGNAL),
        (13_000, HIGH_SIGNAL),
        (14_000, HIGH_SIGNAL),
        (15_000, KNOWLEDGE_CENTER),
        (16_000, HIGH_SIGNAL),
        (17_000, HIGH_SIGNAL),
        (18_000, HIGH_SIGNAL),
    ]


def test_targeted_plan_clamps_deduplicates_and_prioritizes_centres_under_global_cap():
    edge = _window("semantic-edge", 0, high_signal=True)
    later = _window("semantic-later", 9_000)
    plan = KnowledgeFramePlanner(max_frames_per_window=7, max_frames_per_media=2).plan(
        [later, edge], media_duration_ms=10_000
    )

    assert [(item.timestamp_ms, item.extraction_reason) for item in plan] == [
        (0, KNOWLEDGE_CENTER),
        (9_000, KNOWLEDGE_CENTER),
    ]
    assert plan[0].semantic_segment_ids == ("semantic-edge",)
    assert all(0 <= item.timestamp_ms <= 10_000 for item in plan)


def test_frame_identity_binds_media_timestamp_reason_version_and_window_identity():
    window = _window("semantic-one", 5_000)
    request = KnowledgeFramePlanner().plan([window], media_duration_ms=10_000)[0]
    same = frame_id_for(media_artifact_id="media-content-a", request=request)
    changed_media = frame_id_for(media_artifact_id="media-content-b", request=request)
    changed_reason = replace(request, extraction_reason=KNOWLEDGE_CENTER)

    assert same == frame_id_for(media_artifact_id="media-content-a", request=request)
    assert same != changed_media
    assert same != frame_id_for(media_artifact_id="media-content-a", request=changed_reason)
    assert evidence_window_id(window).startswith("kew_")
    assert "https://" not in repr(request)

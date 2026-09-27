from dataclasses import replace
from types import SimpleNamespace

from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.knowledge_evidence_window import (
    HighSignal,
    KnowledgeEvidenceWindow,
    KnowledgeEvidenceWindowPlanner,
)
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


def test_thesis_phase_sampling_keeps_full_window_but_only_three_spread_frames():
    window = KnowledgeEvidenceWindow(
        semantic_segment_id="semantic-storage",
        start_ms=900_000,
        end_ms=990_000,
        center_ms=945_000,
        transcript_segment_ids=tuple(f"segment-{index}" for index in range(570, 626)),
        transcript_anchor_ms=(910_000, 945_000, 980_000),
        high_signals=(),
        knowledge_identity="storage-thesis",
        sampling_strategy="THESIS_PHASES",
    )

    plan = KnowledgeFramePlanner().plan([window], media_duration_ms=2_000_000)

    assert [item.timestamp_ms for item in plan] == [910_000, 945_000, 980_000]
    assert all(item.extraction_reason == KNOWLEDGE_CENTER for item in plan)
    assert len(window.transcript_segment_ids) == 56


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
        (1_000, HIGH_SIGNAL),
        (2_000, HIGH_SIGNAL),
        (3_000, HIGH_SIGNAL),
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


def test_claim_directed_windows_keep_unrelated_drafts_in_one_semantic_segment_distinct():
    transcript = TranscriptArtifact(
        artifact_id="transcript",
        artifact_type="transcript",
        media_artifact_id="media",
        asr_model="fixture",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(segment_index=0, start_seconds=10, end_seconds=12, text="第一个知识点"),
            TranscriptSegmentItem(segment_index=1, start_seconds=100, end_seconds=102, text="第二个知识点 2030年"),
        ],
    )
    windows = KnowledgeEvidenceWindowPlanner().plan_claim_drafts(
        transcript,
        [
            SimpleNamespace(
                semantic_segment_id="semantic-long", evidence_segment_indices=[0], normalized_statement="第一"
            ),
            SimpleNamespace(
                semantic_segment_id="semantic-long", evidence_segment_indices=[1], normalized_statement="第二"
            ),
        ],
        media_duration_ms=120_000,
    )

    assert len(windows) == 2
    assert [item.center_ms for item in windows] == [11_000, 101_000]
    assert len({evidence_window_id(item) for item in windows}) == 2
    plan = KnowledgeFramePlanner(max_frames_per_media=120).plan(windows, media_duration_ms=120_000)
    by_window = {
        window_id: [request.timestamp_ms for request in plan if window_id in request.evidence_window_ids]
        for window_id in {evidence_window_id(item) for item in windows}
    }
    assert by_window[evidence_window_id(windows[0])] == [10_000, 11_000, 12_000]
    assert 101_000 in by_window[evidence_window_id(windows[1])]
    assert len(by_window[evidence_window_id(windows[1])]) == 7  # date cue -> dense +/-1..3 seconds


def test_global_budget_never_silently_drops_an_accepted_window_centre():
    windows = [_window(f"semantic-{index}", index * 1_000) for index in range(121)]
    plan = KnowledgeFramePlanner(max_frames_per_media=120).plan(windows, media_duration_ms=121_000)
    centres = {item.timestamp_ms for item in plan if item.extraction_reason == KNOWLEDGE_CENTER}
    assert centres == {index * 1_000 for index in range(121)}


def test_global_budget_keeps_all_centres_and_the_full_high_signal_plus_minus_three_window():
    windows = [_window(f"semantic-{index}", index * 10_000) for index in range(121)]
    windows[73] = _window("semantic-73", 730_000, high_signal=True)

    plan = KnowledgeFramePlanner(max_frames_per_media=120).plan(windows, media_duration_ms=1_300_000)

    centres = {item.timestamp_ms for item in plan if item.extraction_reason == KNOWLEDGE_CENTER}
    assert centres == {index * 10_000 for index in range(121)}
    assert {item.timestamp_ms for item in plan if "semantic-73" in item.semantic_segment_ids} == {
        727_000, 728_000, 729_000, 730_000, 731_000, 732_000, 733_000,
    }


def test_claim_window_anchors_every_evidence_segment_and_each_high_signal_segment():
    transcript = TranscriptArtifact(
        artifact_id="transcript-anchors", artifact_type="transcript", media_artifact_id="media",
        asr_model="fixture", asr_model_version="1",
        segments=[
            TranscriptSegmentItem(segment_index=0, start_seconds=0, end_seconds=1, text="600000收入增长12%"),
            TranscriptSegmentItem(segment_index=1, start_seconds=100, end_seconds=101, text="另一条证据"),
        ],
    )
    window = KnowledgeEvidenceWindowPlanner().plan_claim_drafts(
        transcript,
        [SimpleNamespace(
            semantic_segment_id="semantic-two-anchors",
            evidence_segment_indices=[0, 1],
            normalized_statement="两条证据",
        )],
        media_duration_ms=105_000,
    )[0]

    plan = KnowledgeFramePlanner(max_frames_per_media=1).plan([window], media_duration_ms=105_000)

    timestamps = {item.timestamp_ms for item in plan}
    assert 500 in timestamps
    assert 100_500 in timestamps
    assert {0, 500, 1_500, 2_500, 3_500}.issubset(timestamps)


def test_target_cue_uses_actual_spoken_anchor_and_keeps_mandatory_evidence_under_soft_caps():
    window = KnowledgeEvidenceWindow(
        semantic_segment_id="semantic-target", start_ms=0, end_ms=10_000, center_ms=5_000,
        transcript_segment_ids=("intro", "target"), transcript_anchor_ms=(1_000, 9_000),
        high_signals=(HighSignal("TARGET_CUE", "displayed_target_cue", ("target",), (9_000,)),),
    )
    plan = KnowledgeFramePlanner(max_frames_per_window=7, max_frames_per_media=1).plan(
        [window], media_duration_ms=10_000
    )
    timestamps = {item.timestamp_ms for item in plan}
    assert {1_000, 9_000, 6_000, 7_000, 8_000, 10_000} <= timestamps
    assert any(item.timestamp_ms == 9_000 and item.extraction_reason == KNOWLEDGE_CENTER for item in plan)
    assert all(item.planner_version == "knowledge-frame-plan.v3" for item in plan)

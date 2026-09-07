from stock_content.application.quality_report import QualityMetrics, evaluate_quality
from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.knowledge_evidence_window import KnowledgeEvidenceWindowPlanner
from stock_content.domain.semantic_segment import materialize_semantic_segments


def _transcript() -> TranscriptArtifact:
    return TranscriptArtifact(
        artifact_id="transcript-window-test",
        artifact_type="transcript",
        media_artifact_id="media-window-test",
        asr_model="fixture-asr",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(segment_index=0, start_seconds=0, end_seconds=2, text="普通开场。"),
            TranscriptSegmentItem(
                segment_index=1,
                start_seconds=2,
                end_seconds=7,
                text="2026年9月，600519贵州茅台股份有限公司收入增长12%，政策文件支持半导体行业。",
            ),
            TranscriptSegmentItem(
                segment_index=2,
                start_seconds=7,
                end_seconds=10,
                text="GPU产品走势图显示银行板块成交量变化。",
            ),
        ],
    )


def test_plans_are_monotonic_bounded_and_deduplicated_by_semantic_identity():
    transcript = _transcript()
    first, second = materialize_semantic_segments(transcript, [{"after_segment_index": 0}])
    plans = KnowledgeEvidenceWindowPlanner(padding_ms=1500).plan(
        transcript, [second, first, second], media_duration_ms=8_000
    )
    assert [item.semantic_segment_id for item in plans] == [first.semantic_segment_id, second.semantic_segment_id]
    assert [(item.start_ms, item.end_ms, item.center_ms) for item in plans] == [(0, 3_500, 1_000), (500, 8_000, 5_000)]
    assert all(0 <= item.start_ms <= item.center_ms <= item.end_ms <= 8_000 for item in plans)
    assert plans[1].transcript_segment_ids == tuple(item.segment_id for item in transcript.segments[1:])


def test_plans_are_deterministic_and_classify_all_high_signal_kinds():
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    planner = KnowledgeEvidenceWindowPlanner()
    first = planner.plan(transcript, [segment], media_duration_ms=10_000)
    second = planner.plan(transcript, [segment], media_duration_ms=10_000)
    assert first == second
    assert {item.kind for item in first[0].high_signals} == {
        "NUMERIC", "UNIT", "DATE", "TICKER", "COMPANY", "PRODUCT", "INDUSTRY", "CHART", "POLICY_DOCUMENT"
    }
    assert all(item.reason and item.transcript_segment_ids for item in first[0].high_signals)
    assert "https://" not in repr(first[0])


def test_empty_no_signal_segment_is_planned_without_mutating_transcript_quality_behavior():
    transcript = TranscriptArtifact(
        artifact_id="empty-window-test",
        artifact_type="transcript",
        media_artifact_id="media-empty-test",
        asr_model="fixture-asr",
        asr_model_version="1",
        segments=[],
    )
    # An empty transcript has no semantic segments and produces no plan.
    assert KnowledgeEvidenceWindowPlanner().plan(transcript, [], media_duration_ms=0) == ()
    quality_before = evaluate_quality(QualityMetrics(1, 1, 1, 1)).gate_result
    non_signal = TranscriptArtifact(
        artifact_id="no-signal-window-test",
        artifact_type="transcript",
        media_artifact_id="media-no-signal-test",
        asr_model="fixture-asr",
        asr_model_version="1",
        segments=[TranscriptSegmentItem(segment_index=0, start_seconds=3, end_seconds=4, text="你好。")],
    )
    segment = materialize_semantic_segments(non_signal, [])[0]
    plan = KnowledgeEvidenceWindowPlanner().plan(non_signal, [segment], media_duration_ms=4_000)[0]
    assert plan.high_signals == ()
    assert plan.transcript_segment_ids == (non_signal.segments[0].segment_id,)
    assert evaluate_quality(QualityMetrics(1, 1, 1, 1)).gate_result == quality_before

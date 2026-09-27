"""Deterministic Stage 1 protocol and semantic-boundary golden cases.

These tests use a response stub rather than a language model.  The contract
under test is the boundary-only protocol and the deterministic materializer;
the fixture text is intentionally not interpreted by the test runner.
"""

from __future__ import annotations

import itertools
import json

import pytest

from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.semantic_segment import SemanticBoundary
from stock_content.domain.semantic_segmenter import SemanticSegmenter


class BoundaryOnlyGateway:
    def available(self):
        return True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["temperature"] == 0.0
        assert kwargs["response_format"] == {"type": "json_object"}
        return self.responses.pop(0)


def _transcript(labels: list[str], artifact_id: str = "golden-transcript") -> TranscriptArtifact:
    return TranscriptArtifact(
        artifact_id=artifact_id,
        artifact_type="transcript",
        media_artifact_id="golden-media",
        asr_model="stub-asr",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(
                segment_index=index,
                start_seconds=float(index),
                end_seconds=float(index + 1),
                text=label,
                raw_text=label,
                media_artifact_id="golden-media",
                asr_model="stub-asr",
                asr_model_version="1",
            )
            for index, label in enumerate(labels)
        ],
    )


def _response(boundaries: list[dict]) -> dict:
    return {"content": json.dumps({"boundaries": boundaries}, ensure_ascii=False)}


def _adjudication(indices: list[int]) -> dict:
    return {"content": json.dumps({"accepted_after_segment_indices": indices})}


def _boundary(index: int, *, confidence: float = 0.9, subject: str | None = None) -> dict:
    return {
        "after_segment_index": index,
        "boundary_type": "TOPIC_CHANGE",
        "next_topic": "next topic",
        "next_subject": subject,
        "confidence": confidence,
    }


def _ranges(result):
    return [(item.start_segment_index, item.end_segment_index) for item in result.segments]


def test_stage1_stub_contract_is_boundary_only_and_materializes_full_coverage():
    gateway = BoundaryOnlyGateway([_response([_boundary(1, subject="半导体")])])
    result = SemanticSegmenter(gateway).segment(
        _transcript(["宏观背景", "行业证据", "个股结论"], "golden-contract")
    )

    assert len(gateway.calls) == 1
    assert "Return exactly" in gateway.calls[0]["prompt"]
    assert _ranges(result) == [(0, 1), (2, 2)]
    assert result.segments[1].subject == "半导体"
    assert result.metrics["boundary_count"] == 1.0
    # This is the precision/recall oracle for this fixture: one expected and
    # one predicted boundary, with no over- or under-segmentation.
    assert {1} == {item.after_segment_index for item in [SemanticBoundary(1)]}


def test_fresh_segmentation_labels_the_first_topic_and_persists_it():
    gateway = BoundaryOnlyGateway([{
        "content": json.dumps({
            "initial_topic": "宏观通胀与就业",
            "initial_subject": "宏观经济",
            "boundaries": [_boundary(1, subject="人工智能算力")],
        }, ensure_ascii=False)
    }])
    result = SemanticSegmenter(gateway, require_initial_topic=True).segment(
        _transcript(["居民通胀", "就业变化", "算力需求", "芯片供给"])
    )
    assert result.segments[0].topic == "宏观通胀与就业"
    assert result.segments[0].subject == "宏观经济"
    assert result.artifact.segments[0].topic == "宏观通胀与就业"
    assert result.segments[1].subject == "人工智能算力"


def test_second_pass_recovers_short_subject_switches_inside_coarse_segments():
    gateway = BoundaryOnlyGateway([
        _response([_boundary(3, subject="后半部分")]),
        _response([_boundary(1, subject="新话题 A")]),
        _adjudication([1]),
        _response([_boundary(5, subject="新话题 B")]),
        _adjudication([5]),
    ])
    result = SemanticSegmenter(
        gateway, refine_segments=True, refinement_min_tokens=1
    ).segment(_transcript(["题材一", "题材一", "题材二", "题材二", "题材三", "题材三", "题材四", "题材四"]))
    assert _ranges(result) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert result.metrics["refinement_call_count"] == 2
    assert result.metrics["refinement_added_boundary_count"] == 2
    assert result.metrics["adjudication_call_count"] == 2


def test_refinement_rejects_causal_detail_split_inside_one_thesis():
    gateway = BoundaryOnlyGateway([
        {"content": json.dumps({
            "initial_topic": "算力交付与供应商利润", "initial_subject": "AI 算力",
            "boundaries": [],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "initial_topic": "HBM 对 GPU 交付的约束", "initial_subject": "HBM",
            "boundaries": [_boundary(1, subject="芯片供应商")],
        }, ensure_ascii=False)},
        _adjudication([]),
        {"content": json.dumps({
            "initial_topic": "算力交付与供应商利润", "initial_subject": "AI 算力",
            "boundaries": [],
        }, ensure_ascii=False)},
    ])
    result = SemanticSegmenter(
        gateway, refine_segments=True, require_initial_topic=True
    ).segment(_transcript([
        "算力交付受 HBM 良率约束", "HBM 产能决定服务器交付",
        "算力需求不等于芯片供应商利润增长", "供应商利润要看订单与资本开支风险",
    ]))
    assert _ranges(result) == [(0, 3)]
    assert result.segments[0].topic == "算力交付与供应商利润"
    assert result.metrics["adjudication_rejected_boundary_count"] == 1


def test_brief_label_grounding_corrects_a_misnamed_instrument_from_exact_quote():
    gateway = BoundaryOnlyGateway([
        {"content": json.dumps({
            "initial_topic": "市場背景", "initial_subject": "市場",
            "boundaries": [{
                "after_segment_index": 6, "boundary_type": "TOPIC_CHANGE",
                "next_topic": "券商反彈", "next_subject": "券商", "confidence": 0.9,
            }],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "topic": "能否反彈", "subject": None, "source_quote": "轉債會不會反彈",
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "topic": "轉債反彈可能性", "subject": "轉債", "source_quote": "轉債會不會反彈",
        }, ensure_ascii=False)},
    ])
    result = SemanticSegmenter(
        gateway, require_initial_topic=True, verify_brief_topic_labels=True
    ).segment(_transcript(["市場背景"] * 7 + ["看看轉債會不會反彈", "轉債目前偏弱"]))
    assert _ranges(result) == [(0, 6), (7, 8)]
    assert result.segments[1].subject == "轉債"
    assert result.artifact.segments[1].topic == "轉債反彈可能性"
    assert result.metrics["brief_label_corrected_count"] == 1
    assert result.metrics["repair_count"] == 1


def test_refinement_replaces_a_mixed_coarse_topic_label():
    gateway = BoundaryOnlyGateway([
        {"content": json.dumps({
            "initial_topic": "宏观和科技混合主题", "initial_subject": None,
            "boundaries": [_boundary(2, subject="科技")],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "initial_topic": "宏观通胀与就业", "initial_subject": "宏观经济", "boundaries": [],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "initial_topic": "AI 算力供给", "initial_subject": "人工智能算力", "boundaries": [],
        }, ensure_ascii=False)},
    ])
    result = SemanticSegmenter(
        gateway, refine_segments=True, refinement_min_tokens=1, require_initial_topic=True
    ).segment(_transcript(["宏观", "通胀", "就业", "算力", "芯片", "供给"]))
    assert [segment.topic for segment in result.segments] == ["宏观通胀与就业", "AI 算力供给"]
    assert result.segments[1].subject == "人工智能算力"


def test_composite_topic_audit_recovers_independent_subjects_in_a_recap():
    gateway = BoundaryOnlyGateway([
        {"content": json.dumps({
            "initial_topic": "造船、设备与医药方向", "initial_subject": "行业方向", "boundaries": [],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "initial_topic": "造船、设备与医药方向", "initial_subject": "行业方向", "boundaries": [],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "initial_topic": "造船可关注", "initial_subject": "造船",
            "boundaries": [
                _boundary(1, subject="上游设备"),
                _boundary(3, subject="AI 医药"),
            ],
        }, ensure_ascii=False)},
        _adjudication([1, 3]),
    ])
    result = SemanticSegmenter(
        gateway, refine_segments=True, require_initial_topic=True
    ).segment(_transcript(["造船", "船舶订单", "科技设备", "设备弹性", "AI 医药", "AIDD 研发"]))
    assert _ranges(result) == [(0, 1), (2, 3), (4, 5)]
    assert result.segments[0].topic == "造船可关注"
    assert result.metrics["topic_audit_call_count"] == 1
    assert result.metrics["topic_audit_added_boundary_count"] == 2


def test_topic_coverage_audit_recovers_brief_asset_view_missing_from_label():
    gateway = BoundaryOnlyGateway([
        {"content": json.dumps({
            "initial_topic": "成交量变化", "initial_subject": "市场成交量", "boundaries": [],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "initial_topic": "成交量变化", "initial_subject": "市场成交量", "boundaries": [],
        }, ensure_ascii=False)},
        {"content": json.dumps({
            "initial_topic": "转债反弹可能性", "initial_subject": "转债",
            "boundaries": [_boundary(1, subject="成交量")],
        }, ensure_ascii=False)},
        _adjudication([1]),
    ])
    rows = [
        "转债虽然相对偏弱，但可能出现反弹，需要独立观察价格与风险。" * 3,
        "转债观点的前提是市场风险偏好改善，不能直接等同于大盘量能。" * 3,
        "量化交易减少后，成交量更接近真实交易，需看持续性。" * 3,
        "成交量的有效性还要结合价格走势判断。" * 3,
    ]
    result = SemanticSegmenter(
        gateway, refine_segments=True, require_initial_topic=True
    ).segment(_transcript(rows))
    assert _ranges(result) == [(0, 1), (2, 3)]
    assert result.segments[0].topic == "转债反弹可能性"
    assert result.metrics["topic_audit_added_boundary_count"] == 1


def test_narrow_overlap_audit_recovers_late_subject_change_in_many_short_rows():
    rows = ["格陵兰航线"] * 60 + ["中美谈判稀土"] * 31
    gateway = BoundaryOnlyGateway([
        _response([]),  # coarse pass misses the late switch
        _response([]),  # broad refinement also misses it
        _response([]),  # first narrow audit window has no switch
        _response([_boundary(59, subject="中美谈判")]),
        _adjudication([59]),
    ])
    result = SemanticSegmenter(
        gateway, refine_segments=True, require_initial_topic=False
    ).segment(_transcript(rows))
    assert _ranges(result) == [(0, 59), (60, 90)]
    assert result.metrics["topic_audit_call_count"] == 2
    assert result.metrics["adjudication_call_count"] == 1
    assert all("fixed elapsed time" in call["prompt"] or "INTERNAL topic splits" in call["prompt"]
               for call in gateway.calls if "prompt" in call)


@pytest.mark.parametrize(
    ("case", "labels", "expected_boundaries", "expected_ranges"),
    [
        (
            "single thesis over a long span",
            ["thesis evidence risk condition"] * 10,
            [],
            [(0, 9)],
        ),
        (
            "rapid stock switching",
            ["A thesis", "A evidence", "A conclusion", "B thesis", "B evidence", "B risk"],
            [2],
            [(0, 2), (3, 5)],
        ),
        (
            "macro to industry to stock",
            ["macro", "macro evidence", "industry", "industry evidence", "stock", "stock risk"],
            [1, 3],
            [(0, 1), (2, 3), (4, 5)],
        ),
        (
            "advertisement and disclaimer",
            ["analysis", "advertisement", "advertisement copy", "disclaimer", "analysis"],
            [0, 2, 3],
            [(0, 0), (1, 2), (3, 3), (4, 4)],
        ),
        (
            "Q&A with multiple speakers",
            ["speaker one thesis", "speaker one evidence", "question", "answer", "speaker two thesis"],
            [1, 3],
            [(0, 1), (2, 3), (4, 4)],
        ),
        (
            "conclusion and risk remain one thesis",
            ["thesis", "evidence", "conclusion", "risk", "failure condition"],
            [],
            [(0, 4)],
        ),
        (
            "same subject has two theses",
            [
                "Company A growth thesis",
                "Company A growth evidence",
                "Company A valuation thesis",
                "Company A valuation risk",
            ],
            [1],
            [(0, 1), (2, 3)],
        ),
        (
            "title and body disagree",
            ["title says Company A", "body discusses macro cycle", "macro evidence", "macro conclusion"],
            [],
            [(0, 3)],
        ),
    ],
)
def test_semantic_segmentation_golden_cases(case, labels, expected_boundaries, expected_ranges):
    del case  # The case name documents the oracle; the stub does no NLP.
    response = _response(
        [_boundary(index, subject=f"subject-{index}") for index in expected_boundaries]
    )
    result = SemanticSegmenter(BoundaryOnlyGateway([response])).segment(
        _transcript(labels, f"golden-{len(labels)}-{len(expected_boundaries)}")
    )

    assert _ranges(result) == expected_ranges
    assert len(result.segments) == len(expected_boundaries) + 1
    assert result.segments[0].start_segment_index == 0
    assert result.segments[-1].end_segment_index == len(labels) - 1
    assert all(
        left.end_segment_index + 1 == right.start_segment_index
        for left, right in zip(result.segments, result.segments[1:])
    )


def test_long_video_overlap_reconciliation_is_order_independent_and_avoids_overcut():
    # With serialized-row budgeting, this produces [0:6], [4:10],
    # [8:12]. The first two blocks disagree at adjacent coordinates;
    # coordinate 4 has repeated support and wins.
    responses = [
        _response([_boundary(4, confidence=0.8, subject="repeated")]),
        _response([_boundary(4, confidence=0.8, subject="repeated"), _boundary(5, confidence=0.8)]),
        _response([]),
    ]
    gateway = BoundaryOnlyGateway(responses)
    row_budget = SemanticSegmenter()._prompt_overhead() + 50
    result = SemanticSegmenter(gateway, safe_tokens=1, block_tokens=row_budget, segment_overlap=2).segment(
        _transcript([f"S{i}" for i in range(12)], "golden-long-overlap")
    )

    assert len(gateway.calls) == 3
    assert _ranges(result) == [(0, 4), (5, 11)]
    assert result.metrics["boundary_count"] == 1.0

    proposals = [
        SemanticBoundary(3, confidence=0.8),
        SemanticBoundary(2, confidence=0.8, next_subject="repeated"),
        SemanticBoundary(2, confidence=0.8, next_subject="repeated"),
    ]
    expected = [
        (item.after_segment_index, item.next_subject)
        for item in SemanticSegmenter._reconcile(proposals)
    ]
    for permutation in itertools.permutations(proposals):
        assert [
            (item.after_segment_index, item.next_subject)
            for item in SemanticSegmenter._reconcile(list(permutation))
        ] == expected


def test_stage1_repair_is_single_attempt_and_invalid_coordinates_fail_closed():
    bad = _response([_boundary(99)])
    valid = _response([])
    gateway = BoundaryOnlyGateway([bad, valid])
    result = SemanticSegmenter(gateway).segment(_transcript(["one", "two"], "golden-repair"))
    assert len(gateway.calls) == 2
    assert _ranges(result) == [(0, 1)]

    permanently_bad = BoundaryOnlyGateway([bad, bad])
    with pytest.raises(ValueError, match="after one repair"):
        SemanticSegmenter(permanently_bad).segment(_transcript(["one", "two"], "golden-fail"))


def test_chinese_long_video_budget_uses_serialized_rows_and_keeps_block_context():
    transcript = _transcript(["市场回踩后的条件判断" for _ in range(360)], "golden-cjk-budget")
    segmenter = SemanticSegmenter(safe_tokens=3200, block_tokens=3200, segment_overlap=12)
    items = list(transcript.segments)
    blocks = segmenter._blocks(items)

    assert segmenter._token_count(transcript) > 3200
    assert len(blocks) > 1
    assert all(
        sum(segmenter._item_token_cost(item) for item in items[start:end]) <= 3200 - segmenter._prompt_overhead()
        for start, end in blocks
    )
    assert all(left_end - right_start == 12 for (_, left_end), (right_start, _) in zip(blocks, blocks[1:]))
    assert "市场回踩后的条件判断" in segmenter._prompt(items, *blocks[0])


def test_very_long_asr_rows_cannot_exceed_budget_or_stall_block_progress():
    items = list(_transcript(["药物研发" * 28 for _ in range(24)]).segments)
    segmenter = SemanticSegmenter(block_tokens=800, segment_overlap=12)
    blocks = segmenter._blocks(items)
    assert len(blocks) <= len(items) // 2
    assert all(start < end for start, end in blocks)
    assert all(next_start - start >= 2 for (start, _), (next_start, _) in zip(blocks, blocks[1:]))
    assert blocks[-1][1] == len(items)

    too_long = list(_transcript(["药物研发" * 300]).segments)
    with pytest.raises(ValueError, match="single transcript segment exceeds"):
        segmenter._blocks(too_long)


def test_runtime_model_identity_is_required_only_for_live_model_calls():
    transcript = _transcript(["宏观", "设备"], "golden-model-identity")
    with pytest.raises(RuntimeError, match="runtime model identity mismatch"):
        SemanticSegmenter(
            BoundaryOnlyGateway([_response([])]), model_id="gpt-6-sol", require_model_identity=True
        ).segment(transcript)

    response = {**_response([]), "model": "gpt-6-sol"}
    result = SemanticSegmenter(
        BoundaryOnlyGateway([response]), model_id="gpt-6-sol", require_model_identity=True
    ).segment(transcript)
    assert _ranges(result) == [(0, 1)]


def test_explicit_offline_fixture_never_uses_model_gateway():
    gateway = BoundaryOnlyGateway([])
    result = SemanticSegmenter(gateway).segment(
        _transcript(["开场", "行业主题", "风险"], "golden-offline"), offline_fixture=True
    )
    assert gateway.calls == []
    assert _ranges(result) == [(0, 2)]

def test_stage1_block_prompt_and_repair_state_dynamic_legal_coordinate_range():
    items = list(_transcript([f"segment-{index}" for index in range(8)]).segments)
    # In the nonzero block [3:7], index 6 is the final supplied segment.  It
    # is intentionally invalid even though it is globally a valid transcript
    # coordinate; the repair request must repeat the same local constraint.
    gateway = BoundaryOnlyGateway([_response([_boundary(6)]), _response([])])
    segmenter = SemanticSegmenter(gateway)

    boundaries, initial_topic, initial_subject, repairs = segmenter._call_with_repair(items, 3, 7)

    assert boundaries == []
    assert initial_topic is None
    assert initial_subject is None
    assert repairs == 1
    assert len(gateway.calls) == 2
    expected_constraint = "inclusive range [3, 5]. Never emit 6"
    assert expected_constraint in gateway.calls[0]["prompt"]
    assert expected_constraint in gateway.calls[1]["prompt"]


def test_stage1_rejects_claims_timestamps_and_other_non_boundary_output():
    invalid = {
        "content": json.dumps(
            {
                "boundaries": [
                    {
                        **_boundary(0),
                        "claim": "forbidden",
                        "timestamp": 1.0,
                    }
                ]
            }
        )
    }
    valid = _response([])
    gateway = BoundaryOnlyGateway([invalid, valid])
    result = SemanticSegmenter(gateway).segment(_transcript(["one", "two"], "golden-protocol"))
    assert len(gateway.calls) == 2
    assert _ranges(result) == [(0, 1)]

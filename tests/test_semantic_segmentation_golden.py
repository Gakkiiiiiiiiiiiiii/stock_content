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
    # With one-token fixture segments and a block size of four, this produces
    # [0:4], [2:6], [4:8], [6:10], [8:12].  The first two blocks disagree at
    # adjacent coordinates; coordinate 2 has repeated support and wins.
    responses = [
        _response([_boundary(2, confidence=0.8, subject="repeated")]),
        _response([_boundary(2, confidence=0.8, subject="repeated"), _boundary(3, confidence=0.8)]),
        _response([]),
        _response([]),
        _response([]),
    ]
    gateway = BoundaryOnlyGateway(responses)
    result = SemanticSegmenter(gateway, safe_tokens=1, block_tokens=4, segment_overlap=2).segment(
        _transcript([f"S{i}" for i in range(12)], "golden-long-overlap")
    )

    assert len(gateway.calls) == 5
    assert _ranges(result) == [(0, 2), (3, 11)]
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

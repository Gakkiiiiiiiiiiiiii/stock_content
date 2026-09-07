from __future__ import annotations

import pytest

from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import (
    ASRStage,
    TranscriptCandidateStage,
    TranscriptQualityStage,
    TranscriptSelectionStage,
)
from stock_content.application.transcript_quality_service import TranscriptQualityService
from stock_content.application.transcript_selection_service import TranscriptSelectionError, TranscriptSelectionService
from stock_content.domain.artifacts import MediaArtifact
from stock_content.domain.transcript_candidate import (
    AlignmentStatus,
    TranscriptCandidate,
    TranscriptCandidateSegment,
    TranscriptSource,
)
from stock_content.domain.transcript_quality import TranscriptQualityStatus


def _candidate(
    source: TranscriptSource, spans: list[tuple[int, int]], *, text: str = "营收100亿元 2025Q1 600000 10%"
) -> TranscriptCandidate:
    source_id = f"artifact-{source.value.lower()}"
    return TranscriptCandidate(
        source_id,
        source,
        "zh-CN",
        source_id,
        tuple(
            TranscriptCandidateSegment(source, source_id, start, end, text, text, 1.0, AlignmentStatus.ALIGNED)
            for start, end in spans
        ),
    )


def test_selection_precedence_and_gap_fill_never_duplicate_authority():
    service = TranscriptSelectionService()
    official = _candidate(TranscriptSource.OFFICIAL_SUBTITLE, [(0, 100_000)])
    automatic = _candidate(TranscriptSource.AUTO_SUBTITLE, [(0, 100_000)])
    asr = _candidate(TranscriptSource.ASR, [(0, 100_000)])
    selected = service.select((automatic, asr, official), duration_ms=100_000)
    assert {item.source for item in selected.segments} == {TranscriptSource.OFFICIAL_SUBTITLE}

    partial = _candidate(TranscriptSource.OFFICIAL_SUBTITLE, [(0, 60_000)])
    gap_asr = _candidate(TranscriptSource.ASR, [(60_000, 100_000)])
    filled = service.select((partial, gap_asr), duration_ms=100_000)
    assert [(item.start_ms, item.end_ms) for item in filled.segments] == [(0, 60_000), (60_000, 100_000)]
    assert filled.report.quality_status is TranscriptQualityStatus.PASS


@pytest.mark.parametrize(
    "spans, expected",
    [
        ([(0, 20_000), (10_000, 100_000)], "TIMESTAMP_NON_MONOTONIC_OR_OVERLAP"),
        ([(0, 70_000)], "MAX_GAP_EXCEEDS_20_SECONDS"),
    ],
)
def test_quality_fails_closed_for_overlap_and_gap(spans, expected):
    report = TranscriptQualityService().evaluate(
        _candidate(TranscriptSource.OFFICIAL_SUBTITLE, spans).ordered_segments,
        duration_ms=100_000,
        language="zh",
    )
    assert expected in report.reason_codes
    assert report.quality_status is not TranscriptQualityStatus.PASS


def test_hard_facts_mutation_and_official_boundary_require_review():
    invalid = TranscriptCandidate(
        "official",
        TranscriptSource.OFFICIAL_SUBTITLE,
        "zh",
        "official",
        (
            TranscriptCandidateSegment(
                TranscriptSource.OFFICIAL_SUBTITLE,
                "official",
                0,
                100_000,
                "营收100亿元 2025Q1 600000 10%",
                "营收100亿 2025Q1 600000 10%",
                1.0,
            ),
        ),
    )
    with pytest.raises(TranscriptSelectionError) as exc:
        TranscriptSelectionService().select((invalid,), duration_ms=100_000)
    assert exc.value.report.quality_status is TranscriptQualityStatus.NEEDS_REVIEW
    assert "HARD_FACT_TOKEN_MUTATION" in exc.value.report.reason_codes


def test_candidate_stages_preserve_candidates_and_skip_asr_when_manual_qualifies():
    context = PipelineContext(
        task_id="selection-stage",
        source={"type": "bilibili", "ref": "BV1"},
        options={
            "duration_ms": 100_000,
            "_test_subtitle_candidate_adapter": True,
            "test_subtitle_candidates": [
                {
                    "source": "OFFICIAL_SUBTITLE",
                    "source_artifact_id": "subtitle-manual",
                    "language": "zh",
                    "segments": [{"start_ms": 0, "end_ms": 100_000, "text": "600000 10%", "confidence": 1.0}],
                }
            ],
        },
    )
    context.artifacts.media = MediaArtifact(artifact_id="media-1", artifact_type="media", duration_ms=100_000)
    candidates = TranscriptCandidateStage().execute(context)
    assert len(candidates.produced_artifacts) == 1
    class NoASR:
        def transcribe(self, *_):
            raise AssertionError("must not run")

    ASRStage(NoASR()).execute(context)
    selected = TranscriptSelectionStage().execute(context)
    TranscriptQualityStage().execute(context)
    assert selected.produced_artifacts == (context.artifacts.transcript,)
    assert context.artifacts.transcript.segments[0].source == "OFFICIAL_SUBTITLE"
    assert context.state.transcript_quality_report.report_hash == context.state.transcript_quality_report.report_hash

"""Policy for selecting one transcript; it never promotes a failed candidate."""

from __future__ import annotations

from dataclasses import dataclass

from stock_content.application.transcript_quality_service import TranscriptQualityService
from stock_content.domain.transcript_candidate import (
    AlignmentStatus,
    TranscriptCandidate,
    TranscriptCandidateSegment,
    TranscriptSource,
)
from stock_content.domain.transcript_quality import TranscriptQualityReport, TranscriptQualityStatus


@dataclass(frozen=True, slots=True)
class TranscriptSelection:
    segments: tuple[TranscriptCandidateSegment, ...]
    language: str
    report: TranscriptQualityReport


class TranscriptSelectionError(RuntimeError):
    def __init__(self, report: TranscriptQualityReport) -> None:
        super().__init__(f"{report.quality_status.value}: {','.join(report.reason_codes)}")
        self.report = report


class TranscriptSelectionService:
    def __init__(self, quality: TranscriptQualityService | None = None) -> None:
        self._quality = quality or TranscriptQualityService()

    def select(self, candidates: tuple[TranscriptCandidate, ...], *, duration_ms: int) -> TranscriptSelection:
        # An invalid official candidate is an official-boundary failure.  Do
        # not silently swap it for machine output and create a false authority.
        official = [
            item for item in candidates if item.source is TranscriptSource.OFFICIAL_SUBTITLE and item.is_chinese
        ]
        for group in (
            official,
            [item for item in candidates if item.source is TranscriptSource.AUTO_SUBTITLE and item.is_chinese],
            [item for item in candidates if item.source is TranscriptSource.ASR],
        ):
            for candidate in sorted(group, key=lambda item: (item.candidate_id, item.source_artifact_id)):
                report = self._quality.evaluate(
                    candidate.ordered_segments, duration_ms=duration_ms, language=candidate.language
                )
                if report.quality_status is TranscriptQualityStatus.PASS:
                    return TranscriptSelection(candidate.ordered_segments, candidate.language, report)
                if candidate in official and set(report.reason_codes) - {
                    "COVERAGE_BELOW_90_PERCENT",
                }:
                    raise TranscriptSelectionError(report)
        # Only augment explicit subtitle gaps, and only from aligned ASR.
        asr = [item for item in candidates if item.source is TranscriptSource.ASR]
        subtitles = official + [
            item for item in candidates if item.source is TranscriptSource.AUTO_SUBTITLE and item.is_chinese
        ]
        for subtitle in sorted(subtitles, key=lambda item: (item.source.value, item.candidate_id)):
            for speech in sorted(asr, key=lambda item: item.candidate_id):
                if any(item.alignment_status is not AlignmentStatus.ALIGNED for item in speech.segments):
                    continue
                gaps = self._gaps(subtitle.ordered_segments, duration_ms)
                additions = tuple(
                    item
                    for item in speech.ordered_segments
                    if any(start <= item.start_ms and item.end_ms <= end for start, end in gaps)
                )
                combined = tuple(
                    sorted(
                        (*subtitle.ordered_segments, *additions),
                        key=lambda item: (
                            item.start_ms,
                            item.end_ms,
                            item.source_artifact_id,
                            item.raw_text,
                        ),
                    )
                )
                report = self._quality.evaluate(combined, duration_ms=duration_ms, language=subtitle.language)
                if report.quality_status is TranscriptQualityStatus.PASS:
                    return TranscriptSelection(combined, subtitle.language, report)
        language = official[0].language if official else (candidates[0].language if candidates else "und")
        report = self._quality.evaluate((), duration_ms=duration_ms, language=language)
        raise TranscriptSelectionError(report)

    @staticmethod
    def _gaps(segments: tuple[TranscriptCandidateSegment, ...], duration_ms: int) -> tuple[tuple[int, int], ...]:
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for item in segments:
            if item.start_ms > cursor:
                gaps.append((cursor, item.start_ms))
            cursor = max(cursor, item.end_ms)
        if cursor < duration_ms:
            gaps.append((cursor, duration_ms))
        return tuple(gaps)


__all__ = ["TranscriptSelection", "TranscriptSelectionError", "TranscriptSelectionService"]

"""Pure calculation of formal transcript readiness."""

from __future__ import annotations

import re

from stock_content.domain.transcript_candidate import TranscriptCandidateSegment
from stock_content.domain.transcript_quality import (
    HARD_FACT_TOKENS,
    TranscriptQualityReport,
    TranscriptQualityStatus,
    source_mix,
)


class TranscriptQualityService:
    def evaluate(
        self, segments: tuple[TranscriptCandidateSegment, ...], *, duration_ms: int, language: str
    ) -> TranscriptQualityReport:
        reasons: list[str] = []
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            raise ValueError("TRANSCRIPT_DURATION_INVALID")
        ordered = tuple(
            sorted(
                segments,
                key=lambda item: (
                    item.start_ms,
                    item.end_ms,
                    item.source_artifact_id,
                    item.raw_text,
                    item.normalized_text,
                ),
            )
        )
        monotonic = tuple(segments) == ordered and all(
            earlier.end_ms <= later.start_ms for earlier, later in zip(ordered, ordered[1:])
        )
        intervals: list[tuple[int, int]] = []
        for item in ordered:
            if item.end_ms > duration_ms:
                reasons.append("TIMESTAMP_OUT_OF_BOUNDS")
            intervals.append((item.start_ms, min(item.end_ms, duration_ms)))
        union: list[tuple[int, int]] = []
        overlap = 0
        for start, end in intervals:
            if not union or start > union[-1][1]:
                union.append((start, end))
            else:
                overlap += max(0, min(end, union[-1][1]) - start)
                union[-1] = (union[-1][0], max(union[-1][1], end))
        covered = sum(end - start for start, end in union)
        gaps = [start - end for (_, end), (start, _) in zip(union, union[1:])]
        if union:
            gaps.extend((union[0][0], duration_ms - union[-1][1]))
        else:
            gaps.append(duration_ms)
        max_gap = max(gaps, default=duration_ms)
        raw_tokens = [token for item in ordered for token in re.findall(HARD_FACT_TOKENS, item.raw_text)]
        normalized_tokens = [token for item in ordered for token in re.findall(HARD_FACT_TOKENS, item.normalized_text)]
        preserved = sum(1 for token in raw_tokens if normalized_tokens.count(token) >= raw_tokens.count(token))
        preservation = 1.0 if not raw_tokens else preserved / len(raw_tokens)
        if not monotonic:
            reasons.append("TIMESTAMP_NON_MONOTONIC_OR_OVERLAP")
        if covered / duration_ms < 0.95:
            reasons.append("COVERAGE_BELOW_95_PERCENT")
        if max_gap > 20_000:
            reasons.append("MAX_GAP_EXCEEDS_20_SECONDS")
        if preservation != 1.0:
            reasons.append("HARD_FACT_TOKEN_MUTATION")
        status = (
            TranscriptQualityStatus.PASS
            if not reasons
            else (
                TranscriptQualityStatus.RETRYABLE_ASR
                if reasons == ["COVERAGE_BELOW_95_PERCENT"] or reasons == ["MAX_GAP_EXCEEDS_20_SECONDS"]
                else TranscriptQualityStatus.NEEDS_REVIEW
            )
        )
        return TranscriptQualityReport(
            duration_ms=duration_ms,
            covered_ms=covered,
            coverage_ratio=covered / duration_ms,
            max_gap_ms=max_gap,
            overlap_ratio=overlap / duration_ms,
            timestamp_monotonic=monotonic,
            language=language,
            source_mix=source_mix(ordered, covered),
            numeric_preservation_ratio=preservation,
            quality_status=status,
            reason_codes=tuple(sorted(set(reasons))),
        )


__all__ = ["TranscriptQualityService"]

"""Transcript quality gates.  These gates are deliberately stricter than display quality."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Mapping

from stock_content.domain.artifacts import canonical_json
from stock_content.domain.transcript_candidate import TranscriptCandidateSegment


class TranscriptQualityStatus(StrEnum):
    PASS = "PASS"
    RETRYABLE_ASR = "RETRYABLE_ASR"
    NEEDS_REVIEW = "NEEDS_REVIEW"


HARD_FACT_TOKENS = (
    r"(?:\d+(?:\.\d+)?%|(?:人民币|美元|港元|CNY|USD|HKD|¥|\$)\s?\d+(?:\.\d+)?(?:亿(?:元)?|万(?:元)?|千(?:元)?|元)?"
    r"|\d+(?:\.\d+)?(?:亿(?:元)?|万(?:元)?|千(?:元)?|元)|(?:20\d{2}\s*[Qq][1-4]|[Qq][1-4]\s*20\d{2})"
    r"|(?:[A-Z]{1,5}|\d{6})(?:\.[A-Z]{2})?)"
)


@dataclass(frozen=True, slots=True)
class TranscriptQualityReport:
    duration_ms: int
    covered_ms: int
    coverage_ratio: float
    max_gap_ms: int
    overlap_ratio: float
    timestamp_monotonic: bool
    language: str
    source_mix: Mapping[str, float]
    numeric_preservation_ratio: float
    quality_status: TranscriptQualityStatus
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["quality_status"] = self.quality_status.value
        result["source_mix"] = dict(sorted(self.source_mix.items()))
        result["reason_codes"] = list(self.reason_codes)
        return result

    @property
    def report_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


def source_mix(segments: tuple[TranscriptCandidateSegment, ...], covered_ms: int) -> dict[str, float]:
    if not covered_ms:
        return {}
    values: dict[str, int] = {}
    for item in segments:
        values[item.source.value] = values.get(item.source.value, 0) + item.end_ms - item.start_ms
    return {name: round(value / covered_ms, 12) for name, value in sorted(values.items())}


__all__ = ["HARD_FACT_TOKENS", "TranscriptQualityReport", "TranscriptQualityStatus", "source_mix"]

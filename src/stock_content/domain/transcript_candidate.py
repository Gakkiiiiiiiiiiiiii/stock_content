"""Deterministic, auditable transcript candidate inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class TranscriptSource(StrEnum):
    OFFICIAL_SUBTITLE = "OFFICIAL_SUBTITLE"
    AUTO_SUBTITLE = "AUTO_SUBTITLE"
    ASR = "ASR"


class AlignmentStatus(StrEnum):
    ALIGNED = "ALIGNED"
    UNALIGNED = "UNALIGNED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TranscriptCandidateSegment:
    source: TranscriptSource
    source_artifact_id: str
    start_ms: int
    end_ms: int
    raw_text: str
    normalized_text: str
    confidence: float
    alignment_status: AlignmentStatus = AlignmentStatus.ALIGNED

    def __post_init__(self) -> None:
        if not self.source_artifact_id:
            raise ValueError("transcript segment requires source_artifact_id")
        if not isinstance(self.start_ms, int) or not isinstance(self.end_ms, int):
            raise ValueError("transcript timestamps must be integer milliseconds")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("transcript segment has invalid duration")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("transcript confidence must be finite and within [0, 1]")
        if not self.raw_text.strip() or not self.normalized_text.strip():
            raise ValueError("transcript segment text must not be empty")
        forbidden = ("signature=", "token=", "credential=", "cookie=")
        if any(marker in self.raw_text.lower() or marker in self.normalized_text.lower() for marker in forbidden):
            raise ValueError("transcript text must not contain credentials or signed locators")


@dataclass(frozen=True, slots=True)
class TranscriptCandidate:
    candidate_id: str
    source: TranscriptSource
    language: str
    source_artifact_id: str
    segments: tuple[TranscriptCandidateSegment, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source_artifact_id:
            raise ValueError("transcript candidate requires stable ids")
        if not self.language:
            raise ValueError("transcript candidate requires language")

    @property
    def is_chinese(self) -> bool:
        return self.language.lower().replace("_", "-").startswith("zh")

    @property
    def ordered_segments(self) -> tuple[TranscriptCandidateSegment, ...]:
        return tuple(
            sorted(
                self.segments,
                key=lambda item: (
                    item.start_ms,
                    item.end_ms,
                    item.source_artifact_id,
                    item.raw_text,
                    item.normalized_text,
                ),
            )
        )


def as_segments(values: Iterable[TranscriptCandidateSegment]) -> tuple[TranscriptCandidateSegment, ...]:
    return tuple(values)


__all__ = [
    "AlignmentStatus",
    "TranscriptCandidate",
    "TranscriptCandidateSegment",
    "TranscriptSource",
    "as_segments",
]

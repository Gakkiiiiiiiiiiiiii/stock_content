"""Fail-closed validation of Stage 1 boundary points."""

from __future__ import annotations

from typing import Any, Iterable

from .semantic_segment import SemanticBoundary


def validate_boundaries(boundaries: Iterable[SemanticBoundary | dict[str, Any]], segment_count: int) -> list[int]:
    values: list[int] = []
    for raw in boundaries:
        value = raw.after_segment_index if isinstance(raw, SemanticBoundary) else raw.get("after_segment_index")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("boundary after_segment_index must be an int")
        if value < 0 or value >= segment_count - 1:
            raise ValueError("boundary must satisfy 0 <= boundary < N-1")
        if values and value <= values[-1]:
            raise ValueError("boundaries must be strictly ascending and unique")
        values.append(value)
    return values


def validate_full_coverage(segments: list[Any], segment_count: int) -> None:
    if not segments and segment_count:
        raise ValueError("semantic segments must cover the transcript")
    expected = 0
    for segment in segments:
        if segment.start_segment_index != expected or segment.end_segment_index < segment.start_segment_index:
            raise ValueError("semantic segments must be contiguous and gap-free")
        expected = segment.end_segment_index + 1
    if expected != segment_count:
        raise ValueError("semantic segments must cover every transcript segment")


validate_semantic_boundaries = validate_boundaries


class SemanticBoundaryValidator:
    """Object-form adapter used by stage orchestration and integrations."""

    @staticmethod
    def validate(boundaries, segment_count: int) -> list[int]:
        return validate_boundaries(boundaries, segment_count)

    @staticmethod
    def validate_full_coverage(segments, segment_count: int) -> None:
        validate_full_coverage(segments, segment_count)


__all__ = [
    "validate_boundaries",
    "validate_semantic_boundaries",
    "validate_full_coverage",
    "SemanticBoundary",
    "SemanticBoundaryValidator",
]

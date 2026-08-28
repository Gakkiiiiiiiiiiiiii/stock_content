"""Deterministic semantic segmentation over immutable transcript coordinates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from .artifacts import SemanticSegmentArtifact, TranscriptArtifact, canonical_json

SEMANTIC_SEGMENT_SCHEMA_VERSION = "semantic-segment.v1"


def semantic_segment_id(
    transcript_artifact_id: str,
    start_segment_id: str,
    end_segment_id: str,
    schema_version: str = SEMANTIC_SEGMENT_SCHEMA_VERSION,
) -> str:
    if not transcript_artifact_id or not start_segment_id or not end_segment_id:
        raise ValueError("transcript and boundary segment ids are required")
    payload = {
        "transcript_artifact_id": transcript_artifact_id,
        "start_segment_id": start_segment_id,
        "end_segment_id": end_segment_id,
        "schema_version": schema_version,
    }
    # The database contract allocates 64 characters. Preserve the frozen
    # ``semseg_`` identity prefix and retain 228 bits of the content hash.
    return "semseg_" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:57]


@dataclass(frozen=True)
class SemanticBoundary:
    after_segment_index: int
    boundary_type: str = "TOPIC_CHANGE"
    next_topic: str | None = None
    next_subject: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class SemanticSegment:
    semantic_segment_id: str
    transcript_artifact_id: str
    segment_index: int
    start_segment_index: int
    end_segment_index: int
    start_segment_id: str
    end_segment_id: str
    start_ms: int
    end_ms: int
    topic: str | None = None
    subject: str | None = None
    segment_type: str = "ANALYSIS"
    model_id: str = ""
    prompt_version: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class SemanticSegmentItem:
    semantic_segment_id: str
    segment_index: int
    start_segment_index: int
    end_segment_index: int
    start_segment_id: str
    end_segment_id: str
    start_ms: int
    end_ms: int
    topic: str | None = None
    subject: str | None = None
    segment_type: str = "ANALYSIS"
    confidence: float | None = None


def materialize_semantic_segments(
    transcript: TranscriptArtifact,
    boundaries: Iterable[SemanticBoundary | dict[str, Any]],
    *,
    model_id: str = "",
    prompt_version: str = "",
    schema_version: str = SEMANTIC_SEGMENT_SCHEMA_VERSION,
) -> list[SemanticSegment]:
    """Turn validated boundary points into contiguous, gap-free segments."""
    from .semantic_boundary_validator import validate_boundaries

    items = list(transcript.segments)
    boundary_items = [
        point if isinstance(point, SemanticBoundary) else SemanticBoundary(**point) for point in boundaries
    ]
    points = validate_boundaries(boundary_items, len(items))
    if not items:
        return []
    starts = [0, *[point + 1 for point in points]]
    ends = [*points, len(items) - 1]
    output: list[SemanticSegment] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        first, last = items[start], items[end]
        output.append(
            SemanticSegment(
                semantic_segment_id=semantic_segment_id(
                    transcript.artifact_id, first.segment_id, last.segment_id, schema_version
                ),
                transcript_artifact_id=transcript.artifact_id,
                segment_index=index,
                start_segment_index=start,
                end_segment_index=end,
                start_segment_id=first.segment_id,
                end_segment_id=last.segment_id,
                start_ms=first.start_ms,
                end_ms=last.end_ms,
                model_id=model_id,
                prompt_version=prompt_version,
                # Boundary metadata describes the segment after the boundary.
                topic=(boundary_items[index - 1].next_topic if index > 0 else None),
                subject=(boundary_items[index - 1].next_subject if index > 0 else None),
                confidence=(boundary_items[index - 1].confidence if index > 0 else None),
            )
        )
    return output


def build_semantic_segment_artifact(
    transcript: TranscriptArtifact,
    boundaries: Iterable[SemanticBoundary | dict[str, Any]],
    *,
    artifact_id: str = "semantic-segments-pending",
    model_id: str = "",
    prompt_version: str = "",
    schema_version: str = SEMANTIC_SEGMENT_SCHEMA_VERSION,
) -> SemanticSegmentArtifact:
    segments = materialize_semantic_segments(
        transcript,
        boundaries,
        model_id=model_id,
        prompt_version=prompt_version,
        schema_version=schema_version,
    )
    return SemanticSegmentArtifact(
        artifact_id=artifact_id,
        artifact_type="semantic_segments",
        transcript_artifact_id=transcript.artifact_id,
        segments=[
            SemanticSegmentItem(
                semantic_segment_id=item.semantic_segment_id,
                segment_index=item.segment_index,
                start_segment_index=item.start_segment_index,
                end_segment_index=item.end_segment_index,
                start_segment_id=item.start_segment_id,
                end_segment_id=item.end_segment_id,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                topic=item.topic,
                subject=item.subject,
                segment_type=item.segment_type,
                confidence=item.confidence,
            )
            for item in segments
        ],
        model_id=model_id,
        prompt_version=prompt_version,
        segmentation_schema_version=schema_version,
    )


semantic_segment_id_of = semantic_segment_id
materialize_full_coverage = materialize_semantic_segments


__all__ = [
    "SEMANTIC_SEGMENT_SCHEMA_VERSION",
    "SemanticBoundary",
    "SemanticSegment",
    "SemanticSegmentItem",
    "SemanticSegmentArtifact",
    "semantic_segment_id",
    "materialize_semantic_segments",
    "semantic_segment_id_of",
    "materialize_full_coverage",
    "build_semantic_segment_artifact",
]

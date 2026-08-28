"""Build complete multimodal context; semantic segments remain authoritative."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .artifacts import FrameArtifact, OCRArtifact, TranscriptArtifact, VisionArtifact
from .semantic_segment import SemanticSegment


@dataclass
class SemanticContext:
    semantic_segment_id: str
    start_ms: int
    end_ms: int
    transcript_segments: list[dict[str, Any]] = field(default_factory=list)
    transcript_text: str = ""
    ocr_evidence: list[dict[str, Any]] = field(default_factory=list)
    vision_evidence: list[dict[str, Any]] = field(default_factory=list)
    frame_refs: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)


class SemanticContextBuilder:
    def __init__(self, padding_ms: int = 4000):
        if padding_ms < 0:
            raise ValueError("padding_ms must be non-negative")
        self.padding_ms = padding_ms

    def build(
        self,
        semantic_segment: SemanticSegment,
        transcript: TranscriptArtifact,
        frames: Iterable[FrameArtifact] = (),
        ocr: Iterable[OCRArtifact] = (),
        vision: Iterable[VisionArtifact] = (),
        temporal_windows: Iterable[Any] = (),
    ) -> SemanticContext:
        # temporal_windows is accepted for adapter compatibility only and is
        # intentionally never consulted for semantic bounds.
        selected = [
            item
            for item in transcript.segments
            if semantic_segment.start_segment_index <= item.segment_index <= semantic_segment.end_segment_index
        ]
        low, high = semantic_segment.start_ms - self.padding_ms, semantic_segment.end_ms + self.padding_ms
        frame_list = [item for item in frames if low <= item.timestamp_ms <= high]
        frame_ids = {item.frame_id for item in frame_list}
        ocr_list = [item for item in ocr if item.frame_artifact_id in frame_ids]
        vision_list = [item for item in vision if item.frame_artifact_id in frame_ids]
        return SemanticContext(
            semantic_segment_id=semantic_segment.semantic_segment_id,
            start_ms=semantic_segment.start_ms,
            end_ms=semantic_segment.end_ms,
            transcript_segments=[item.__dict__.copy() for item in selected],
            transcript_text=" ".join(item.text for item in selected),
            ocr_evidence=[item.__dict__.copy() for item in ocr_list],
            vision_evidence=[item.__dict__.copy() for item in vision_list],
            frame_refs=[item.artifact_id for item in frame_list],
            speaker_ids=sorted({item.speaker_id for item in selected if item.speaker_id}),
        )


build_semantic_context = SemanticContextBuilder().build


__all__ = ["SemanticContext", "SemanticContextBuilder", "build_semantic_context"]

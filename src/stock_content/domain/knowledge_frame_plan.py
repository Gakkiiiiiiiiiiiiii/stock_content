"""Deterministic targeted-frame requests derived from evidence windows.

The transcript remains the primary evidence source.  This module only turns
already-planned, transcript-anchored windows into bounded visual corroboration
requests; it deliberately has no ffmpeg, OCR, or model dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .artifacts import canonical_json
from .knowledge_evidence_window import (
    KNOWLEDGE_EVIDENCE_WINDOW_PLANNER_VERSION,
    KnowledgeEvidenceWindow,
)

KNOWLEDGE_FRAME_PLANNER_VERSION = "knowledge-frame-plan.v1"

KNOWLEDGE_CENTER = "KNOWLEDGE_CENTER"
KNOWLEDGE_NEARBY = "KNOWLEDGE_NEARBY"
HIGH_SIGNAL = "HIGH_SIGNAL"


@dataclass(frozen=True, slots=True)
class KnowledgeFrameRequest:
    """A replay-stable request for exactly one visual timestamp."""

    timestamp_ms: int
    extraction_reason: str
    semantic_segment_ids: tuple[str, ...]
    evidence_window_ids: tuple[str, ...]
    planner_version: str = KNOWLEDGE_FRAME_PLANNER_VERSION


def evidence_window_id(window: KnowledgeEvidenceWindow) -> str:
    """Identify the immutable transcript-derived request, without source text."""
    payload = {
        "semantic_segment_id": window.semantic_segment_id,
        "start_ms": window.start_ms,
        "end_ms": window.end_ms,
        "center_ms": window.center_ms,
        "transcript_segment_ids": window.transcript_segment_ids,
        "high_signals": [
            (item.kind, item.reason, item.transcript_segment_ids) for item in window.high_signals
        ],
        "planner_version": window.planner_version,
    }
    return "kew_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:57]


class KnowledgeFramePlanner:
    """Choose compact centre/nearby samples, with denser high-signal coverage."""

    def __init__(
        self,
        *,
        nearby_offsets_ms: tuple[int, ...] = (-1_000, 1_000),
        high_signal_offsets_ms: tuple[int, ...] = (-3_000, -2_000, -1_000, 1_000, 2_000, 3_000),
        max_frames_per_window: int = 7,
        max_frames_per_media: int = 120,
        planner_version: str = KNOWLEDGE_FRAME_PLANNER_VERSION,
    ) -> None:
        if not nearby_offsets_ms or any(abs(value) not in {1_000, 2_000, 3_000} for value in nearby_offsets_ms):
            raise ValueError("nearby offsets must be non-empty +/-1..3 second values")
        if any(abs(value) not in {1_000, 2_000, 3_000} for value in high_signal_offsets_ms):
            raise ValueError("high-signal offsets must be +/-1..3 second values")
        if max_frames_per_window < 1 or max_frames_per_media < 1:
            raise ValueError("frame caps must be positive")
        if not planner_version:
            raise ValueError("planner_version is required")
        self._nearby_offsets_ms = tuple(sorted(set(nearby_offsets_ms)))
        self._high_signal_offsets_ms = tuple(sorted(set(high_signal_offsets_ms)))
        self._max_frames_per_window = max_frames_per_window
        self._max_frames_per_media = max_frames_per_media
        self._planner_version = planner_version

    def plan(
        self, windows: Iterable[KnowledgeEvidenceWindow], *, media_duration_ms: int
    ) -> tuple[KnowledgeFrameRequest, ...]:
        if media_duration_ms < 0:
            raise ValueError("media_duration_ms must be non-negative")
        candidates: list[tuple[int, int, str, str, str]] = []
        # priority preserves the centre if a cap or timestamp collision occurs.
        priority = {KNOWLEDGE_CENTER: 0, HIGH_SIGNAL: 1, KNOWLEDGE_NEARBY: 2}
        ordered = sorted(windows, key=lambda item: (item.start_ms, item.end_ms, item.semantic_segment_id))
        for window in ordered:
            window_id = evidence_window_id(window)
            offsets = self._high_signal_offsets_ms if window.high_signals else self._nearby_offsets_ms
            local = [(window.center_ms, KNOWLEDGE_CENTER)] + [
                (window.center_ms + offset, HIGH_SIGNAL if window.high_signals else KNOWLEDGE_NEARBY)
                for offset in offsets
            ]
            # Normalise a boundary-clamped window before applying its local cap.
            local = sorted(
                {(min(max(0, timestamp), media_duration_ms), reason) for timestamp, reason in local},
                key=lambda item: (priority[item[1]], item[0], item[1]),
            )[: self._max_frames_per_window]
            candidates.extend(
                (timestamp, priority[reason], reason, window.semantic_segment_id, window_id)
                for timestamp, reason in local
            )

        merged: dict[int, list[tuple[int, str, str, str]]] = {}
        for timestamp, item_priority, reason, semantic_id, window_id in candidates:
            merged.setdefault(timestamp, []).append((item_priority, reason, semantic_id, window_id))
        output: list[KnowledgeFrameRequest] = []
        for timestamp in sorted(merged):
            values = sorted(merged[timestamp])
            best_priority, best_reason, _, _ = values[0]
            del best_priority
            output.append(
                KnowledgeFrameRequest(
                    timestamp_ms=timestamp,
                    extraction_reason=best_reason,
                    semantic_segment_ids=tuple(sorted({item[2] for item in values})),
                    evidence_window_ids=tuple(sorted({item[3] for item in values})),
                    planner_version=self._planner_version,
                )
            )
        # Preserve every centre before lower-priority nearby samples when a
        # global cap is reached, then restore chronological extraction order.
        selected = sorted(
            output,
            key=lambda item: (priority[item.extraction_reason], item.timestamp_ms, item.semantic_segment_ids),
        )[: self._max_frames_per_media]
        return tuple(sorted(selected, key=lambda item: (item.timestamp_ms, item.extraction_reason)))


def frame_id_for(
    *, media_artifact_id: str, request: KnowledgeFrameRequest
) -> str:
    """Bind a frame identity to immutable media, plan version, timestamp and reason."""
    if not media_artifact_id:
        raise ValueError("media_artifact_id is required")
    payload = {
        "media_artifact_id": media_artifact_id,
        "timestamp_ms": request.timestamp_ms,
        "extraction_reason": request.extraction_reason,
        "semantic_segment_ids": request.semantic_segment_ids,
        "evidence_window_ids": request.evidence_window_ids,
        "planner_version": request.planner_version,
        "evidence_window_planner_version": KNOWLEDGE_EVIDENCE_WINDOW_PLANNER_VERSION,
    }
    return "frame_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:58]


def request_id_for(request: KnowledgeFrameRequest) -> str:
    """Stable, non-secret identity for a complete targeted-frame request."""
    payload = {
        "timestamp_ms": request.timestamp_ms,
        "extraction_reason": request.extraction_reason,
        "semantic_segment_ids": request.semantic_segment_ids,
        "evidence_window_ids": request.evidence_window_ids,
        "planner_version": request.planner_version,
    }
    return "kfr_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:58]


__all__ = [
    "HIGH_SIGNAL",
    "KNOWLEDGE_CENTER",
    "KNOWLEDGE_FRAME_PLANNER_VERSION",
    "KNOWLEDGE_NEARBY",
    "KnowledgeFramePlanner",
    "KnowledgeFrameRequest",
    "evidence_window_id",
    "frame_id_for",
    "request_id_for",
]

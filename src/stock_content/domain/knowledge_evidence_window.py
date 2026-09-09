"""Deterministic evidence-window plans for knowledge-directed frame sampling.

The planner deliberately has no media, OCR, or model dependency.  Transcript
coordinates remain the primary evidence authority; a later frame stage may
only use these plans to *add* visual corroboration.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .artifacts import TranscriptArtifact, canonical_json
from .semantic_segment import SemanticSegment

KNOWLEDGE_EVIDENCE_WINDOW_PLANNER_VERSION = "knowledge-evidence-window.v1"


@dataclass(frozen=True, slots=True)
class HighSignal:
    """A deterministic cue that merits denser visual sampling.

    ``reason`` is deliberately a fixed classifier label, never a transcript
    excerpt.  This keeps the plan auditable without copying source text,
    credentials, or storage locations into a downstream frame request.
    """

    kind: str
    reason: str
    transcript_segment_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeEvidenceWindow:
    """One bounded, replay-stable visual-evidence request per semantic unit."""

    semantic_segment_id: str
    start_ms: int
    end_ms: int
    center_ms: int
    transcript_segment_ids: tuple[str, ...]
    high_signals: tuple[HighSignal, ...]
    planner_version: str = KNOWLEDGE_EVIDENCE_WINDOW_PLANNER_VERSION
    # A semantic segment can contain several unrelated atomic propositions.
    # This opaque, deterministic identity keeps their visual plans distinct
    # without putting source text into a frame request.
    knowledge_identity: str = ""


_SIGNAL_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("NUMERIC", "numeric_value", re.compile(r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9])")),
    (
        "UNIT",
        "numeric_unit",
        re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|亿元|万亿元|万元|万|亿|倍|个|台|吨|美元|人民币|USD|CNY)"),
    ),
    (
        "DATE",
        "date_expression",
        re.compile(r"(?:19|20)\d{2}(?:年|[-/.])(?:0?[1-9]|1[0-2])(?:月|[-/.])?(?:0?[1-9]|[12]\d|3[01])?"),
    ),
    ("TICKER", "six_digit_ticker", re.compile(r"(?<!\d)\d{6}(?!\d)")),
    (
        "COMPANY",
        "company_name_pattern",
        re.compile(r"[\u4e00-\u9fffA-Za-z]{2,16}(?:股份有限公司|有限公司|集团|银行|证券|科技|控股)"),
    ),
    ("PRODUCT", "product_or_model_cue", re.compile(r"(?:产品|型号|芯片|GPU|CPU|服务器|软件|平台|版本)")),
    ("INDUSTRY", "industry_or_sector_cue", re.compile(r"(?:行业|板块|赛道|产业链|金融|算力|半导体|银行业)")),
    ("CHART", "chart_or_price_action_cue", re.compile(r"(?:K线|走势图|曲线|图表|成交量|均线|支撑位|压力位)")),
    ("POLICY_DOCUMENT", "policy_document_cue", re.compile(r"(?:政策|文件|通知|公告|规划|条例|意见|增资|降准)")),
)


class KnowledgeEvidenceWindowPlanner:
    """Plan transcript-anchored windows for a completed semantic segmentation."""

    def __init__(self, *, padding_ms: int = 1500, planner_version: str = KNOWLEDGE_EVIDENCE_WINDOW_PLANNER_VERSION):
        if padding_ms < 0:
            raise ValueError("padding_ms must be non-negative")
        if not planner_version:
            raise ValueError("planner_version is required")
        self._padding_ms = padding_ms
        self._planner_version = planner_version

    def plan(
        self,
        transcript: TranscriptArtifact,
        semantic_segments: Iterable[SemanticSegment],
        *,
        media_duration_ms: int | None = None,
    ) -> tuple[KnowledgeEvidenceWindow, ...]:
        """Return one monotonic, de-duplicated plan per semantic identity.

        The duration is explicit when the media artifact is available.  A
        transcript-only caller gets the maximum authoritative transcript end
        as the safe upper bound, rather than an unbounded frame timestamp.
        """
        if media_duration_ms is not None and media_duration_ms < 0:
            raise ValueError("media_duration_ms must be non-negative")
        duration_ms = media_duration_ms
        if duration_ms is None:
            duration_ms = max((item.end_ms for item in transcript.segments), default=0)

        transcript_by_index = {item.segment_index: item for item in transcript.segments}
        # Sorting before identity de-duplication means replay is stable even
        # when an adapter returns an unordered collection.
        ordered = sorted(
            semantic_segments,
            key=lambda item: (
                item.start_ms,
                item.end_ms,
                item.start_segment_index,
                item.end_segment_index,
                item.semantic_segment_id,
            ),
        )
        output: list[KnowledgeEvidenceWindow] = []
        seen_ids: set[str] = set()
        for semantic in ordered:
            if semantic.semantic_segment_id in seen_ids:
                continue
            seen_ids.add(semantic.semantic_segment_id)
            selected = tuple(
                transcript_by_index[index]
                for index in range(semantic.start_segment_index, semantic.end_segment_index + 1)
                if index in transcript_by_index
            )
            spoken_start = self._clamp(semantic.start_ms, duration_ms)
            spoken_end = self._clamp(semantic.end_ms, duration_ms)
            if spoken_end < spoken_start:
                spoken_end = spoken_start
            start_ms = self._clamp(spoken_start - self._padding_ms, duration_ms)
            end_ms = self._clamp(spoken_end + self._padding_ms, duration_ms)
            center_ms = self._clamp((spoken_start + spoken_end) // 2, duration_ms)
            output.append(
                KnowledgeEvidenceWindow(
                    semantic_segment_id=semantic.semantic_segment_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    center_ms=center_ms,
                    transcript_segment_ids=tuple(item.segment_id for item in selected),
                    high_signals=self._signals(selected),
                    planner_version=self._planner_version,
                )
            )
        return tuple(sorted(output, key=lambda item: (item.start_ms, item.end_ms, item.semantic_segment_id)))

    def plan_claim_drafts(
        self,
        transcript: TranscriptArtifact,
        claim_drafts: Iterable[Any],
        *,
        media_duration_ms: int | None = None,
    ) -> tuple[KnowledgeEvidenceWindow, ...]:
        """Plan one visual window per transcript-grounded atomic draft.

        This is intentionally before OCR/vision.  It uses only accepted
        transcript coordinates, so visual material cannot influence which
        proposition is selected or where its evidence window begins.
        """
        if media_duration_ms is not None and media_duration_ms < 0:
            raise ValueError("media_duration_ms must be non-negative")
        duration_ms = (
            media_duration_ms
            if media_duration_ms is not None
            else max((item.end_ms for item in transcript.segments), default=0)
        )
        by_index = {item.segment_index: item for item in transcript.segments}
        output: list[KnowledgeEvidenceWindow] = []
        seen: set[str] = set()
        for draft in claim_drafts:
            semantic_id = str(getattr(draft, "semantic_segment_id", "") or "")
            indices = tuple(
                sorted(
                    {int(value) for value in getattr(draft, "evidence_segment_indices", ()) if int(value) in by_index}
                )
            )
            if not semantic_id or not indices:
                # An atomic draft without transcript coordinates is never a
                # visual-planning authority; later grounding will reject it.
                continue
            selected = tuple(by_index[index] for index in indices)
            identity_payload = {
                "semantic_segment_id": semantic_id,
                "evidence_segment_indices": indices,
                "normalized_statement": str(
                    getattr(draft, "normalized_statement", "") or getattr(draft, "conclusion", "")
                ),
            }
            knowledge_identity = (
                "kd_" + hashlib.sha256(canonical_json(identity_payload).encode("utf-8")).hexdigest()[:57]
            )
            if knowledge_identity in seen:
                continue
            seen.add(knowledge_identity)
            spoken_start = self._clamp(min(item.start_ms for item in selected), duration_ms)
            spoken_end = self._clamp(max(item.end_ms for item in selected), duration_ms)
            output.append(
                KnowledgeEvidenceWindow(
                    semantic_segment_id=semantic_id,
                    start_ms=self._clamp(spoken_start - self._padding_ms, duration_ms),
                    end_ms=self._clamp(spoken_end + self._padding_ms, duration_ms),
                    center_ms=self._clamp((spoken_start + spoken_end) // 2, duration_ms),
                    transcript_segment_ids=tuple(item.segment_id for item in selected),
                    high_signals=self._signals(selected),
                    planner_version=self._planner_version,
                    knowledge_identity=knowledge_identity,
                )
            )
        return tuple(sorted(output, key=lambda item: (item.start_ms, item.end_ms, item.knowledge_identity)))

    @staticmethod
    def _clamp(value: int, duration_ms: int) -> int:
        return min(max(0, value), duration_ms)

    @staticmethod
    def _signals(transcript_segments: tuple[object, ...]) -> tuple[HighSignal, ...]:
        signals: list[HighSignal] = []
        for kind, reason, pattern in _SIGNAL_PATTERNS:
            ids = tuple(
                item.segment_id
                for item in transcript_segments
                if pattern.search(item.text or item.normalized_text or item.raw_text or "")
            )
            if ids:
                signals.append(HighSignal(kind=kind, reason=reason, transcript_segment_ids=ids))
        return tuple(signals)


__all__ = [
    "KNOWLEDGE_EVIDENCE_WINDOW_PLANNER_VERSION",
    "HighSignal",
    "KnowledgeEvidenceWindow",
    "KnowledgeEvidenceWindowPlanner",
]

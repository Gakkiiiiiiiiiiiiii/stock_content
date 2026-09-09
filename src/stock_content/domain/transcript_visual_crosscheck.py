"""Deterministic, transcript-primary checks for targeted visual evidence.

The output is deliberately an internal trace rather than a public artifact.
It decides whether a frame can enter optional visual context; it never changes
the selected transcript or creates claim evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

_TICKER = re.compile(r"(?<!\d)\d{6}(?!\d)")
_NUMBER = re.compile(
    r"(?<![\d.])\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|％|亿元|万亿元|万元|万|亿|倍|个|台|吨|美元|人民币|USD|CNY)?"
)
_DATE = re.compile(
    r"(?:20\d{2}\s*年?\s*)?[Qq][1-4]|(?:19|20)\d{2}(?:年|[-/.])(?:0?[1-9]|1[0-2])(?:月|[-/.])?(?:0?[1-9]|[12]\d|3[01])?|\d{1,2}月(?:\d{1,2}日)?"
)
_ENTITY = re.compile(
    r"[\u4e00-\u9fffA-Za-z]{2,20}(?:股份有限公司|有限公司|集团|银行|证券|科技|控股|汽车|能源|算力|半导体|芯片|金融|保险|茅台|时代|英伟达|GPU|CPU|服务器|平台)"
)
_TERMS = (
    "政策",
    "文件",
    "公告",
    "通知",
    "银行",
    "券商",
    "保险",
    "算力",
    "半导体",
    "芯片",
    "GPU",
    "CPU",
    "K线",
    "走势图",
    "图表",
    "成交量",
    "增资",
    "降准",
)
_DIRECTION = {
    "UP": ("上涨", "上升", "增长", "走高", "利好", "突破", "increase", "up"),
    "DOWN": ("下跌", "下降", "回落", "走低", "利空", "跌破", "decrease", "down"),
}
_SEMANTIC_ANCHOR_KINDS = frozenset(("TERM", "ENTITY", "TICKER"))
_MATERIAL_FACT_KINDS = frozenset(("NUMBER", "DATE", "DIRECTION"))


def _normal(value: str) -> str:
    return re.sub(r"\s+", "", value).upper().replace("％", "%").replace(",", "")


def _facts(text: str) -> dict[str, set[str]]:
    normalized = _normal(text)
    directions = {name for name, words in _DIRECTION.items() if any(word.upper() in normalized for word in words)}
    return {
        # Preserve word boundaries for six-digit symbols; collapsing whitespace
        # before this match would turn "600519 收入" into a false non-ticker.
        "TICKER": set(_TICKER.findall(text)),
        "NUMBER": {_normal(item) for item in _NUMBER.findall(text) if re.search(r"\d", item)},
        "DATE": {_normal(item).replace("年Q", "Q") for item in _DATE.findall(text)},
        "ENTITY": {_normal(item) for item in _ENTITY.findall(text)},
        "TERM": {term.upper() for term in _TERMS if term.upper() in normalized},
        "DIRECTION": directions,
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return None
    result = float(value)
    return result if 0 <= result <= 1 else None


def _is_structurally_valid_vision_item(value: Any) -> bool:
    """Recognize the normalized vision shape emitted by the vision stage."""
    if not isinstance(value, dict) or not isinstance(value.get("visual_summary"), str):
        return False
    if not value["visual_summary"].strip() or not isinstance(value.get("narration_aligned"), bool):
        return False
    for field, require_item in (("labels", True), ("themes", False), ("symbols", False)):
        items = value.get(field)
        if not isinstance(items, list) or (require_item and not items):
            return False
        if any(not isinstance(item, str) or not item.strip() for item in items):
            return False
    return (
        _finite(value.get("confidence_score")) is not None
        and isinstance(value.get("model"), str)
        and bool(value["model"].strip())
        and isinstance(value.get("model_version"), str)
        and bool(value["model_version"].strip())
    )


@dataclass(frozen=True)
class TranscriptVisualCrossChecker:
    """Classify a frame only against its owning transcript coordinates."""

    ocr_correction_threshold: float = 0.98
    version: str = "transcript-visual-crosscheck.v1"

    def check(
        self,
        *,
        frame: dict[str, Any],
        ocr_items: Iterable[dict[str, Any]],
        vision_item: dict[str, Any] | None,
        transcript_segments: Iterable[Any],
    ) -> dict[str, Any]:
        semantic_ids = tuple(sorted(str(value) for value in frame.get("semantic_segment_ids") or () if str(value)))
        window_ids = tuple(sorted(str(value) for value in frame.get("evidence_window_ids") or () if str(value)))
        segment_list = list(transcript_segments)
        transcript_text = " ".join(str(getattr(item, "text", "")) for item in segment_list)
        transcript_ids = tuple(
            str(getattr(item, "segment_id", "")) for item in segment_list if getattr(item, "segment_id", "")
        )
        ocr_list = [item for item in ocr_items if isinstance(item, dict)]
        ocr_text = " ".join(str(item.get("evidence_text") or item.get("text") or "") for item in ocr_list)
        vision_text = ""
        if vision_item:
            vision_text = " ".join(
                [
                    str(vision_item.get("visual_summary") or ""),
                    " ".join(str(item) for item in vision_item.get("symbols") or ()),
                    " ".join(str(item) for item in vision_item.get("labels") or ()),
                ]
            )
        transcript_facts, ocr_facts, vision_facts = _facts(transcript_text), _facts(ocr_text), _facts(vision_text)
        # OCR is the independently grounded visual evidence.  A vision model's
        # narration remains auditable, but cannot supply a financial-fact match
        # when OCR is available.
        visual_facts = ocr_facts if ocr_text else vision_facts
        reasons: list[str] = []
        matches: dict[str, list[str]] = {}
        mismatches: dict[str, dict[str, list[str]]] = {}
        if not semantic_ids or not window_ids or not transcript_ids:
            relation = "UNKNOWN"
            reasons.append("MISSING_OWNING_TRANSCRIPT_PROVENANCE")
        elif not ocr_text and not vision_text:
            relation = "UNKNOWN"
            reasons.append("NO_VISUAL_FACT_EVIDENCE")
        elif (
            _is_structurally_valid_vision_item(vision_item)
            and vision_item["narration_aligned"] is False
            and "secondary-news-page" in set(vision_item.get("labels") or ())
        ):
            # A displayed secondary page can prove only that the video showed
            # that page.  It is deliberately not narration support and never
            # an external verification result.  Claim binding applies this
            # narrow relation only to explicit attributed-secondary reports.
            relation = "SUPPORTS_DISPLAYED_SECONDARY"
            reasons.extend(("DISPLAYED_SECONDARY_PAGE_ONLY", "VISION_NARRATION_NOT_ALIGNED"))
        elif _is_structurally_valid_vision_item(vision_item) and vision_item["narration_aligned"] is False:
            # OCR from a dense market UI can accidentally repeat a ticker, term,
            # or number from the narration.  A normalized vision result that
            # explicitly rejects alignment makes that overlap ineligible.
            relation = "UNRELATED"
            reasons.append("VISION_NARRATION_NOT_ALIGNED")
        else:
            for kind, expected in transcript_facts.items():
                seen = visual_facts[kind]
                overlap = sorted(expected & seen)
                if overlap:
                    matches[kind] = overlap
                    reasons.append(f"EXACT_{kind}_MATCH")
                expected_only = expected - seen
                seen_only = seen - expected
                if expected_only and seen_only:
                    mismatches[kind] = {"transcript": sorted(expected_only), "visual": sorted(seen_only)}
                    reasons.append(f"{kind}_MISMATCH")
            semantic_anchors = _SEMANTIC_ANCHOR_KINDS & matches.keys()
            material_mismatches = _MATERIAL_FACT_KINDS & mismatches.keys()
            if matches and not ocr_text:
                # A vision response saying that it agrees with the narration is
                # not an independent financial-fact source.  Keep it auditable
                # but out of knowledge context unless OCR or another grounded
                # visual source carries the matching hard fact.
                relation = "UNKNOWN"
                reasons.append("MODEL_ONLY_FACT_NOT_INDEPENDENT_SUPPORT")
            elif semantic_anchors and material_mismatches:
                # A shared subject/action anchor makes a conflicting number,
                # date, or direction a material contradiction rather than a
                # partial visual confirmation.
                relation = "CONTRADICTS"
            elif matches:
                relation = "SUPPORTS"
            elif "TICKER" in mismatches:
                relation = "CONTRADICTS"
            else:
                relation = "UNRELATED"
                reasons.append("NO_SHARED_HARD_FACT")
        ocr_confidences = [
            value
            for item in ocr_list
            if (value := _finite(item.get("confidence_score", item.get("score")))) is not None
        ]
        vision_confidence = _finite((vision_item or {}).get("confidence_score"))
        candidates = self._correction_candidates(
            frame=frame,
            ocr_items=ocr_list,
            transcript_facts=transcript_facts,
            ocr_facts=ocr_facts,
            transcript_segment_ids=transcript_ids,
        )
        if vision_item and vision_item.get("narration_aligned") is True and not matches:
            reasons.append("NARRATION_ALIGNMENT_NOT_FACT_SUPPORT")
        return {
            "crosscheck_version": self.version,
            "frame_id": str(frame.get("frame_id") or ""),
            "frame_artifact_id": str(frame.get("frame_artifact_id") or ""),
            "timestamp_ms": int(frame.get("timestamp_ms") or 0),
            "semantic_segment_ids": list(semantic_ids),
            "evidence_window_ids": list(window_ids),
            "transcript_segment_ids": list(transcript_ids),
            "relation": relation,
            "reason_codes": sorted(set(reasons)),
            "matches": matches,
            "mismatches": mismatches,
            "confidence_inputs": {
                "ocr_confidence_scores": ocr_confidences,
                "vision_confidence_score": vision_confidence,
                "transcript_confidence_scores": [
                    value for item in segment_list if (value := _finite(getattr(item, "confidence", None))) is not None
                ],
            },
            # Candidates are trace-only: a later guarded stage must decide whether
            # independently grounded transcript correction is appropriate.
            "correction_candidates": candidates,
        }

    def _correction_candidates(
        self, *, frame, ocr_items, transcript_facts, ocr_facts, transcript_segment_ids
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for kind in ("TICKER", "NUMBER", "DATE", "ENTITY"):
            replacement = sorted(ocr_facts[kind] - transcript_facts[kind])
            original = sorted(transcript_facts[kind] - ocr_facts[kind])
            if not replacement or not original:
                continue
            high_confidence = [
                item
                for item in ocr_items
                if _finite(item.get("confidence_score", item.get("score"))) is not None
                and _finite(item.get("confidence_score", item.get("score"))) >= self.ocr_correction_threshold
            ]
            if not high_confidence:
                continue
            candidates.append(
                {
                    "kind": kind,
                    "original": original,
                    "replacement": replacement,
                    "frame_id": str(frame.get("frame_id") or ""),
                    "transcript_segment_ids": list(transcript_segment_ids),
                    "ocr_engine": str(high_confidence[0].get("ocr_engine") or ""),
                    "ocr_engine_version": str(high_confidence[0].get("ocr_engine_version") or ""),
                    "ocr_confidence_score": _finite(
                        high_confidence[0].get("confidence_score", high_confidence[0].get("score"))
                    ),
                    "status": "TRACE_ONLY_REQUIRES_INDEPENDENT_TRANSCRIPT_GROUNDING",
                }
            )
        return candidates


__all__ = ["TranscriptVisualCrossChecker"]

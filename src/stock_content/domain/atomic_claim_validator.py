"""Fail-closed validation for untrusted atomic-claim extraction JSON.

The validator deliberately proves only deterministic, transcript-local facts.
It does not attempt to make a plausible claim acceptable: uncertainty is a
rejection and no model, search, OCR, or clock access is used here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from .artifacts import TranscriptArtifact, canonical_json
from .claim_draft import AtomicClaimDraft
from .semantic_segment import SemanticSegment
from .transcript_quality import TranscriptQualityStatus


class ClaimRejectionCode(StrEnum):
    MALFORMED_JSON = "MALFORMED_JSON"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    TRANSCRIPT_QUALITY_NOT_PASS = "TRANSCRIPT_QUALITY_NOT_PASS"
    SEMANTIC_SEGMENT_MISMATCH = "SEMANTIC_SEGMENT_MISMATCH"
    EVIDENCE_COORDINATE_INVALID = "EVIDENCE_COORDINATE_INVALID"
    QUOTE_NOT_VERBATIM = "QUOTE_NOT_VERBATIM"
    NOT_ATOMIC = "NOT_ATOMIC"
    CONDITION_NOT_GROUNDED = "CONDITION_NOT_GROUNDED"
    INVALIDATION_NOT_GROUNDED = "INVALIDATION_NOT_GROUNDED"
    HARD_FACT_MISMATCH = "HARD_FACT_MISMATCH"
    PREDICATE_MISMATCH = "PREDICATE_MISMATCH"
    ENTITY_MISMATCH = "ENTITY_MISMATCH"
    SUBJECT_MISMATCH = "SUBJECT_MISMATCH"
    POLARITY_MISMATCH = "POLARITY_MISMATCH"
    TENSE_MISMATCH = "TENSE_MISMATCH"
    TEMPORAL_NOT_GROUNDED = "TEMPORAL_NOT_GROUNDED"
    VISUAL_ANCHOR_INVALID = "VISUAL_ANCHOR_INVALID"
    INFERRED_VISUAL_HIGH_RISK = "INFERRED_VISUAL_HIGH_RISK"
    REPAIR_NOT_PERMITTED = "REPAIR_NOT_PERMITTED"


@dataclass(frozen=True, slots=True)
class ClaimRejection:
    reason_code: ClaimRejectionCode
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ValidatedAtomicClaim:
    draft: AtomicClaimDraft
    contradiction_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class AtomicClaimValidationResult:
    accepted: tuple[ValidatedAtomicClaim, ...]
    rejected: tuple[ClaimRejection, ...]


class NumericAlignmentService:
    """Extract hard financial tokens without semantic expansion."""

    _patterns = (
        r"(?<!\d)\d{6}(?!\d)",  # mainland ticker
        r"(?<!\d)\d+(?:[.,]\d+)?(?:%|％)(?!\d)",
        r"(?:人民币|美元|港元|CNY|USD|HKD|¥|￥|\$)\s?\d+(?:[.,]\d+)?(?:万|亿|百万|千|元)?",
        r"\d+(?:[.,]\d+)?(?:万亿|亿|万|百万|千|元)",
        r"(?:19|20)\d{2}\s*(?:年|[-/.]\d{1,2}(?:月|[-/.]\d{1,2}日?)?)",
        r"(?:19|20)\d{2}\s*[Qq季度]\s*[1-4]?|[Qq]\s*[1-4]\s*(?:19|20)\d{2}",
        r"[零一二三四五六七八九十百千万亿两]+(?:元|年|月|日|季度|%)?",
    )

    def tokens(self, value: str) -> set[str]:
        return {self._normal(token) for pattern in self._patterns for token in re.findall(pattern, value)}

    def aligned(self, statement: str, evidence: str) -> bool:
        return self.tokens(statement).issubset(self.tokens(evidence))

    @staticmethod
    def _normal(value: str) -> str:
        return re.sub(r"\s+", "", value).replace("％", "%").upper()


class EntityAlignmentService:
    """Require stated subjects/entities to appear in the selected evidence."""

    _entity_pattern = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Za-z0-9.&-]{1,}|[\u4e00-\u9fff]{2,}")

    @staticmethod
    def _contains(value: str, evidence: str) -> bool:
        return value.casefold().strip() in evidence.casefold()

    def subject_aligned(self, draft: AtomicClaimDraft, evidence: str) -> bool:
        values = [draft.subject.subject_key, draft.subject.subject_name or ""]
        return all(not value or self._contains(value, evidence) for value in values)

    def statement_entities_aligned(self, statement: str, evidence: str) -> bool:
        # Chinese runs include ordinary words, so only enforce candidates which
        # are not predicate-like common grammar.  Subjects are always checked
        # independently above.
        ignored = {"如果", "预计", "可能", "公司", "行业", "增长", "下降", "同比", "环比", "季度", "今年", "明年"}
        entities = {item for item in self._entity_pattern.findall(statement) if item not in ignored and len(item) >= 2}
        return all(self._contains(item, evidence) for item in entities)


class TemporalGroundingService:
    def validate(
        self,
        draft: AtomicClaimDraft,
        text_for_indices: Mapping[int, str],
        valid_indices: set[int],
    ) -> bool:
        for expression in draft.temporal_expressions:
            if not set(expression.evidence_segment_indices).issubset(valid_indices):
                return False
            evidence = " ".join(text_for_indices[index] for index in expression.evidence_segment_indices)
            if not _contains(expression.raw_expression, evidence):
                return False
            if expression.target_period and not (
                _contains(expression.target_period, evidence) or expression.target_period == expression.raw_expression
            ):
                return False
        return True


class ContradictionService:
    """Group conflicting values; never select a last writer."""

    def group(self, claims: Iterable[AtomicClaimDraft]) -> dict[int, str | None]:
        ordered = list(claims)
        buckets: dict[tuple[str, str, str], list[int]] = {}
        for index, draft in enumerate(ordered):
            target = "|".join(
                sorted(
                    expression.target_period or expression.raw_expression for expression in draft.temporal_expressions
                )
            )
            key = (draft.subject.subject_key or draft.subject.subject_name or "", draft.predicate.casefold(), target)
            buckets.setdefault(key, []).append(index)
        groups: dict[int, str | None] = {index: None for index in range(len(ordered))}
        for key, indexes in buckets.items():
            values = {
                canonical_json(
                    {
                        "object": draft.object.model_dump(mode="json") if draft.object else None,
                        "polarity": draft.polarity,
                        "sentiment": draft.sentiment,
                    }
                )
                for draft in (ordered[index] for index in indexes)
            }
            if len(values) > 1:
                group_id = "contradiction_" + hashlib.sha256(canonical_json(key).encode()).hexdigest()[:24]
                for index in indexes:
                    groups[index] = group_id
        return groups


class RestrictedAtomicClaimRepair:
    """The single repair representation allowed at this boundary.

    Full replacement payloads are prohibited because they could add facts. A
    caller may only drop existing coordinates or replace a quote with a real
    substring after JSON/type parsing has already succeeded.
    """

    def apply(
        self, payload: Mapping[str, Any], operations: Iterable[Mapping[str, Any]], evidence_text: str
    ) -> dict[str, Any]:
        repaired = json.loads(json.dumps(payload, ensure_ascii=False))
        for operation in operations:
            kind = operation.get("kind")
            if kind == "DELETE_COORDINATE":
                field = operation.get("field")
                index = operation.get("index")
                if field not in {
                    "evidence_segment_indices",
                    "condition_evidence_segment_indices",
                    "invalidation_evidence_segment_indices",
                } or not isinstance(index, int):
                    raise ValueError(ClaimRejectionCode.REPAIR_NOT_PERMITTED.value)
                values = repaired.get(field)
                if not isinstance(values, list) or index not in values:
                    raise ValueError(ClaimRejectionCode.REPAIR_NOT_PERMITTED.value)
                repaired[field] = [item for item in values if item != index]
            elif kind == "REPLACE_QUOTE":
                quote = operation.get("quote")
                if not isinstance(quote, str) or not _contains(quote, evidence_text):
                    raise ValueError(ClaimRejectionCode.REPAIR_NOT_PERMITTED.value)
                repaired["verbatim_quote"] = quote
            else:
                # JSON and type repairs are parser-only; they cannot carry a
                # field mutation.  Any other mutation would be fact creation.
                raise ValueError(ClaimRejectionCode.REPAIR_NOT_PERMITTED.value)
        return repaired


class AtomicClaimDraftValidator:
    """Validate atomic claims against the selected authoritative transcript."""

    def __init__(
        self,
        *,
        numeric: NumericAlignmentService | None = None,
        entities: EntityAlignmentService | None = None,
        temporal: TemporalGroundingService | None = None,
        contradictions: ContradictionService | None = None,
    ) -> None:
        self.numeric = numeric or NumericAlignmentService()
        self.entities = entities or EntityAlignmentService()
        self.temporal = temporal or TemporalGroundingService()
        self.contradictions = contradictions or ContradictionService()

    def validate_payloads(
        self,
        payload: str | bytes | Mapping[str, Any] | list[Any],
        transcript: TranscriptArtifact,
        semantic_segments: Iterable[SemanticSegment],
        *,
        transcript_quality_status: str | TranscriptQualityStatus = TranscriptQualityStatus.PASS,
    ) -> AtomicClaimValidationResult:
        if str(transcript_quality_status) != TranscriptQualityStatus.PASS.value:
            return AtomicClaimValidationResult((), (ClaimRejection(ClaimRejectionCode.TRANSCRIPT_QUALITY_NOT_PASS),))
        try:
            raw = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        except (TypeError, json.JSONDecodeError):
            return AtomicClaimValidationResult((), (ClaimRejection(ClaimRejectionCode.MALFORMED_JSON),))
        if isinstance(raw, Mapping):
            if set(raw) != {"claims"} or not isinstance(raw.get("claims"), list):
                return AtomicClaimValidationResult((), (ClaimRejection(ClaimRejectionCode.SCHEMA_INVALID),))
            raw = raw["claims"]
        if not isinstance(raw, list):
            return AtomicClaimValidationResult((), (ClaimRejection(ClaimRejectionCode.SCHEMA_INVALID),))
        by_id = {item.semantic_segment_id: item for item in semantic_segments}
        accepted: list[AtomicClaimDraft] = []
        rejected: list[ClaimRejection] = []
        for item in raw:
            try:
                draft = AtomicClaimDraft.model_validate(item)
            except (ValidationError, TypeError, ValueError):
                rejected.append(ClaimRejection(ClaimRejectionCode.SCHEMA_INVALID))
                continue
            code = self._validate_one(draft, transcript, by_id)
            if code:
                rejected.append(ClaimRejection(code))
            else:
                accepted.append(draft)
        groups = self.contradictions.group(accepted)
        return AtomicClaimValidationResult(
            tuple(ValidatedAtomicClaim(draft, groups[index]) for index, draft in enumerate(accepted)), tuple(rejected)
        )

    def validate(
        self,
        draft: AtomicClaimDraft | Mapping[str, Any],
        transcript: TranscriptArtifact,
        semantic_segments: Iterable[SemanticSegment],
        *,
        transcript_quality_status: str | TranscriptQualityStatus = TranscriptQualityStatus.PASS,
    ) -> AtomicClaimValidationResult:
        """Single-draft convenience boundary with the same JSON validation path."""
        return self.validate_payloads(
            {"claims": [draft.model_dump(mode="json") if isinstance(draft, AtomicClaimDraft) else draft]},
            transcript,
            semantic_segments,
            transcript_quality_status=transcript_quality_status,
        )

    def _validate_one(
        self, draft: AtomicClaimDraft, transcript: TranscriptArtifact, by_id: Mapping[str, SemanticSegment]
    ) -> ClaimRejectionCode | None:
        segment = by_id.get(draft.semantic_segment_id)
        if segment is None:
            return ClaimRejectionCode.SEMANTIC_SEGMENT_MISMATCH
        by_index = {item.segment_index: item for item in transcript.segments}
        valid = set(range(segment.start_segment_index, segment.end_segment_index + 1))
        all_indices = (
            draft.evidence_segment_indices
            + draft.condition_evidence_segment_indices
            + draft.invalidation_evidence_segment_indices
            + [index for expression in draft.temporal_expressions for index in expression.evidence_segment_indices]
        )
        if not all_indices or any(index not in valid or index not in by_index for index in all_indices):
            return ClaimRejectionCode.EVIDENCE_COORDINATE_INVALID
        text_for_indices = {index: _segment_text(item) for index, item in by_index.items()}
        primary = " ".join(text_for_indices[index] for index in draft.evidence_segment_indices)
        if not _contains(draft.verbatim_quote, primary):
            return ClaimRejectionCode.QUOTE_NOT_VERBATIM
        if not self._atomic(draft.normalized_statement):
            return ClaimRejectionCode.NOT_ATOMIC
        if any(token in draft.verbatim_quote for token in ("如果", "若", "条件")) and not draft.condition_text:
            return ClaimRejectionCode.CONDITION_NOT_GROUNDED
        if any(token in draft.verbatim_quote for token in ("证伪", "失效")) and not draft.invalidation_text:
            return ClaimRejectionCode.INVALIDATION_NOT_GROUNDED
        if draft.condition_text and not (
            draft.condition_evidence_segment_indices
            and _contains(
                draft.condition_text,
                " ".join(text_for_indices[index] for index in draft.condition_evidence_segment_indices),
            )
        ):
            return ClaimRejectionCode.CONDITION_NOT_GROUNDED
        if draft.invalidation_text and not (
            draft.invalidation_evidence_segment_indices
            and _contains(
                draft.invalidation_text,
                " ".join(text_for_indices[index] for index in draft.invalidation_evidence_segment_indices),
            )
        ):
            return ClaimRejectionCode.INVALIDATION_NOT_GROUNDED
        if not self.numeric.aligned(draft.normalized_statement, primary):
            return ClaimRejectionCode.HARD_FACT_MISMATCH
        if not _contains(draft.predicate, primary):
            return ClaimRejectionCode.PREDICATE_MISMATCH
        if draft.object and draft.object.text and not _contains(draft.object.text, primary):
            return ClaimRejectionCode.HARD_FACT_MISMATCH
        if draft.object and draft.object.value is not None and not _contains(str(draft.object.value), primary):
            return ClaimRejectionCode.HARD_FACT_MISMATCH
        if not self.entities.subject_aligned(draft, primary):
            return ClaimRejectionCode.SUBJECT_MISMATCH
        if (
            _polarity(draft.normalized_statement) != _polarity(primary)
            and _polarity(draft.normalized_statement) != "NEUTRAL"
        ):
            return ClaimRejectionCode.POLARITY_MISMATCH
        if _tense(draft.normalized_statement) != _tense(primary) and _tense(draft.normalized_statement) != "UNKNOWN":
            return ClaimRejectionCode.TENSE_MISMATCH
        if draft.assertion_tense != "UNKNOWN" and draft.assertion_tense != _tense(primary):
            return ClaimRejectionCode.TENSE_MISMATCH
        if not self.entities.statement_entities_aligned(draft.normalized_statement, primary):
            return ClaimRejectionCode.ENTITY_MISMATCH
        if not self.temporal.validate(draft, text_for_indices, valid):
            return ClaimRejectionCode.TEMPORAL_NOT_GROUNDED
        if any(not _visual_valid(anchor) for anchor in draft.visual_anchors):
            return ClaimRejectionCode.VISUAL_ANCHOR_INVALID
        if any(anchor.support_type == "INFERRED_VISUAL" for anchor in draft.visual_anchors) and _high_risk(draft):
            return ClaimRejectionCode.INFERRED_VISUAL_HIGH_RISK
        return None

    @staticmethod
    def _atomic(statement: str) -> bool:
        clauses = [item.strip() for item in re.split(r"[；;。！？!?]", statement) if item.strip()]
        if len(clauses) != 1:
            return False
        # A date plus a metric value is one fact, so token counting would be
        # over-strict.  Instead reject explicit joins between two financial
        # predicates; later packets may represent derived relations explicitly.
        metrics = r"(?:营收|收入|利润|毛利率|销量|订单|价格|估值)"
        return not bool(re.search(metrics + r".*(?:以及|同时|并且|，|、|和).*" + metrics, statement))


def _segment_text(item: Any) -> str:
    return str(getattr(item, "raw_text", None) or getattr(item, "text", ""))


def _contains(needle: str, haystack: str) -> bool:
    return needle.casefold().strip() in haystack.casefold()


def _polarity(text: str) -> str:
    negative = ("下降", "下滑", "减少", "亏损", "利空", "不", "未")
    positive = ("增长", "上升", "改善", "盈利", "利好")
    if any(token in text for token in negative):
        return "NEGATIVE"
    if any(token in text for token in positive):
        return "POSITIVE"
    return "NEUTRAL"


def _tense(text: str) -> str:
    if any(token in text for token in ("如果", "若", "条件")):
        return "CONDITIONAL"
    if any(token in text for token in ("将", "预计", "预测", "未来", "明年")):
        return "FUTURE"
    if any(token in text for token in ("去年", "曾", "已经", "此前")):
        return "PAST"
    return "PRESENT" if text else "UNKNOWN"


def _high_risk(draft: AtomicClaimDraft) -> bool:
    financial = ("营收", "收入", "利润", "估值", "价格", "收益", "增长", "财务")
    return bool(NumericAlignmentService().tokens(draft.normalized_statement)) or any(
        token in (draft.predicate + draft.normalized_statement) for token in financial
    )


def _visual_valid(anchor: Any) -> bool:
    return (
        bool(anchor.frame_id and anchor.model_id and anchor.model_version)
        and bool(anchor.ocr_text or anchor.visual_label)
        and len(anchor.bbox) == 4
        and all(math.isfinite(float(value)) for value in anchor.bbox)
    )


__all__ = [
    "AtomicClaimDraftValidator",
    "AtomicClaimValidationResult",
    "ClaimRejection",
    "ClaimRejectionCode",
    "ContradictionService",
    "EntityAlignmentService",
    "NumericAlignmentService",
    "RestrictedAtomicClaimRepair",
    "TemporalGroundingService",
    "ValidatedAtomicClaim",
]

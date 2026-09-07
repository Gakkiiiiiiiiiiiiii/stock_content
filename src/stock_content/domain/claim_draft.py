from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClaimSubject(BaseModel):
    """The explicit subject of an extracted atomic claim.

    Keeping the identifier separate from display text prevents a model from
    silently swapping a company or ticker during normalization.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_type: str = "UNKNOWN"
    subject_key: str = ""
    subject_name: str | None = None

    @model_validator(mode="after")
    def _has_identity(self) -> "ClaimSubject":
        if not self.subject_key and not self.subject_name:
            raise ValueError("subject requires subject_key or subject_name")
        return self


class ClaimObject(BaseModel):
    """Structured object/value supplied by the extractor, never inferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    value: str | int | float | None = None
    unit: str | None = None
    currency: str | None = None


class VisualEvidenceAnchor(BaseModel):
    """A complete visual anchor.  It is intentionally not an EvidenceItem."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str
    timestamp_ms: int = Field(ge=0)
    bbox: tuple[float, float, float, float]
    ocr_text: str | None = None
    visual_label: str | None = None
    model_id: str
    model_version: str
    confidence: float
    support_type: Literal["OCR", "LABEL", "INFERRED_VISUAL"] = "OCR"

    @field_validator("confidence")
    @classmethod
    def _finite_confidence(cls, value: float) -> float:
        import math

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence must be finite and within [0, 1]")
        return value

    @model_validator(mode="after")
    def _has_visual_content(self) -> "VisualEvidenceAnchor":
        if not (self.ocr_text or self.visual_label):
            raise ValueError("visual anchor requires OCR text or visual label")
        if len(self.bbox) != 4 or any(not isinstance(value, (int, float)) for value in self.bbox):
            raise ValueError("visual anchor bbox must contain four numeric values")
        return self


class AtomicTemporalExpression(BaseModel):
    """A temporal assertion and its independent transcript coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_expression: str
    target_period: str | None = None
    pit_meaning: Literal["REPORTING_PERIOD", "FORECAST_TARGET", "SOURCE_PUBLISHED_AT", "UNKNOWN"] = "UNKNOWN"
    evidence_segment_indices: list[int] = Field(min_length=1)
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _finite_confidence(cls, value: float) -> float:
        import math

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence must be finite and within [0, 1]")
        return value


class AtomicClaimDraft(BaseModel):
    """Untrusted model output before any semantic acceptance.

    This is deliberately separate from ``ClaimOccurrenceDraft``.  The latter
    remains the legacy downstream DTO; only accepted AtomicClaimDraft values
    may be converted by a future projection packet.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_segment_id: str
    # The atomic validator is the acceptance boundary.  Classification is a
    # deterministic extraction label, not a later knowledge projection guess.
    claim_type: Literal[
        "PRICE", "RETURN", "VALUATION", "FINANCIAL_METRIC", "CORPORATE_EVENT",
        "INDUSTRY_RELATION", "FORECAST", "OPINION", "INFERENCE",
    ] = "INDUSTRY_RELATION"
    knowledge_kind: str = "CLAIM"
    verbatim_quote: str = Field(min_length=1)
    normalized_statement: str = Field(min_length=1)
    subject: ClaimSubject
    predicate: str = Field(min_length=1)
    object: ClaimObject | None = None
    condition_text: str | None = None
    invalidation_text: str | None = None
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL", "UNCERTAIN"]
    polarity: Literal["ASSERTS", "DENIES", "NEUTRAL"] = "ASSERTS"
    assertion_tense: Literal["PAST", "PRESENT", "FUTURE", "CONDITIONAL", "UNKNOWN"] = "UNKNOWN"
    evidence_segment_indices: list[int] = Field(min_length=1)
    condition_evidence_segment_indices: list[int] = Field(default_factory=list)
    invalidation_evidence_segment_indices: list[int] = Field(default_factory=list)
    temporal_expressions: list[AtomicTemporalExpression] = Field(default_factory=list)
    visual_anchors: list[VisualEvidenceAnchor] = Field(default_factory=list)
    extraction_confidence: float

    @field_validator("extraction_confidence")
    @classmethod
    def _finite_confidence(cls, value: float) -> float:
        import math

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence must be finite and within [0, 1]")
        return value


class TemporalExpressionDraft(BaseModel):
    role: str
    raw_expression: str
    scope_hint: str | None = None
    anchor: str | None = None
    evidence_segment_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ClaimOccurrenceDraft(BaseModel):
    semantic_segment_id: str
    knowledge_kind: str
    claim_type: str
    subject_type: str | None = None
    subject_key: str = ""
    subject_name: str | None = None
    predicate_key: str = ""
    conclusion: str = ""
    value: Any = None
    unit: str | None = None
    currency: str | None = None
    sentiment: str = "NEUTRAL"
    condition_text: str | None = None
    invalidation_text: str | None = None
    evidence_segment_indices: list[int] = Field(default_factory=list)
    condition_evidence_segment_indices: list[int] = Field(default_factory=list)
    invalidation_evidence_segment_indices: list[int] = Field(default_factory=list)
    temporal_expressions: list[TemporalExpressionDraft] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    entity_corrections: list[dict[str, Any]] = Field(default_factory=list)
    extraction_confidence: float = Field(default=0.0, ge=0, le=1)
    extraction_model_id: str = ""
    extraction_prompt_version: str = ""
    # These fields are only populated by the SC-07A accepted-draft adapter.
    # Their legacy defaults are intentionally fail-closed for formal use.
    verbatim_quote: str | None = None
    normalized_statement: str | None = None
    grounding_status: str = "LEGACY_UNGROUNDED"
    grounding_reason_codes: list[str] = Field(default_factory=list)
    contradiction_group_id: str | None = None
    # The old canonical DTO still produces the final occurrence-owned claim
    # shape, but is explicitly denied formal eligibility by its grounding
    # marker until a validator-backed backfill is performed.
    claim_schema_version: str = "claim.final.v1"
    legacy_grounding_incomplete: bool = True


__all__ = ["ClaimOccurrenceDraft", "TemporalExpressionDraft"]

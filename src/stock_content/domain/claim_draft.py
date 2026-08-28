from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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


__all__ = ["ClaimOccurrenceDraft", "TemporalExpressionDraft"]

"""FinancialClaim 正式领域模型（详细修改方案 §5 P1-1/P1-3）。

必须区分 Fact / Forecast / Opinion / Inference：
“明年 GLP-1 API 大概率放量”与“公司 2025 年营收 30 亿元”不能进入同一事实层。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .temporal_identity import canonical_bindings, canonical_relations
from .temporal_semantics import ClaimTemporalBinding, ClaimTemporalRelation

ClaimType = Literal[
    "PRICE",
    "RETURN",
    "VALUATION",
    "FINANCIAL_METRIC",
    "CORPORATE_EVENT",
    "INDUSTRY_RELATION",
    "FORECAST",
    "OPINION",
    "INFERENCE",
]

CLAIM_TYPES: tuple[str, ...] = (
    "PRICE",
    "RETURN",
    "VALUATION",
    "FINANCIAL_METRIC",
    "CORPORATE_EVENT",
    "INDUSTRY_RELATION",
    "FORECAST",
    "OPINION",
)

# claim_type -> 事实层分类。FACT 层允许进入外部核验；其余层禁止冒充事实。
CLAIM_CATEGORY: dict[str, str] = {
    "PRICE": "FACT",
    "RETURN": "FACT",
    "VALUATION": "FACT",
    "FINANCIAL_METRIC": "FACT",
    "CORPORATE_EVENT": "FACT",
    "INDUSTRY_RELATION": "FACT",
    "FORECAST": "FORECAST",
    "OPINION": "OPINION",
    # Kept outside the legacy CLAIM_TYPES tuple for v1 consumers while being
    # a distinct canonical category in the v2 model.
    "INFERENCE": "INFERENCE",
}

# 可绑定 Quant 市场快照进行外部核验的类型（P1-3）。
QUANT_VERIFIABLE_TYPES: frozenset[str] = frozenset({"PRICE", "RETURN", "VALUATION", "FINANCIAL_METRIC"})

VERIFICATION_STATUSES = (
    "EXTRACTED",
    "VERIFICATION_PENDING",
    "NOT_REQUIRED",
    "VERIFIED",
    "CONTRADICTED",
    "PARTIALLY_VERIFIED",
    "NOT_VERIFIABLE",
    "EXPIRED",
    "MANUAL_REVIEW",
)
SUPPORTED_CLAIM_TYPES: frozenset[str] = frozenset((*CLAIM_TYPES, "INFERENCE"))


class FinancialClaim(BaseModel):
    claim_id: str = ""
    claim_type: ClaimType
    fact_category: str = ""

    subject_type: str
    subject_id: str
    ticker: str | None = None

    predicate: str
    value: Any = None
    unit: str | None = None
    currency: str | None = None

    fact_time: datetime | date | None = None
    period_start: date | None = None
    period_end: date | None = None
    published_at: datetime | None = None

    evidence_refs: list[str] = Field(default_factory=list)
    source_support_status: Literal[
        "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "AMBIGUOUS"
    ] = "UNSUPPORTED"

    source_confidence: float = Field(ge=0.0, le=1.0)
    extractor_confidence: float = Field(ge=0.0, le=1.0)
    extraction_model_id: str = "unknown"
    extraction_prompt_version: str = "unknown"
    condition_text: str | None = None
    invalidation_text: str | None = None
    condition_key: str | None = None
    temporal_bindings: list[ClaimTemporalBinding] = Field(default_factory=list)
    temporal_relations: list[ClaimTemporalRelation] = Field(default_factory=list)
    claim_schema_version: str = "claim.v2"
    normalization_version: str = "normalization.v1"

    @field_validator("claim_type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in SUPPORTED_CLAIM_TYPES:
            raise ValueError(f"unknown claim_type: {value}")
        return value

    @field_validator("source_support_status", mode="before")
    @classmethod
    def _normalize_support_status(cls, value: object) -> object:
        return "PARTIALLY_SUPPORTED" if str(value).upper() == "PARTIAL" else value

    @model_validator(mode="after")
    def _invariants(self) -> "FinancialClaim":
        # 所有 claim 必须有 evidence（最终验收标准）。
        if not self.evidence_refs and self.claim_schema_version != "claim.final.v1":
            raise ValueError("claim requires at least one evidence_ref")
        if not self.fact_category:
            self.fact_category = CLAIM_CATEGORY[self.claim_type]
        if not self.claim_id:
            self.claim_id = claim_id_of(self)
        return self

    def content_payload(self) -> dict[str, Any]:
        payload = {
            "claim_type": self.claim_type,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "value": self.value,
            "unit": self.unit,
            "currency": self.currency,
            "fact_time": str(self.fact_time) if self.fact_time else None,
            "period_start": str(self.period_start) if self.period_start else None,
            "period_end": str(self.period_end) if self.period_end else None,
            "claim_schema_version": self.claim_schema_version,
            "normalization_version": self.normalization_version,
        }
        if self.condition_key or self.temporal_bindings or self.temporal_relations:
            # Legacy fact_time/period fields are compatibility projections and
            # cannot alter identity once temporal bindings are authoritative.
            payload.pop("fact_time", None)
            payload.pop("period_start", None)
            payload.pop("period_end", None)
            payload.update({
                "condition_key": self.condition_key,
                "temporal_bindings": [x.temporal_binding_id for x in canonical_bindings(self.temporal_bindings)],
                "temporal_relations": [x.temporal_relation_id for x in canonical_relations(self.temporal_relations)],
            })
        return payload


def claim_id_of(claim: FinancialClaim) -> str:
    payload = json.dumps(
        claim.content_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return "claim-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def is_quant_verifiable(claim: FinancialClaim) -> bool:
    return claim.claim_type in QUANT_VERIFIABLE_TYPES


class VerificationResult(BaseModel):
    """P1-3：核验结果必须绑定 Quant Snapshot，禁止只保存 verified=true。"""

    claim_id: str
    status: Literal[
        "VERIFIED",
        "CONTRADICTED",
        "PARTIALLY_VERIFIED",
        "NOT_VERIFIABLE",
        "NOT_REQUIRED",
        "MANUAL_REVIEW",
        "EXPIRED",
        "VERIFICATION_PENDING",
    ]

    market_snapshot_id: str | None = None
    market_data_version: str | None = None
    fact_date: date | datetime | None = None
    adjustment: str | None = None  # 复权口径：NONE / FORWARD / BACKWARD
    verification_timestamp: datetime | None = None
    verification_rule_version: str = "verification_rule.v1"

    reference_value: Any = None
    deviation: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _binding_invariants(self) -> "VerificationResult":
        if self.status in {"VERIFIED", "CONTRADICTED", "PARTIALLY_VERIFIED"}:
            missing = [
                name
                for name in ("market_snapshot_id", "market_data_version", "fact_date", "verification_timestamp")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"{self.status} requires quant snapshot binding, missing: {missing}")
        return self


__all__ = [
    "CLAIM_CATEGORY",
    "CLAIM_TYPES",
    "SUPPORTED_CLAIM_TYPES",
    "ClaimType",
    "FinancialClaim",
    "QUANT_VERIFIABLE_TYPES",
    "VERIFICATION_STATUSES",
    "VerificationResult",
    "claim_id_of",
    "is_quant_verifiable",
]

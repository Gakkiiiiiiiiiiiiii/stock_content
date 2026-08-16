"""FinancialClaim 类型系统测试（详细修改方案 §5 P1-1）。

Fact / Forecast / Opinion 必须明确区分，不能进入同一事实层。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_content.domain.claims import (
    CLAIM_CATEGORY,
    CLAIM_TYPES,
    FinancialClaim,
    claim_id_of,
    is_quant_verifiable,
)


def _claim(**overrides) -> FinancialClaim:
    payload = {
        "claim_type": "FINANCIAL_METRIC",
        "subject_type": "EQUITY",
        "subject_id": "600519.SH",
        "predicate": "revenue",
        "value": 30.0,
        "unit": "亿元",
        "evidence_refs": ["ev-1"],
        "source_confidence": 0.9,
        "extractor_confidence": 0.8,
    }
    payload.update(overrides)
    return FinancialClaim(**payload)


def test_fact_forecast_opinion_never_share_the_same_layer():
    fact = _claim(claim_type="FINANCIAL_METRIC")
    forecast = _claim(claim_type="FORECAST", predicate="glp1_volume")
    opinion = _claim(claim_type="OPINION", predicate="outlook")
    assert fact.fact_category == "FACT"
    assert forecast.fact_category == "FORECAST"
    assert opinion.fact_category == "OPINION"
    assert {CLAIM_CATEGORY[claim_type] for claim_type in CLAIM_TYPES} == {"FACT", "FORECAST", "OPINION"}


def test_claim_requires_evidence():
    with pytest.raises(ValidationError):
        _claim(evidence_refs=[])


def test_unknown_claim_type_rejected():
    with pytest.raises(ValidationError):
        _claim(claim_type="RUMOR")


def test_claim_id_is_content_addressed_and_stable():
    first = _claim()
    second = _claim()
    assert first.claim_id == second.claim_id == claim_id_of(first)
    assert _claim(value=12.0).claim_id != first.claim_id


def test_only_fact_layer_claims_are_quant_verifiable():
    assert is_quant_verifiable(_claim(claim_type="PRICE"))
    assert is_quant_verifiable(_claim(claim_type="VALUATION"))
    assert not is_quant_verifiable(_claim(claim_type="FORECAST"))
    assert not is_quant_verifiable(_claim(claim_type="OPINION"))


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        _claim(source_confidence=1.5)

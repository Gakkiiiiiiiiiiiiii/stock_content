"""知识冲突测试（详细修改方案 §5 P1-4）：不能简单覆盖，必须显式登记。"""
from __future__ import annotations

import pytest

from stock_content.application.conflict_service import ConflictService
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.conflict import auto_resolve, conflict_key_of, detect_conflicts


def _claim(value: float, confidence: float = 0.5, **overrides) -> FinancialClaim:
    payload = {
        "claim_type": "FINANCIAL_METRIC",
        "subject_type": "EQUITY",
        "subject_id": "600519.SH",
        "predicate": "net_profit_2026",
        "value": value,
        "unit": "亿元",
        "fact_time": "2026-12-31",
        "evidence_refs": ["ev-1"],
        "source_confidence": confidence,
        "extractor_confidence": 0.7,
    }
    payload.update(overrides)
    return FinancialClaim(**payload)


def test_same_entity_same_window_different_value_is_conflict():
    claim_a = _claim(10.0)
    claim_b = _claim(12.0)
    assert conflict_key_of(claim_a) == conflict_key_of(claim_b)
    groups = detect_conflicts([claim_a, claim_b])
    assert len(groups) == 1
    assert {claim.claim_id for claim in groups[0]} == {claim_a.claim_id, claim_b.claim_id}


def test_same_value_is_not_conflict():
    assert detect_conflicts([_claim(10.0), _claim(10.0)]) == []


def test_conflict_not_silently_overwritten():
    service = ConflictService()
    first = service.register_claims([_claim(10.0, confidence=0.5)])
    assert first == []
    conflicts = service.register_claims([_claim(12.0, confidence=0.52)])
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.resolution_status == "OPEN"
    assert conflict.resolution_policy == "MANUAL_REVIEW"
    assert len(conflict.claim_ids) == 2

    listed = service.list_conflicts(status="OPEN")
    assert [item["conflict_id"] for item in listed] == [conflict.conflict_id]


def test_confidence_gap_auto_resolves():
    service = ConflictService()
    service.register_claims([_claim(10.0, confidence=0.5)])
    conflicts = service.register_claims([_claim(12.0, confidence=0.9)])
    assert conflicts[0].resolution_status == "AUTO_RESOLVED"
    assert conflicts[0].resolution_policy == "HIGHER_SOURCE_CONFIDENCE"
    assert conflicts[0].preferred_claim_id is not None


def test_external_verification_wins():
    claim_a = _claim(10.0, confidence=0.5)
    claim_b = _claim(12.0, confidence=0.9)
    conflict = auto_resolve([claim_a, claim_b], verified_claim_ids={claim_a.claim_id})
    assert conflict.resolution_status == "AUTO_RESOLVED"
    assert conflict.resolution_policy == "EXTERNAL_VERIFICATION_WINS"
    assert conflict.preferred_claim_id == claim_a.claim_id


def test_manual_resolve_requires_member_claim():
    service = ConflictService()
    service.register_claims([_claim(10.0)])
    conflicts = service.register_claims([_claim(12.0)])
    conflict_id = conflicts[0].conflict_id

    with pytest.raises(ValueError):
        service.manual_resolve(conflict_id, "claim-not-in-group")

    preferred = conflicts[0].claim_ids[0]
    resolved = service.manual_resolve(conflict_id, preferred, resolution_evidence={"reviewer": "ops"})
    assert resolved.resolution_status == "MANUAL_RESOLVED"
    assert resolved.preferred_claim_id == preferred

    with pytest.raises(ValueError):
        service.manual_resolve(conflict_id, preferred)  # 已解析不得重复解析


def test_different_time_window_is_not_conflict():
    early = _claim(10.0, fact_time="2025-01-10")
    late = _claim(12.0, fact_time="2026-06-10")
    assert conflict_key_of(early) != conflict_key_of(late)
    assert detect_conflicts([early, late]) == []

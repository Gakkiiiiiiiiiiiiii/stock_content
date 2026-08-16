"""Quant Verification 生命周期测试（详细修改方案 §5 P1-2/P1-3）。

Quant 短暂不可用不得失败主链；按退避重试；达到阈值进入 DLQ/manual review。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stock_content.application.verification_service import (
    RETRY_SCHEDULE_SECONDS,
    VerificationService,
    VerificationStateError,
    run_verification_pass,
    validate_transition,
)
from stock_content.domain.claims import FinancialClaim, VerificationResult


class FakeQuantProvider:
    def __init__(self, fail_times: int = 0, status: str = "VERIFIED") -> None:
        self._fail_times = fail_times
        self._status = status
        self.calls = 0

    def verify(self, claim: FinancialClaim) -> VerificationResult:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("quant unavailable")
        return VerificationResult(
            claim_id=claim.claim_id,
            status=self._status,
            market_snapshot_id="market-snap-1",
            market_data_version="md.v1",
            fact_date=datetime(2026, 8, 15, tzinfo=UTC).date(),
            adjustment="FORWARD",
            verification_timestamp=datetime.now(UTC),
            reference_value=30.0,
            deviation=0.0,
        )


def _price_claim() -> FinancialClaim:
    return FinancialClaim(
        claim_type="PRICE",
        subject_type="EQUITY",
        subject_id="600519.SH",
        predicate="close_price",
        value=1700.0,
        evidence_refs=["ev-1"],
        source_confidence=0.9,
        extractor_confidence=0.8,
    )


def test_quant_unavailable_keeps_ingest_alive_and_schedules_retry():
    service = VerificationService(provider=None)
    item = service.submit(_price_claim())
    assert item.status == "VERIFICATION_PENDING"

    now = datetime.now(UTC)
    service.attempt(item.claim.claim_id, now)
    refreshed = service.get(item.claim.claim_id)
    assert refreshed.status == "VERIFICATION_PENDING"
    assert refreshed.retry_count == 1
    assert refreshed.next_retry_at == now + timedelta(seconds=RETRY_SCHEDULE_SECONDS[0])


def test_retry_schedule_escalates_to_dlq():
    service = VerificationService(provider=None)
    item = service.submit(_price_claim())
    now = datetime.now(UTC)
    for index in range(len(RETRY_SCHEDULE_SECONDS) + 1):
        service.attempt(item.claim.claim_id, now)
        item.next_retry_at = now  # 模拟到期
    final = service.get(item.claim.claim_id)
    assert final.status == "MANUAL_REVIEW"
    assert service.dlq() == [item.claim.claim_id]


def test_successful_verification_reaches_terminal_state():
    provider = FakeQuantProvider(fail_times=1)
    service = VerificationService(provider=provider)
    item = service.submit(_price_claim())

    now = datetime.now(UTC)
    service.attempt(item.claim.claim_id, now)  # 第一次失败 -> 退避
    item.next_retry_at = now
    result = service.attempt(item.claim.claim_id, now)
    assert result.status == "VERIFIED"
    assert result.result.market_snapshot_id == "market-snap-1"
    assert result.result.market_data_version == "md.v1"
    # 终态不再重试
    assert service.attempt(item.claim.claim_id, now).status == "VERIFIED"
    assert service.dlq() == []


def test_non_quant_claims_go_directly_to_not_verifiable():
    service = VerificationService(provider=None)
    opinion = FinancialClaim(
        claim_type="OPINION",
        subject_type="EQUITY",
        subject_id="600519.SH",
        predicate="outlook",
        value="看好",
        evidence_refs=["ev-1"],
        source_confidence=0.6,
        extractor_confidence=0.5,
    )
    item = service.submit(opinion)
    assert item.status == "NOT_VERIFIABLE"
    assert service.due_items() == []


def test_state_machine_rejects_invalid_transitions():
    with pytest.raises(VerificationStateError):
        validate_transition("VERIFIED", "VERIFICATION_PENDING")
    with pytest.raises(VerificationStateError):
        validate_transition("EXTRACTED", "VERIFIED")
    validate_transition("VERIFICATION_PENDING", "CONTRADICTED")  # 合法


def test_worker_pass_processes_only_due_items():
    provider = FakeQuantProvider()
    service = VerificationService(provider=provider)
    first = service.submit(_price_claim())
    second = service.submit(
        FinancialClaim(
            claim_type="RETURN",
            subject_type="EQUITY",
            subject_id="600519.SH",
            predicate="weekly_return",
            value=0.02,
            evidence_refs=["ev-2"],
            source_confidence=0.9,
            extractor_confidence=0.8,
        )
    )
    now = datetime.now(UTC)
    # first 未到重试时间（模拟上次失败后的退避）
    first.next_retry_at = now + timedelta(hours=1)

    processed = run_verification_pass(service, now)
    assert [entry.claim.claim_id for entry in processed] == [second.claim.claim_id]
    assert second.status == "VERIFIED"

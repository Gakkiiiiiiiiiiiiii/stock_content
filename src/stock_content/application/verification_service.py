"""Quant Verification 异步生命周期（详细修改方案 §5 P1-2）。

状态机：
    EXTRACTED -> VERIFICATION_PENDING -> VERIFIED / CONTRADICTED /
        PARTIALLY_VERIFIED / NOT_VERIFIABLE / EXPIRED

主 ingest pipeline 不因 Quant 短暂不可用失败：外部服务不可用时
claim 进入 VERIFICATION_PENDING，由 verification_worker 按退避重试，
达到阈值进入 DLQ / manual review。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from stock_content.domain.claims import FinancialClaim, VerificationResult, is_quant_verifiable

# Retry 退避序列（秒）：1 min / 5 min / 30 min / 2 h / 12 h。
RETRY_SCHEDULE_SECONDS: tuple[int, ...] = (60, 300, 1800, 7200, 43200)

TERMINAL_VERIFICATION_STATUSES = frozenset(
    {"VERIFIED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE", "EXPIRED", "MANUAL_REVIEW"}
)

_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "EXTRACTED": frozenset({"VERIFICATION_PENDING", "NOT_VERIFIABLE"}),
    "VERIFICATION_PENDING": frozenset(
        {"VERIFIED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE", "EXPIRED", "MANUAL_REVIEW"}
    ),
}


class VerificationStateError(Exception):
    """非法状态迁移。"""


def validate_transition(current: str, target: str) -> None:
    if current == target:
        return
    allowed = _VALID_TRANSITIONS.get(current)
    if allowed is None or target not in allowed:
        raise VerificationStateError(f"invalid verification transition: {current} -> {target}")


@dataclass
class VerificationItem:
    claim: FinancialClaim
    status: str = "EXTRACTED"
    retry_count: int = 0
    next_retry_at: datetime | None = None
    result: VerificationResult | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim.claim_id,
            "claim_type": self.claim.claim_type,
            "status": self.status,
            "retry_count": self.retry_count,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "result": self.result.model_dump(mode="json") if self.result else None,
        }


class QuantVerificationProvider(Protocol):
    def verify(self, claim: FinancialClaim) -> VerificationResult: ...


class VerificationService:
    """Claim 验证生命周期管理（内存实现；生产由 worker + DB 驱动）。"""

    def __init__(self, provider: QuantVerificationProvider | None = None) -> None:
        self._provider = provider
        self._items: dict[str, VerificationItem] = {}
        self._dlq: list[str] = []

    def submit(self, claim: FinancialClaim) -> VerificationItem:
        """主链路提交：不可核验直接 NOT_VERIFIABLE；否则进入 PENDING（不阻塞 ingest）。"""
        if claim.claim_id in self._items:
            return self._items[claim.claim_id]
        if not is_quant_verifiable(claim):
            item = VerificationItem(claim=claim, status="NOT_VERIFIABLE")
            item.history.append({"status": "NOT_VERIFIABLE", "reason": "CLAIM_TYPE_NOT_VERIFIABLE"})
        else:
            item = VerificationItem(claim=claim, status="VERIFICATION_PENDING")
            item.history.append({"status": "VERIFICATION_PENDING"})
        self._items[claim.claim_id] = item
        return item

    def due_items(self, now: datetime | None = None) -> list[VerificationItem]:
        now = now or datetime.now(UTC)
        return [
            item
            for item in self._items.values()
            if item.status == "VERIFICATION_PENDING" and (item.next_retry_at is None or item.next_retry_at <= now)
        ]

    def attempt(self, claim_id: str, now: datetime | None = None) -> VerificationItem:
        """尝试核验一次；Quant 不可用时按退避重试，超过阈值进入 DLQ。"""
        now = now or datetime.now(UTC)
        item = self._items.get(claim_id)
        if item is None:
            raise KeyError(claim_id)
        if item.status != "VERIFICATION_PENDING":
            return item
        try:
            if self._provider is None:
                raise RuntimeError("quant provider unavailable")
            result = self._provider.verify(item.claim)
        except Exception as exc:  # noqa: BLE001 - Quant 不可用不得推翻主链路
            item.retry_count += 1
            item.history.append({"status": "RETRY_SCHEDULED", "error": str(exc), "retry": item.retry_count})
            if item.retry_count > len(RETRY_SCHEDULE_SECONDS):
                item.status = "MANUAL_REVIEW"
                item.history.append({"status": "MANUAL_REVIEW", "reason": "RETRY_EXHAUSTED"})
                self._dlq.append(claim_id)
            else:
                delay = RETRY_SCHEDULE_SECONDS[min(item.retry_count, len(RETRY_SCHEDULE_SECONDS)) - 1]
                item.next_retry_at = now + timedelta(seconds=delay)
            return item
        validate_transition(item.status, result.status)
        item.status = result.status
        item.result = result
        item.history.append({"status": result.status})
        return item

    def manual_resolve(self, claim_id: str, result: VerificationResult) -> VerificationItem:
        item = self._items.get(claim_id)
        if item is None:
            raise KeyError(claim_id)
        if item.status not in {"MANUAL_REVIEW", "VERIFICATION_PENDING", "EXTRACTED"}:
            raise VerificationStateError(f"manual_resolve not allowed from {item.status}")
        item.status = result.status
        item.result = result
        item.history.append({"status": result.status, "actor": "manual"})
        return item

    def expire(self, claim_id: str) -> VerificationItem:
        item = self._items.get(claim_id)
        if item is None:
            raise KeyError(claim_id)
        validate_transition(item.status, "EXPIRED")
        item.status = "EXPIRED"
        item.history.append({"status": "EXPIRED"})
        return item

    def get(self, claim_id: str) -> VerificationItem | None:
        return self._items.get(claim_id)

    def pending_count(self) -> int:
        return sum(1 for item in self._items.values() if item.status == "VERIFICATION_PENDING")

    def verified_claim_ids(self) -> set[str]:
        return {item.claim.claim_id for item in self._items.values() if item.status == "VERIFIED"}

    def dlq(self) -> list[str]:
        return list(self._dlq)


def run_verification_pass(service: VerificationService, now: datetime | None = None) -> list[VerificationItem]:
    """worker 单轮：处理所有到期的 PENDING claim。"""
    return [service.attempt(item.claim.claim_id, now) for item in service.due_items(now)]


__all__ = [
    "RETRY_SCHEDULE_SECONDS",
    "TERMINAL_VERIFICATION_STATUSES",
    "QuantVerificationProvider",
    "VerificationItem",
    "VerificationService",
    "VerificationStateError",
    "run_verification_pass",
    "validate_transition",
]

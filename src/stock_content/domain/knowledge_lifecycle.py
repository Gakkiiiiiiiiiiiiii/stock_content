"""Knowledge 生命周期 TTL 策略（详细修改方案 §5 P1-5）。

不同知识类型有不同保鲜期：
- 实时价格：分钟/小时级；
- 估值：日级；
- 财报事实：长期；
- 行业逻辑：事件驱动更新（无固定 TTL）；
- 预测：到目标日期后自动进入 outcome review。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

# claim_type -> TTL。None 表示无固定 TTL（事件驱动更新）。
KNOWLEDGE_TTL: dict[str, timedelta | None] = {
    "PRICE": timedelta(hours=1),
    "RETURN": timedelta(hours=1),
    "VALUATION": timedelta(days=1),
    "FINANCIAL_METRIC": timedelta(days=3650),  # 财报事实：长期
    "CORPORATE_EVENT": timedelta(days=3650),
    "INDUSTRY_RELATION": None,  # 行业逻辑：事件驱动更新
    "FORECAST": None,  # 预测：到目标日期后进入 outcome review
    "OPINION": timedelta(days=90),
}

LIFECYCLE_ACTIVE_SET = frozenset({"ACTIVE", "VALIDATED"})


@dataclass(frozen=True)
class LifecycleEvaluation:
    current_status: str
    action: str  # KEEP / MARK_STALE / OUTCOME_REVIEW
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"current_status": self.current_status, "action": self.action, "reason": self.reason}


def ttl_for(claim_type: str) -> timedelta | None:
    return KNOWLEDGE_TTL.get(claim_type)


def evaluate_lifecycle(
    *,
    claim_type: str,
    current_status: str,
    as_of: datetime,
    available_from: datetime | None = None,
    fact_time: datetime | date | None = None,
) -> LifecycleEvaluation:
    """判定单条知识是否需要生命周期迁移。"""
    if current_status not in LIFECYCLE_ACTIVE_SET:
        return LifecycleEvaluation(current_status, "KEEP", "NOT_ACTIVE")

    if claim_type == "FORECAST":
        target = fact_time
        if target is not None:
            target_date = target.date() if isinstance(target, datetime) else target
            if target_date <= as_of.date():
                return LifecycleEvaluation(current_status, "OUTCOME_REVIEW", "FORECAST_TARGET_DATE_REACHED")
        return LifecycleEvaluation(current_status, "KEEP", "FORECAST_PENDING")

    ttl = ttl_for(claim_type)
    if ttl is None:
        return LifecycleEvaluation(current_status, "KEEP", "EVENT_DRIVEN")

    anchor = available_from or as_of
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    if as_of - anchor > ttl:
        return LifecycleEvaluation(current_status, "MARK_STALE", f"TTL_EXPIRED:{claim_type}")
    return LifecycleEvaluation(current_status, "KEEP", "WITHIN_TTL")


__all__ = [
    "KNOWLEDGE_TTL",
    "LIFECYCLE_ACTIVE_SET",
    "LifecycleEvaluation",
    "evaluate_lifecycle",
    "ttl_for",
]

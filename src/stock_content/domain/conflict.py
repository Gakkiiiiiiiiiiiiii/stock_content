"""知识冲突领域模型（详细修改方案 §5 P1-4）。

同一实体、同一时间窗口的不同数值不能简单覆盖，
必须产生显式 KnowledgeConflict 记录并走解析流程。
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from stock_content.domain.claims import FinancialClaim

CONFLICT_STATUSES = ("OPEN", "AUTO_RESOLVED", "MANUAL_RESOLVED", "SUPERSEDED")

RESOLUTION_POLICIES = (
    "HIGHER_SOURCE_CONFIDENCE",
    "EXTERNAL_VERIFICATION_WINS",
    "LATEST_PUBLISHED",
    "MANUAL_REVIEW",
)


def _conflict_window(fact_time: Any, days: int = 30) -> str:
    """时间窗口归一：同一窗口内的断言才构成冲突。"""
    if fact_time is None:
        return "no-time"
    if hasattr(fact_time, "toordinal"):
        return f"window-{fact_time.toordinal() // days}"
    digest = hashlib.sha256(str(fact_time).encode("utf-8")).hexdigest()
    return f"window-{int(digest[:8], 16) % 10000}"


def conflict_key_of(claim: FinancialClaim, *, window_days: int = 30) -> str:
    return "|".join(
        [
            claim.subject_type,
            claim.subject_id,
            claim.predicate,
            _conflict_window(claim.fact_time, window_days),
        ]
    )


@dataclass
class KnowledgeConflict:
    conflict_id: str
    claim_ids: tuple[str, ...]
    conflict_type: str  # VALUE_CONFLICT / STATUS_CONFLICT / TEMPORAL_CONFLICT
    resolution_status: str = "OPEN"
    preferred_claim_id: str | None = None
    resolution_policy: str | None = None
    resolution_evidence: dict[str, Any] = field(default_factory=dict)
    conflict_key: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["claim_ids"] = list(self.claim_ids)
        return payload


def detect_conflicts(claims: list[FinancialClaim], *, window_days: int = 30) -> list[list[FinancialClaim]]:
    """同一实体+谓词+时间窗口内存在多个不同取值即构成冲突组。"""
    groups: dict[str, list[FinancialClaim]] = {}
    for claim in claims:
        groups.setdefault(conflict_key_of(claim, window_days=window_days), []).append(claim)
    conflicts: list[list[FinancialClaim]] = []
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        values = {str(item.value) for item in group}
        if len(values) > 1:
            conflicts.append(group)
    return conflicts


def _conflict_id(conflict_key: str, claim_ids: tuple[str, ...]) -> str:
    payload = conflict_key + "::" + ",".join(sorted(claim_ids))
    return "conflict-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def auto_resolve(
    group: list[FinancialClaim],
    *,
    verified_claim_ids: set[str] | None = None,
) -> KnowledgeConflict:
    """确定性解析策略：外部核验赢 > 来源置信度高 > 最新披露。

    置信度无法拉开差距时保持 OPEN，进入人工审核，绝不静默覆盖。
    """
    verified = verified_claim_ids or set()
    claim_ids = tuple(sorted(claim.claim_id for claim in group))
    conflict_key = conflict_key_of(group[0])
    conflict = KnowledgeConflict(
        conflict_id=_conflict_id(conflict_key, claim_ids),
        claim_ids=claim_ids,
        conflict_type="VALUE_CONFLICT",
        conflict_key=conflict_key,
    )

    verified_in_group = [claim for claim in group if claim.claim_id in verified]
    if len(verified_in_group) == 1:
        conflict.resolution_status = "AUTO_RESOLVED"
        conflict.preferred_claim_id = verified_in_group[0].claim_id
        conflict.resolution_policy = "EXTERNAL_VERIFICATION_WINS"
        conflict.resolution_evidence = {"verified_claim_ids": [claim.claim_id for claim in verified_in_group]}
        return conflict

    top = max(group, key=lambda claim: claim.source_confidence)
    runner_up = sorted(group, key=lambda claim: claim.source_confidence, reverse=True)[1]
    if top.source_confidence - runner_up.source_confidence >= 0.1:
        conflict.resolution_status = "AUTO_RESOLVED"
        conflict.preferred_claim_id = top.claim_id
        conflict.resolution_policy = "HIGHER_SOURCE_CONFIDENCE"
        conflict.resolution_evidence = {
            "preferred_confidence": top.source_confidence,
            "runner_up_confidence": runner_up.source_confidence,
        }
        return conflict

    conflict.resolution_status = "OPEN"
    conflict.resolution_policy = "MANUAL_REVIEW"
    conflict.resolution_evidence = {"reason": "CONFIDENCE_TOO_CLOSE"}
    return conflict


__all__ = [
    "CONFLICT_STATUSES",
    "KnowledgeConflict",
    "RESOLUTION_POLICIES",
    "auto_resolve",
    "conflict_key_of",
    "detect_conflicts",
]

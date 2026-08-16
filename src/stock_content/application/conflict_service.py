"""知识冲突服务（详细修改方案 §5 P1-4）。"""
from __future__ import annotations

from typing import Any

from stock_content.domain.claims import FinancialClaim
from stock_content.domain.conflict import (
    CONFLICT_STATUSES,
    KnowledgeConflict,
    auto_resolve,
    detect_conflicts,
)


class ConflictService:
    """冲突登记、自动解析与人工解析（内存实现）。"""

    def __init__(self) -> None:
        self._claims: dict[str, FinancialClaim] = {}
        self._conflicts: dict[str, KnowledgeConflict] = {}

    def register_claims(
        self,
        claims: list[FinancialClaim],
        *,
        verified_claim_ids: set[str] | None = None,
        window_days: int = 30,
    ) -> list[KnowledgeConflict]:
        """登记新 claim 并检测冲突；冲突不会被静默覆盖。"""
        for claim in claims:
            self._claims[claim.claim_id] = claim
        created: list[KnowledgeConflict] = []
        for group in detect_conflicts(list(self._claims.values()), window_days=window_days):
            conflict = auto_resolve(group, verified_claim_ids=verified_claim_ids)
            existing = self._conflicts.get(conflict.conflict_id)
            if existing is None:
                self._conflicts[conflict.conflict_id] = conflict
                created.append(conflict)
            elif existing.resolution_status == "OPEN" and conflict.resolution_status != "OPEN":
                # 新的证据可以把 OPEN 冲突推进为自动解析，但不允许降级。
                self._conflicts[conflict.conflict_id] = conflict
                created.append(conflict)
        return created

    def manual_resolve(
        self,
        conflict_id: str,
        preferred_claim_id: str,
        *,
        resolution_evidence: dict[str, Any] | None = None,
    ) -> KnowledgeConflict:
        conflict = self._conflicts.get(conflict_id)
        if conflict is None:
            raise KeyError(conflict_id)
        if conflict.resolution_status in {"MANUAL_RESOLVED", "SUPERSEDED"}:
            raise ValueError(f"conflict already resolved: {conflict.resolution_status}")
        if preferred_claim_id not in conflict.claim_ids:
            raise ValueError("preferred_claim_id must be part of the conflict")
        conflict.resolution_status = "MANUAL_RESOLVED"
        conflict.preferred_claim_id = preferred_claim_id
        conflict.resolution_policy = "MANUAL_REVIEW"
        conflict.resolution_evidence = dict(resolution_evidence or {})
        return conflict

    def supersede(self, conflict_id: str, *, reason: str = "") -> KnowledgeConflict:
        conflict = self._conflicts.get(conflict_id)
        if conflict is None:
            raise KeyError(conflict_id)
        conflict.resolution_status = "SUPERSEDED"
        conflict.resolution_evidence = {**conflict.resolution_evidence, "superseded_reason": reason}
        return conflict

    def get(self, conflict_id: str) -> KnowledgeConflict | None:
        return self._conflicts.get(conflict_id)

    def list_conflicts(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in CONFLICT_STATUSES:
            raise ValueError(f"unknown conflict status: {status}")
        items = [
            conflict
            for conflict in self._conflicts.values()
            if status is None or conflict.resolution_status == status
        ]
        return [conflict.to_dict() for conflict in sorted(items, key=lambda item: item.created_at)]


__all__ = ["ConflictService"]

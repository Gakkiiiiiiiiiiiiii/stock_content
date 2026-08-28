"""Lifecycle state transitions; business-time parsing is intentionally absent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .lifecycle_event import KnowledgeLifecycleEvent, LifecycleTargetType


@dataclass(frozen=True)
class LifecycleEvaluation:
    current_status: str
    action: str
    reason_code: str


class LifecyclePolicy:
    def evaluate(
        self, *, claim_type: str, current_status: str, now: datetime | None = None, target_expired: bool = False
    ) -> LifecycleEvaluation:
        # LifecyclePolicy does not parse a business date.  The caller supplies
        # the already-resolved ``target_expired`` fact (if any).
        if current_status in {"REJECTED", "RETIRED", "RETRACTED", "INVALID", "SUPERSEDED"}:
            return LifecycleEvaluation(current_status, "KEEP", "TERMINAL")
        if target_expired and claim_type == "FORECAST" and current_status == "ACTIVE":
            return LifecycleEvaluation(current_status, "OUTCOME_REVIEW", "FORECAST_TARGET_DATE_REACHED")
        return LifecycleEvaluation(current_status, "KEEP", "WITHIN_POLICY")

    def outcome_review_evaluation(self, verification_status: str) -> LifecycleEvaluation:
        """Map an external verification result to the next lifecycle state.

        The policy only maps an already-produced verification status.  It does
        not fetch truth, invoke a model, or decide whether the status is valid.
        """
        status = str(verification_status).upper()
        if status not in OUTCOME_REVIEW_OUTCOMES:
            raise ValueError(f"unsupported outcome review status: {verification_status}")
        # VerificationResult uses VERIFIED for a successful external check;
        # lifecycle vocabulary names the corresponding terminal state
        # VALIDATED.  This is a deterministic vocabulary mapping only.
        status = "VALIDATED" if status == "VERIFIED" else status
        return LifecycleEvaluation(
            "OUTCOME_REVIEW",
            status,
            f"OUTCOME_REVIEW_RESULT:{status}",
        )

    # Short alias for callers that treat this as a status mapping operation.
    map_outcome = outcome_review_evaluation

    def outcome_review_event(
        self,
        *,
        claim_id: str,
        verification_status: str,
        effective_at: datetime,
        recorded_at: datetime,
        policy_version: str = "lifecycle.v1",
    ) -> KnowledgeLifecycleEvent:
        evaluation = self.outcome_review_evaluation(verification_status)
        return KnowledgeLifecycleEvent(
            target_type=LifecycleTargetType.CLAIM,
            target_id=claim_id,
            from_status=evaluation.current_status,
            to_status=evaluation.action,
            effective_at=effective_at,
            recorded_at=recorded_at,
            reason_code=evaluation.reason_code,
            policy_version=policy_version,
        )


OUTCOME_REVIEW_OUTCOMES = frozenset(
    {"VALIDATED", "VERIFIED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE"}
)


__all__ = ["LifecyclePolicy", "LifecycleEvaluation", "OUTCOME_REVIEW_OUTCOMES"]

"""Deterministic forecast target expiry and outcome-review lifecycle use case.

This module does not schedule itself, ask an LLM to interpret dates, or obtain
external truth for an outcome.  Once a caller supplies an already completed
external verification result, it can materialize the immutable verification /
lifecycle artifacts and a child content snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.artifacts import (
    ClaimArtifact,
    LifecycleArtifact,
    VerificationArtifact,
    artifact_id_of,
)
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.lifecycle_event import (
    KnowledgeLifecycleEvent,
    LifecycleTargetType,
    select_lifecycle_event,
)
from stock_content.domain.lifecycle_policy import LifecyclePolicy
from stock_content.domain.temporal_semantics import ClaimTemporalBinding, TemporalRole


@dataclass(frozen=True)
class ForecastOutcomeReviewResult:
    claim_id: str
    transitioned: bool
    lifecycle_event: KnowledgeLifecycleEvent | None
    reason_code: str
    verification_artifact: VerificationArtifact | None = None
    lifecycle_artifact: LifecycleArtifact | None = None
    snapshot: Any | None = None


class ForecastOutcomeReview:
    """Project an expired canonical forecast claim into ``OUTCOME_REVIEW``.

    ``evaluation_time`` is the caller's wall-clock/knowledge time.  The target
    date is read only from a normalized ``FORECAST_TARGET`` binding, while the
    current status is read through the claim-scoped bitemporal lifecycle
    selector.  Therefore an occurrence event can never affect this use case.
    """

    def __init__(
        self,
        claim_repository: Any,
        lifecycle_repository: Any,
        *,
        policy: LifecyclePolicy | None = None,
        policy_version: str = "lifecycle.v1",
        artifact_repository: Any | None = None,
        artifact_store: Any | None = None,
        snapshot_service: SnapshotService | None = None,
    ) -> None:
        self._claims = claim_repository
        self._lifecycle = lifecycle_repository
        self._policy = policy or LifecyclePolicy()
        self._policy_version = policy_version
        self._artifacts = artifact_repository or artifact_store
        self._snapshots = snapshot_service

    def review_claim(
        self,
        claim_id: str,
        *,
        evaluation_time: datetime,
        recorded_at: datetime | None = None,
    ) -> ForecastOutcomeReviewResult:
        claim = self._claims.get(claim_id)
        if claim is None:
            return self._skipped(claim_id, "CLAIM_NOT_FOUND")
        if not isinstance(claim, FinancialClaim):
            # Repository adapters are expected to return the canonical model;
            # fail closed if a compatibility adapter returns another object.
            return self._skipped(claim_id, "CLAIM_NOT_CANONICAL")
        if claim.claim_type != "FORECAST":
            return self._skipped(claim_id, "NOT_FORECAST")

        targets = _forecast_target_bindings(claim)
        target_end = _select_target_end(targets, evaluation_time)
        if target_end is None:
            return self._skipped(claim_id, "FORECAST_TARGET_UNRESOLVED")
        if not _instant_before(target_end, evaluation_time):
            return self._skipped(claim_id, "FORECAST_TARGET_NOT_REACHED")

        current = self._lifecycle.select_as_of(
            target_type=LifecycleTargetType.CLAIM,
            target_id=claim.claim_id,
            business_as_of=evaluation_time,
            knowledge_as_of=evaluation_time,
        )
        if current is None:
            return self._skipped(claim_id, "CLAIM_LIFECYCLE_NOT_FOUND")
        if current.to_status != "ACTIVE":
            return self._skipped(claim_id, "CLAIM_NOT_ACTIVE")

        decision = self._policy.evaluate(
            claim_type=claim.claim_type,
            current_status=current.to_status,
            target_expired=True,
        )
        if decision.action != "OUTCOME_REVIEW":
            return self._skipped(claim_id, decision.reason_code)

        event = KnowledgeLifecycleEvent(
            target_type=LifecycleTargetType.CLAIM,
            target_id=claim.claim_id,
            from_status="ACTIVE",
            to_status="OUTCOME_REVIEW",
            effective_at=target_end,
            recorded_at=recorded_at or evaluation_time,
            reason_code="FORECAST_TARGET_DATE_REACHED",
            policy_version=self._policy_version,
        )

        # The event identity excludes recorded_at by design.  Reading it first
        # makes repeated runs return the original immutable insert instead of
        # attempting an update (or manufacturing a second event).
        existing = _get_event(self._lifecycle, event.lifecycle_event_id)
        if existing is not None:
            return ForecastOutcomeReviewResult(
                claim_id=claim.claim_id,
                transitioned=True,
                lifecycle_event=existing,
                reason_code="ALREADY_OUTCOME_REVIEW",
            )
        self._lifecycle.append(event)
        return ForecastOutcomeReviewResult(
            claim_id=claim.claim_id,
            transitioned=True,
            lifecycle_event=event,
            reason_code=event.reason_code,
        )

    # Names used by different application adapters all retain the same
    # deterministic operation; there is no scheduler hidden behind them.
    evaluate_claim = review_claim
    run = review_claim

    def complete_review(
        self,
        claim_id: str,
        verification_result: VerificationResult | dict[str, Any],
        *,
        parent_snapshot_id: str | None = None,
        parent_snapshot: Any | None = None,
        evaluation_time: datetime,
        recorded_at: datetime | None = None,
        reference_available_at: datetime | None = None,
    ) -> ForecastOutcomeReviewResult:
        """Materialize a completed external outcome review as a child snapshot.

        This operation only consumes the caller-provided verification result;
        it never decides external truth.  Artifact repositories are append-only
        and ``SnapshotService`` preserves the complete parent slot mapping,
        replacing only verification/lifecycle with content-addressed children.
        If the supplied adapters do not expose a shared transaction, a failure
        after an artifact insert can leave an unreferenced immutable artifact;
        the operation therefore does not claim cross-store atomicity.
        """
        if self._artifacts is None or self._snapshots is None:
            raise ValueError("artifact_repository and snapshot_service are required for completion")
        result = VerificationResult.model_validate(verification_result)
        if result.claim_id != claim_id:
            raise ValueError("verification result claim_id does not match requested claim")
        claim = self._claims.get(claim_id)
        if not isinstance(claim, FinancialClaim):
            raise ValueError("canonical claim is required for outcome review completion")

        parent = parent_snapshot
        if parent is None:
            if not parent_snapshot_id:
                raise ValueError("parent_snapshot_id is required for outcome review completion")
            parent = self._snapshots.get(parent_snapshot_id)
        if parent is None:
            raise KeyError("parent content snapshot not found")
        parent_id = str(parent.content_snapshot_id)
        if parent_snapshot_id and parent_snapshot_id != parent_id:
            raise ValueError("parent snapshot id does not match supplied parent snapshot")

        # A parent snapshot is a historical closure, not a hint to consult the
        # global latest lifecycle state.  Resolve only the fixed artifact IDs
        # and their own bitemporal as-of coordinates.  This prevents a newer
        # snapshot (for the same claim or another branch) from contaminating a
        # review rooted at an older snapshot.
        claims_artifact_id = str((parent.artifact_ids or {}).get("claims") or "")
        lifecycle_artifact_id = str((parent.artifact_ids or {}).get("lifecycle") or "")
        if not claims_artifact_id or not lifecycle_artifact_id:
            raise ValueError("parent snapshot claims and lifecycle slots are required")
        claims_artifact = self._artifact_or_none(claims_artifact_id)
        old_lifecycle = self._artifact_or_none(lifecycle_artifact_id)
        if not isinstance(claims_artifact, ClaimArtifact):
            raise ValueError("parent claims slot is not a ClaimArtifact")
        if not any(str(getattr(item, "claim_id", item)) == claim_id for item in claims_artifact.claims):
            raise ValueError("claim is not present in parent ClaimArtifact")
        lifecycle_business_as_of = getattr(old_lifecycle, "lifecycle_business_as_of", None)
        lifecycle_knowledge_as_of = getattr(old_lifecycle, "lifecycle_knowledge_as_of", None)
        parent_claim_event_ids = list(getattr(old_lifecycle, "claim_lifecycle_event_ids", ()) or ())
        if lifecycle_business_as_of is None or lifecycle_knowledge_as_of is None or not parent_claim_event_ids:
            raise ValueError("parent LifecycleArtifact has no fixed claim lifecycle closure")
        parent_events = []
        for event_id in parent_claim_event_ids:
            event = _get_event(self._lifecycle, str(event_id))
            if event is None or event.lifecycle_event_id != str(event_id):
                raise ValueError(f"parent LifecycleArtifact event is missing or mismatched: {event_id}")
            parent_events.append(event)
        current = select_lifecycle_event(
            parent_events,
            target_type=LifecycleTargetType.CLAIM,
            target_id=claim_id,
            business_as_of=lifecycle_business_as_of,
            knowledge_as_of=lifecycle_knowledge_as_of,
        )
        if current is None or current.to_status != "OUTCOME_REVIEW":
            raise ValueError("parent lifecycle closure must select OUTCOME_REVIEW")

        old_verification_id = str((parent.artifact_ids or {}).get("verification") or "")
        old_lifecycle_id = lifecycle_artifact_id
        old_verification = self._artifact_or_none(old_verification_id)
        if not isinstance(old_verification, VerificationArtifact):
            raise ValueError("parent verification slot is not a VerificationArtifact")
        if not isinstance(old_lifecycle, LifecycleArtifact):
            raise ValueError("parent lifecycle slot is not a LifecycleArtifact")
        if old_verification.claim_artifact_id != claims_artifact_id:
            raise ValueError("parent verification claim_artifact_id does not match claims slot")
        prior_results = list(getattr(old_verification, "results", ()) or ())
        merged_results = [item for item in prior_results if getattr(item, "claim_id", None) != claim_id]
        merged_results.append(result)
        verification = VerificationArtifact(
            artifact_id="verification-outcome-review-pending",
            artifact_type="verification",
            producer_stage="forecast_outcome_review",
            claim_artifact_id=str(
                getattr(old_verification, "claim_artifact_id", "")
                or (parent.artifact_ids or {}).get("claims", "")
            ),
            results=merged_results,
            parent_artifact_ids=(old_verification_id,) if old_verification_id else (),
        )
        verification = VerificationArtifact(**{**verification.__dict__, "artifact_id": artifact_id_of(verification)})

        effective_at = result.verification_timestamp or evaluation_time
        lifecycle_event = self._policy.outcome_review_event(
            claim_id=claim_id,
            verification_status=result.status,
            effective_at=effective_at,
            recorded_at=recorded_at or evaluation_time,
            policy_version=self._policy_version,
        )
        existing_event = _get_event(self._lifecycle, lifecycle_event.lifecycle_event_id)
        lifecycle_event = existing_event or lifecycle_event
        prior_claim_events = list(getattr(old_lifecycle, "claim_lifecycle_event_ids", ()) or ())
        prior_occurrence_events = list(getattr(old_lifecycle, "occurrence_lifecycle_event_ids", ()) or ())
        if lifecycle_event.lifecycle_event_id not in prior_claim_events:
            prior_claim_events.append(lifecycle_event.lifecycle_event_id)
        lifecycle = LifecycleArtifact(
            artifact_id="lifecycle-outcome-review-pending",
            artifact_type="lifecycle",
            producer_stage="forecast_outcome_review",
            claim_lifecycle_event_ids=prior_claim_events,
            occurrence_lifecycle_event_ids=prior_occurrence_events,
            lifecycle_business_as_of=evaluation_time,
            lifecycle_knowledge_as_of=recorded_at or evaluation_time,
            policy_version=self._policy_version,
            # The lifecycle artifact is a projection of the new verification
            # result as well as the prior lifecycle history.  Sort the two
            # immutable references so ID generation is input-order invariant.
            parent_artifact_ids=tuple(
                sorted({item for item in (old_lifecycle_id, verification.artifact_id) if item})
            ),
        )
        lifecycle = LifecycleArtifact(**{**lifecycle.__dict__, "artifact_id": artifact_id_of(lifecycle)})

        artifact_ids = dict(parent.artifact_ids or {})
        artifact_ids["verification"] = verification.artifact_id
        artifact_ids["lifecycle"] = lifecycle.artifact_id
        external = list(parent.external_snapshots or parent.quant_market_snapshot_ids or ())
        if result.market_snapshot_id and result.market_snapshot_id not in external:
            external.append(result.market_snapshot_id)
        child_created_at = recorded_at or evaluation_time
        reference_data = None
        if result.market_snapshot_id or result.market_data_version:
            if not result.market_snapshot_id or not result.market_data_version:
                raise ValueError("market snapshot binding requires snapshot_id and data_version")
            if reference_available_at is None:
                raise ValueError("market snapshot binding requires reference_available_at")
            reference_data = {
                "snapshot_id": result.market_snapshot_id,
                "data_version": result.market_data_version,
                "available_at": reference_available_at.isoformat(),
            }
        for dependency_name, dependency_at in (
            ("verification_timestamp", result.verification_timestamp),
            ("reference_available_at", reference_available_at),
        ):
            if dependency_at is not None and not _instant_at_or_before(dependency_at, child_created_at):
                raise ValueError(f"{dependency_name} must be no later than child snapshot created_at")
        model_versions = dict(parent.model_versions or {})
        model_versions.setdefault("parser_version", parent.parser_version)
        model_versions.setdefault("asr_model", parent.asr_model)
        model_versions.setdefault("asr_model_version", parent.asr_model_version)
        model_versions.setdefault("vision_model", parent.vision_model)
        model_versions.setdefault("llm_model", parent.llm_model)
        producer_manifest = dict(parent.producer_manifest or {})
        if reference_data is not None:
            producer_manifest["reference_data"] = reference_data
        child = self._snapshots.record_from_artifacts(
            source_type=parent.source_type,
            source_ref=parent.source_ref,
            source_content_hash=parent.source_content_hash,
            source_artifact_id=parent.source_artifact_id,
            artifact_ids=artifact_ids,
            model_versions=model_versions,
            prompt_versions=parent.prompt_versions,
            producer_manifest=producer_manifest,
            configuration=parent.configuration,
            external_snapshots=tuple(external),
            quant_market_snapshot_ids=tuple(external),
            policy_versions=parent.policy_versions,
            code_sha=parent.code_sha,
            config_hash=parent.config_hash,
            pipeline_version=parent.pipeline_version,
            snapshot_kind="OUTCOME_REVIEW",
            parent_snapshot_id=parent_id,
            supersedes_snapshot_id=parent_id,
            created_at=child_created_at,
            _persist=False,
        )

        # Persist only after every immutable child has been fully built.  A
        # caller with a shared transaction-capable adapter may wrap these calls
        # externally; the base protocols intentionally remain backward-safe.
        self._artifacts.put(verification)
        self._artifacts.put(lifecycle)
        if existing_event is None:
            self._lifecycle.append(lifecycle_event)
        # Replaying the same deterministic input returns the existing snapshot
        # from SnapshotService without changing its historical parent.
        child = _save_snapshot(self._snapshots, child)
        return ForecastOutcomeReviewResult(
            claim_id=claim_id,
            transitioned=True,
            lifecycle_event=lifecycle_event,
            reason_code=f"OUTCOME_REVIEW_COMPLETED:{result.status}",
            verification_artifact=verification,
            lifecycle_artifact=lifecycle,
            snapshot=child,
        )

    complete = complete_review
    finalize = complete_review
    complete_outcome_review = complete_review

    def _artifact_or_none(self, artifact_id: str) -> Any | None:
        if not artifact_id:
            return None
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise KeyError(f"parent artifact not found: {artifact_id}")
        return artifact

    @staticmethod
    def _skipped(claim_id: str, reason_code: str) -> ForecastOutcomeReviewResult:
        return ForecastOutcomeReviewResult(
            claim_id=claim_id,
            transitioned=False,
            lifecycle_event=None,
            reason_code=reason_code,
        )


def _forecast_target_bindings(claim: FinancialClaim) -> list[ClaimTemporalBinding]:
    """Return actionable targets in an order-independent input order."""
    return [
        binding
        for binding in claim.temporal_bindings
        if binding.role is TemporalRole.FORECAST_TARGET
        and str(binding.normalization_status).upper() not in {"PARTIAL", "UNRESOLVED"}
        and (binding.end_date is not None or binding.end_time is not None)
    ]


def _select_target_end(
    bindings: list[ClaimTemporalBinding], evaluation_time: datetime
) -> datetime | None:
    candidates = [
        resolved
        for binding in bindings
        if (resolved := _resolved_target_end(binding, evaluation_time)) is not None
    ]
    # Sorting by the resolved instant makes multiple target bindings
    # deterministic regardless of the order emitted by an upstream adapter.
    return min(candidates, key=lambda value: (_as_utc(value), value.isoformat()), default=None)


def _resolved_target_end(binding: ClaimTemporalBinding, evaluation_time: datetime) -> datetime | None:
    if binding.end_time is not None:
        return binding.end_time
    if binding.end_date is not None:
        # A DATE target remains in force through the stated date.  Its
        # deterministic expiry instant is the end of that calendar day.  A
        # declared IANA timezone is authoritative; an absent timezone uses UTC
        # so the result cannot depend on the process or evaluation timezone.
        timezone = _declared_timezone(binding.timezone)
        if timezone is None:
            return None
        return datetime.combine(binding.end_date, time.max, tzinfo=timezone)
    return None


def _instant_before(left: datetime, right: datetime) -> bool:
    return _as_utc(left) < _as_utc(right)


def _instant_at_or_before(left: datetime, right: datetime) -> bool:
    return _as_utc(left) <= _as_utc(right)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _declared_timezone(timezone_name: str | None):
    if not timezone_name:
        return UTC
    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        # Windows deployments may omit the optional system tzdata package.
        # UTC has no DST rules and is safe to recognize without guessing an
        # offset; every other unavailable/invalid IANA key fails closed.
        if timezone_name in {"UTC", "Etc/UTC", "GMT"}:
            return UTC
        return None


def _get_event(repository: Any, event_id: str) -> KnowledgeLifecycleEvent | None:
    getter = getattr(repository, "get", None)
    if getter is None:
        return None
    return getter(event_id)


def _save_snapshot(snapshot_service: SnapshotService, snapshot: Any) -> Any:
    store = getattr(snapshot_service, "_store", None)
    saver = getattr(store, "save", None)
    if saver is None:
        raise ValueError("snapshot service store does not support immutable save")
    return saver(snapshot)


ForecastOutcomeReviewService = ForecastOutcomeReview


__all__ = ["ForecastOutcomeReview", "ForecastOutcomeReviewService", "ForecastOutcomeReviewResult"]

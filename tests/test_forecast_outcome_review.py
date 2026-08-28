from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from stock_content.application.forecast_outcome_review import ForecastOutcomeReview
from stock_content.application.snapshot_service import InMemorySnapshotStore, SnapshotService
from stock_content.domain.artifacts import (
    ArtifactBase,
    ClaimArtifact,
    LifecycleArtifact,
    VerificationArtifact,
)
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.lifecycle_event import KnowledgeLifecycleEvent, LifecycleTargetType, select_lifecycle_event
from stock_content.domain.lifecycle_policy import LifecyclePolicy
from stock_content.domain.temporal_semantics import (
    ClaimTemporalBinding,
    TemporalPrecision,
    TemporalRole,
    TemporalScope,
    TemporalValueType,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


class FakeClaims:
    def __init__(self, *claims: FinancialClaim):
        self.items = {claim.claim_id: claim for claim in claims}

    def get(self, claim_id: str):
        return self.items.get(claim_id)


class FakeLifecycle:
    def __init__(self, *events: KnowledgeLifecycleEvent):
        self.events = list(events)
        self.appended: list[KnowledgeLifecycleEvent] = []

    def get(self, lifecycle_event_id: str):
        return next((event for event in self.events if event.lifecycle_event_id == lifecycle_event_id), None)

    def select_as_of(self, *, target_type, target_id, business_as_of, knowledge_as_of):
        return select_lifecycle_event(
            self.events,
            target_type=target_type,
            target_id=target_id,
            business_as_of=business_as_of,
            knowledge_as_of=knowledge_as_of,
        )

    def append(self, event):
        self.events.append(event)
        self.appended.append(event)
        return event


class FakeArtifacts:
    def __init__(self, *artifacts):
        self.items = {item.artifact_id: item for item in artifacts}

    def get(self, artifact_id):
        return self.items.get(artifact_id)

    def put(self, artifact):
        existing = self.items.get(artifact.artifact_id)
        if existing is not None and existing.content_hash != artifact.content_hash:
            raise ValueError("artifact id conflict")
        self.items[artifact.artifact_id] = artifact
        return artifact


def _binding(*, status: str = "NORMALIZED", end_date: date | None = None, end_time=None):
    return ClaimTemporalBinding(
        role=TemporalRole.FORECAST_TARGET,
        scope=TemporalScope.FORECAST,
        value_type=TemporalValueType.TIMESTAMP if end_time else TemporalValueType.DATE,
        end_time=end_time,
        end_date=end_date,
        precision=TemporalPrecision.EXACT if end_time else TemporalPrecision.DAY,
        normalization_status=status,
    )


def _claim(*bindings):
    return FinancialClaim(
        claim_type="FORECAST",
        subject_type="company",
        subject_id="acme",
        predicate="revenue",
        value=100,
        unit="CNY",
        source_confidence=0.9,
        extractor_confidence=0.9,
        evidence_refs=["ev-1"],
        temporal_bindings=list(bindings),
    )


def _active(claim_id: str, *, recorded_at: datetime = NOW - timedelta(days=2)):
    return KnowledgeLifecycleEvent(
        target_type=LifecycleTargetType.CLAIM,
        target_id=claim_id,
        from_status="EXTRACTED",
        to_status="ACTIVE",
        effective_at=NOW - timedelta(days=2),
        recorded_at=recorded_at,
        reason_code="ACTIVATED",
        policy_version="lifecycle.v1",
    )


def test_expired_normalized_forecast_creates_claim_scoped_event_with_bitemporal_times():
    claim = _claim(_binding(end_date=NOW.date() - timedelta(days=1)))
    lifecycle = FakeLifecycle(_active(claim.claim_id))
    result = ForecastOutcomeReview(FakeClaims(claim), lifecycle).review_claim(
        claim.claim_id,
        evaluation_time=NOW,
        recorded_at=NOW + timedelta(minutes=3),
    )

    assert result.transitioned is True
    assert result.lifecycle_event is not None
    assert result.lifecycle_event.target_type is LifecycleTargetType.CLAIM
    assert result.lifecycle_event.target_id == claim.claim_id
    assert result.lifecycle_event.from_status == "ACTIVE"
    assert result.lifecycle_event.to_status == "OUTCOME_REVIEW"
    assert result.lifecycle_event.reason_code == "FORECAST_TARGET_DATE_REACHED"
    assert result.lifecycle_event.effective_at == datetime.combine(
        NOW.date() - timedelta(days=1), datetime.max.time(), tzinfo=UTC
    )
    assert result.lifecycle_event.recorded_at == NOW + timedelta(minutes=3)


def test_repeated_run_is_idempotent_and_does_not_update_recorded_at():
    claim = _claim(_binding(end_time=NOW - timedelta(hours=1)))
    lifecycle = FakeLifecycle(_active(claim.claim_id))
    use_case = ForecastOutcomeReview(FakeClaims(claim), lifecycle)
    first = use_case.review_claim(claim.claim_id, evaluation_time=NOW, recorded_at=NOW)
    second = use_case.review_claim(
        claim.claim_id,
        evaluation_time=NOW + timedelta(hours=1),
        recorded_at=NOW + timedelta(hours=1),
    )

    assert first.lifecycle_event is not None
    assert second.transitioned is False
    assert second.reason_code == "CLAIM_NOT_ACTIVE"
    assert len(lifecycle.appended) == 1
    assert lifecycle.appended[0].recorded_at == NOW


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        (_claim(_binding(status="PARTIAL", end_date=NOW.date() - timedelta(days=1))), "FORECAST_TARGET_UNRESOLVED"),
        (_claim(_binding(status="UNRESOLVED", end_date=NOW.date() - timedelta(days=1))), "FORECAST_TARGET_UNRESOLVED"),
        (_claim(_binding()), "FORECAST_TARGET_UNRESOLVED"),
    ],
)
def test_unresolved_or_missing_target_does_not_transition(claim, reason):
    lifecycle = FakeLifecycle(_active(claim.claim_id))
    result = ForecastOutcomeReview(FakeClaims(claim), lifecycle).review_claim(
        claim.claim_id, evaluation_time=NOW
    )
    assert result.transitioned is False
    assert result.reason_code == reason
    assert lifecycle.appended == []


def test_not_yet_reached_non_forecast_terminal_and_missing_claim_are_untouched():
    future = _claim(_binding(end_time=NOW + timedelta(hours=1)))
    ordinary = FinancialClaim(
        claim_type="OPINION",
        subject_type="company",
        subject_id="acme",
        predicate="quality",
        value="good",
        source_confidence=0.9,
        extractor_confidence=0.9,
        evidence_refs=["ev-2"],
    )
    lifecycle = FakeLifecycle(_active(future.claim_id), _active(ordinary.claim_id))
    use_case = ForecastOutcomeReview(FakeClaims(future, ordinary), lifecycle)
    assert use_case.review_claim(future.claim_id, evaluation_time=NOW).reason_code == "FORECAST_TARGET_NOT_REACHED"
    assert use_case.review_claim(ordinary.claim_id, evaluation_time=NOW).reason_code == "NOT_FORECAST"
    assert use_case.review_claim("missing", evaluation_time=NOW).reason_code == "CLAIM_NOT_FOUND"

    terminal = _claim(_binding(end_time=NOW - timedelta(hours=1)))
    terminal_event = _active(terminal.claim_id).model_copy(update={"to_status": "RETRACTED"})
    terminal_lifecycle = FakeLifecycle(terminal_event)
    assert ForecastOutcomeReview(FakeClaims(terminal), terminal_lifecycle).review_claim(
        terminal.claim_id, evaluation_time=NOW
    ).reason_code == "CLAIM_NOT_ACTIVE"
    assert terminal_lifecycle.appended == []


def test_occurrence_event_cannot_supply_claim_status():
    claim = _claim(_binding(end_time=NOW - timedelta(hours=1)))
    occurrence_event = _active(claim.claim_id).model_copy(
        update={"target_type": LifecycleTargetType.OCCURRENCE, "to_status": "ACTIVE"}
    )
    lifecycle = FakeLifecycle(occurrence_event)
    result = ForecastOutcomeReview(FakeClaims(claim), lifecycle).review_claim(
        claim.claim_id, evaluation_time=NOW
    )
    assert result.transitioned is False
    assert result.reason_code == "CLAIM_LIFECYCLE_NOT_FOUND"


def test_date_target_uses_declared_timezone_and_fails_closed_for_invalid_timezone():
    tokyo_target = ClaimTemporalBinding(
        role=TemporalRole.FORECAST_TARGET,
        scope=TemporalScope.FORECAST,
        value_type=TemporalValueType.DATE,
        end_date=date(2026, 8, 26),
        precision=TemporalPrecision.DAY,
        timezone="UTC",
        normalization_status="NORMALIZED",
    )
    claim = _claim(tokyo_target)
    lifecycle = FakeLifecycle(_active(claim.claim_id))
    use_case = ForecastOutcomeReview(FakeClaims(claim), lifecycle)
    at_expiry = datetime(2026, 8, 26, 23, 59, 59, 999999, tzinfo=UTC)
    assert use_case.review_claim(claim.claim_id, evaluation_time=at_expiry).transitioned is False
    assert use_case.review_claim(
        claim.claim_id, evaluation_time=at_expiry + timedelta(microseconds=1)
    ).transitioned is True

    invalid = _claim(tokyo_target.model_copy(update={"timezone": "Not/AZone"}))
    invalid_lifecycle = FakeLifecycle(_active(invalid.claim_id))
    invalid_result = ForecastOutcomeReview(FakeClaims(invalid), invalid_lifecycle).review_claim(
        invalid.claim_id, evaluation_time=NOW
    )
    assert invalid_result.transitioned is False
    assert invalid_result.reason_code == "FORECAST_TARGET_UNRESOLVED"


def test_multiple_targets_are_order_independent_and_choose_earliest_resolved_end():
    early = _binding(end_time=NOW - timedelta(hours=2))
    late = _binding(end_time=NOW + timedelta(hours=2))
    first = _claim(early, late)
    second = _claim(late, early)
    assert first.claim_id == second.claim_id
    first_lifecycle = FakeLifecycle(_active(first.claim_id))
    second_lifecycle = FakeLifecycle(_active(second.claim_id))
    first_result = ForecastOutcomeReview(FakeClaims(first), first_lifecycle).review_claim(
        first.claim_id, evaluation_time=NOW
    )
    second_result = ForecastOutcomeReview(FakeClaims(second), second_lifecycle).review_claim(
        second.claim_id, evaluation_time=NOW
    )
    assert first_result.lifecycle_event is not None
    assert second_result.lifecycle_event is not None
    assert first_result.lifecycle_event.effective_at == second_result.lifecycle_event.effective_at


def test_policy_maps_only_supported_outcome_review_results_to_events():
    policy = LifecyclePolicy()
    for status in ("VALIDATED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE"):
        evaluation = policy.map_outcome(status)
        assert evaluation.current_status == "OUTCOME_REVIEW"
        assert evaluation.action == status
        event = policy.outcome_review_event(
            claim_id="claim-1",
            verification_status=status,
            effective_at=NOW,
            recorded_at=NOW,
        )
        assert event.from_status == "OUTCOME_REVIEW"
        assert event.to_status == status
    with pytest.raises(ValueError, match="unsupported outcome review status"):
        policy.map_outcome("ACTIVE")


def test_policy_never_transitions_terminal_forecast_even_when_caller_says_expired():
    decision = LifecyclePolicy().evaluate(
        claim_type="FORECAST", current_status="RETRACTED", target_expired=True
    )
    assert decision.action == "KEEP"


def _review_fixture():
    claim = _claim(_binding(end_time=NOW - timedelta(hours=1)))
    lifecycle = FakeLifecycle(_active(claim.claim_id))
    review = ForecastOutcomeReview(FakeClaims(claim), lifecycle)
    expiry = review.review_claim(claim.claim_id, evaluation_time=NOW, recorded_at=NOW)
    old_claims = ClaimArtifact(artifact_id="claims-parent", artifact_type="claims", claims=[claim.claim_id])
    old_verification = VerificationArtifact(
        artifact_id="verification-parent",
        artifact_type="verification",
        claim_artifact_id=old_claims.artifact_id,
        results=[],
        parent_artifact_ids=(old_claims.artifact_id,),
    )
    old_lifecycle = LifecycleArtifact(
        artifact_id="lifecycle-parent",
        artifact_type="lifecycle",
        claim_lifecycle_event_ids=[expiry.lifecycle_event.lifecycle_event_id],
        lifecycle_business_as_of=NOW,
        lifecycle_knowledge_as_of=NOW,
        policy_version="lifecycle.v1",
    )
    source = ArtifactBase(artifact_id="source-parent", artifact_type="source")
    artifacts = FakeArtifacts(source, old_claims, old_verification, old_lifecycle)
    snapshots = SnapshotService(InMemorySnapshotStore())
    parent = snapshots.record_from_artifacts(
        source_type="fixture",
        source_ref="forecast-review",
        source_content_hash="source-hash",
        source_artifact_id=source.artifact_id,
        artifact_ids={
            "source": source.artifact_id,
            "claims": old_claims.artifact_id,
            "verification": old_verification.artifact_id,
            "lifecycle": old_lifecycle.artifact_id,
        },
        code_sha="test-sha",
        created_at=NOW - timedelta(days=1),
    )
    completed = ForecastOutcomeReview(
        FakeClaims(claim), lifecycle, artifact_repository=artifacts, snapshot_service=snapshots
    )
    return claim, lifecycle, artifacts, snapshots, parent, completed


@pytest.mark.parametrize("status", ["VERIFIED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE"])
def test_completed_review_creates_child_artifacts_and_preserves_parent(status):
    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    parent_payload = parent.to_dict()
    result = VerificationResult(
        claim_id=claim.claim_id,
        status=status,
        market_snapshot_id="market-1" if status != "NOT_VERIFIABLE" else None,
        market_data_version="bars.v1" if status != "NOT_VERIFIABLE" else None,
        fact_date=NOW.date() if status != "NOT_VERIFIABLE" else None,
        verification_timestamp=NOW if status != "NOT_VERIFIABLE" else None,
    )
    completed = operation.complete_review(
        claim.claim_id,
        result,
        parent_snapshot_id=parent.content_snapshot_id,
        evaluation_time=NOW + timedelta(minutes=5),
        recorded_at=NOW + timedelta(minutes=6),
        reference_available_at=NOW + timedelta(minutes=4),
    )

    assert completed.snapshot is not None
    child = completed.snapshot
    assert child.content_snapshot_id != parent.content_snapshot_id
    assert child.parent_snapshot_id == parent.content_snapshot_id
    assert child.supersedes_snapshot_id == parent.content_snapshot_id
    assert child.artifact_ids["source"] == parent.artifact_ids["source"]
    assert child.artifact_ids["claims"] == parent.artifact_ids["claims"]
    assert child.artifact_ids["verification"] == completed.verification_artifact.artifact_id
    assert child.artifact_ids["lifecycle"] == completed.lifecycle_artifact.artifact_id
    assert completed.verification_artifact.results[-1].status == status
    assert completed.lifecycle_event.to_status == ("VALIDATED" if status == "VERIFIED" else status)
    assert set(completed.lifecycle_artifact.parent_artifact_ids) == {
        parent.artifact_ids["lifecycle"], completed.verification_artifact.artifact_id
    }
    if status != "NOT_VERIFIABLE":
        assert child.producer_manifest["reference_data"] == {
            "snapshot_id": "market-1",
            "data_version": "bars.v1",
            "available_at": (NOW + timedelta(minutes=4)).isoformat(),
        }
    assert parent.to_dict() == parent_payload
    assert snapshots.get(parent.content_snapshot_id).to_dict() == parent_payload
    assert artifacts.get(completed.verification_artifact.artifact_id) is not None
    assert artifacts.get(completed.lifecycle_artifact.artifact_id) is not None


def test_completed_review_is_idempotent_and_does_not_mix_latest_into_parent():
    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    result = VerificationResult(
        claim_id=claim.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-1",
        market_data_version="bars.v1",
        fact_date=NOW.date(),
        verification_timestamp=NOW,
    )
    first = operation.complete_review(
        claim.claim_id, result, parent_snapshot_id=parent.content_snapshot_id,
        evaluation_time=NOW + timedelta(minutes=5), recorded_at=NOW + timedelta(minutes=6),
        reference_available_at=NOW + timedelta(minutes=4),
    )
    second = operation.complete_review(
        claim.claim_id, result, parent_snapshot_id=parent.content_snapshot_id,
        evaluation_time=NOW + timedelta(minutes=5), recorded_at=NOW + timedelta(minutes=6),
        reference_available_at=NOW + timedelta(minutes=4),
    )
    assert first.snapshot.content_snapshot_id == second.snapshot.content_snapshot_id
    assert first.verification_artifact.artifact_id == second.verification_artifact.artifact_id
    assert first.lifecycle_artifact.artifact_id == second.lifecycle_artifact.artifact_id
    assert len([item for item in lifecycle.appended if item.to_status == "VALIDATED"]) == 1
    assert (
        snapshots.list_for_source("fixture", "forecast-review")[-1].content_snapshot_id
        == first.snapshot.content_snapshot_id
    )


def test_completion_rejects_non_outcome_review_and_does_not_persist_children():
    claim = _claim(_binding(end_time=NOW - timedelta(hours=1)))
    lifecycle = FakeLifecycle(_active(claim.claim_id))
    claims_artifact = ClaimArtifact(artifact_id="claims-active", artifact_type="claims", claims=[claim.claim_id])
    lifecycle_artifact = LifecycleArtifact(
        artifact_id="lifecycle-active",
        artifact_type="lifecycle",
        claim_lifecycle_event_ids=[lifecycle.events[0].lifecycle_event_id],
        lifecycle_business_as_of=NOW,
        lifecycle_knowledge_as_of=NOW,
        policy_version="lifecycle.v1",
    )
    source = ArtifactBase(artifact_id="source-active", artifact_type="source")
    artifacts = FakeArtifacts(source, claims_artifact, lifecycle_artifact)
    snapshots = SnapshotService(InMemorySnapshotStore())
    parent = snapshots.record_from_artifacts(
        source_type="fixture",
        source_ref="forecast-active",
        source_content_hash="source-hash",
        source_artifact_id=source.artifact_id,
        artifact_ids={
            "source": source.artifact_id,
            "claims": claims_artifact.artifact_id,
            "lifecycle": lifecycle_artifact.artifact_id,
        },
        code_sha="test-sha",
        created_at=NOW,
    )
    operation = ForecastOutcomeReview(
        FakeClaims(claim), lifecycle, artifact_repository=artifacts, snapshot_service=snapshots
    )
    with pytest.raises(ValueError, match="OUTCOME_REVIEW"):
        operation.complete_review(
            claim.claim_id,
            VerificationResult(
                claim_id=claim.claim_id,
                status="VERIFIED",
                market_snapshot_id="market-1",
                market_data_version="bars.v1",
                fact_date=NOW.date(),
                verification_timestamp=NOW,
            ),
            parent_snapshot_id=parent.content_snapshot_id, evaluation_time=NOW,
        )
    assert set(artifacts.items) == {source.artifact_id, claims_artifact.artifact_id, lifecycle_artifact.artifact_id}
    assert lifecycle.appended == []


def _valid_result(claim_id: str, *, verification_timestamp: datetime = NOW):
    return VerificationResult(
        claim_id=claim_id,
        status="VERIFIED",
        market_snapshot_id="market-1",
        market_data_version="bars.v1",
        fact_date=verification_timestamp.date(),
        verification_timestamp=verification_timestamp,
    )


def test_completion_uses_parent_closure_even_when_global_latest_is_validated():
    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    global_result_event = operation._policy.outcome_review_event(  # noqa: SLF001
        claim_id=claim.claim_id,
        verification_status="VERIFIED",
        effective_at=NOW + timedelta(minutes=1),
        recorded_at=NOW + timedelta(minutes=1),
    )
    lifecycle.events.append(global_result_event)
    completed = operation.complete_review(
        claim.claim_id,
        _valid_result(claim.claim_id, verification_timestamp=NOW + timedelta(minutes=1)),
        parent_snapshot_id=parent.content_snapshot_id,
        evaluation_time=NOW + timedelta(minutes=2),
        recorded_at=NOW + timedelta(minutes=2),
        reference_available_at=NOW + timedelta(minutes=1),
    )
    assert completed.lifecycle_event.lifecycle_event_id == global_result_event.lifecycle_event_id
    assert completed.snapshot is not None


def test_completion_fails_closed_for_parent_closure_errors_without_writes():
    # Each case starts from a distinct fixture, and all validation happens
    # before artifact/event/snapshot persistence.
    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    before_keys = set(artifacts.items)
    before_appends = len(lifecycle.appended)
    broken_claims = replace(artifacts.get(parent.artifact_ids["claims"]), claims=[])
    artifacts.items[broken_claims.artifact_id] = broken_claims
    with pytest.raises(ValueError, match="not present"):
        operation.complete_review(
            claim.claim_id, _valid_result(claim.claim_id),
            parent_snapshot_id=parent.content_snapshot_id, evaluation_time=NOW,
        )
    assert set(artifacts.items) == before_keys
    assert len(lifecycle.appended) == before_appends
    assert len(snapshots.list_for_source("fixture", "forecast-review")) == 1

    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    before_appends = len(lifecycle.appended)
    broken_parent = replace(parent, artifact_ids={"source": parent.artifact_ids["source"]})
    with pytest.raises(ValueError, match="claims and lifecycle slots"):
        operation.complete_review(
            claim.claim_id, _valid_result(claim.claim_id),
            parent_snapshot=broken_parent, evaluation_time=NOW,
        )
    assert len(lifecycle.appended) == before_appends
    assert len(snapshots.list_for_source("fixture", "forecast-review")) == 1


def test_completion_requires_every_parent_event_row_and_exact_artifact_types():
    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    # The artifact still declares the fixed event ID, but its row is absent.
    lifecycle.events.clear()
    with pytest.raises(ValueError, match="missing or mismatched"):
        operation.complete_review(
            claim.claim_id, _valid_result(claim.claim_id),
            parent_snapshot_id=parent.content_snapshot_id, evaluation_time=NOW,
        )

    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    broken_slots = dict(parent.artifact_ids)
    broken_slots["verification"] = broken_slots["source"]
    with pytest.raises(ValueError, match="VerificationArtifact"):
        operation.complete_review(
            claim.claim_id, _valid_result(claim.claim_id),
            parent_snapshot=replace(parent, artifact_ids=broken_slots), evaluation_time=NOW,
        )

    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    broken_slots = dict(parent.artifact_ids)
    broken_slots["lifecycle"] = broken_slots["source"]
    with pytest.raises(ValueError, match="LifecycleArtifact"):
        operation.complete_review(
            claim.claim_id, _valid_result(claim.claim_id),
            parent_snapshot=replace(parent, artifact_ids=broken_slots), evaluation_time=NOW,
        )

    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    old_verification = artifacts.get(parent.artifact_ids["verification"])
    artifacts.items[old_verification.artifact_id] = replace(
        old_verification, claim_artifact_id="claims-other"
    )
    with pytest.raises(ValueError, match="claim_artifact_id"):
        operation.complete_review(
            claim.claim_id, _valid_result(claim.claim_id),
            parent_snapshot_id=parent.content_snapshot_id, evaluation_time=NOW,
        )


def test_completion_rejects_late_verification_or_reference_dependencies_before_writes():
    claim, lifecycle, artifacts, snapshots, parent, operation = _review_fixture()
    before_keys = set(artifacts.items)
    before_appends = len(lifecycle.appended)
    with pytest.raises(ValueError, match="verification_timestamp"):
        operation.complete_review(
            claim.claim_id,
            _valid_result(claim.claim_id, verification_timestamp=NOW + timedelta(hours=2)),
            parent_snapshot_id=parent.content_snapshot_id,
            evaluation_time=NOW + timedelta(hours=1),
            recorded_at=NOW + timedelta(hours=1),
            reference_available_at=NOW + timedelta(hours=1),
        )
    assert set(artifacts.items) == before_keys
    assert len(lifecycle.appended) == before_appends
    with pytest.raises(ValueError, match="reference_available_at"):
        operation.complete_review(
            claim.claim_id,
            _valid_result(claim.claim_id),
            parent_snapshot_id=parent.content_snapshot_id,
            evaluation_time=NOW + timedelta(hours=1),
            recorded_at=NOW + timedelta(hours=1),
            reference_available_at=NOW + timedelta(hours=2),
        )
    assert set(artifacts.items) == before_keys
    assert len(lifecycle.appended) == before_appends

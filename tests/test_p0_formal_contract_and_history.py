import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.claim_event_repository import ClaimStateEventRepository
from stock_content.application.historical_claim_projector import HistoricalClaimProjector
from stock_content.application.service import ContentApplication
from stock_content.application.signal_service import SignalService
from stock_content.application.stages import _claim_state_payload
from stock_content.domain.bitemporal_query import FormalContentSignalQueryV2
from stock_content.domain.claim_state_event import ClaimStateEvent, validate_event_chain
from stock_content.domain.signal_contract_v5_1 import (
    AUTHORITY_FORMAL,
    CONTRACT_CHECKSUM,
    CONTRACT_NAME,
    validate_signal_v5_1,
)
from stock_content.domain.temporal_semantics import AvailabilityQuality


def dt(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=UTC)


def query(**changes):
    value = {
        "contract": CONTRACT_NAME,
        "request_id": "r1",
        "symbols": ("600000.SH",),
        "business_as_of": dt(2),
        "knowledge_as_of": dt(2),
        "availability_as_of": dt(2),
        "content_snapshot_id": "snap-1",
        "pit_mode": "PUBLIC_STRICT",
        "min_support": 2,
    }
    value.update(changes)
    return FormalContentSignalQueryV2(**value)


def test_formal_query_is_immutable_and_all_clocks_are_in_identity():
    first = query()
    assert first.query_id != query(knowledge_as_of=dt(3)).query_id
    assert first.query_id != query(availability_as_of=dt(3)).query_id
    assert first.query_id != query(content_snapshot_id="snap-2").query_id
    with pytest.raises(ValueError, match="UTC"):
        query(business_as_of=datetime(2024, 1, 1))


def test_formal_signal_contract_and_checksum():
    signal = {
        "contract": CONTRACT_NAME,
        "contract_checksum": CONTRACT_CHECKSUM,
        "authority": AUTHORITY_FORMAL,
        "formal_eligible": True,
        "signal_id": "s1",
        "claim_id": "c1",
        "occurrence_id": "o1",
        "semantic_segment_id": "seg1",
        "asserted_at": None,
        "source_available_at": None,
        "source_availability_quality": "EXACT",
        "available_from": "2024-01-01T00:00:00Z",
        "temporal_bindings": [],
        "lifecycle_as_of": {"status": "ACTIVE", "known_from": "2024-01-01T00:00:00Z", "artifact_id": "a1"},
        "content_snapshot_id": "snap-1",
        "evidence_refs": [],
        "producer_commit": "abc",
        "signal_policy_version": "signal-policy.v1",
        "business_as_of": "2024-01-02T00:00:00Z",
        "knowledge_as_of": "2024-01-02T00:00:00Z",
        "availability_as_of": "2024-01-02T00:00:00Z",
    }
    assert validate_signal_v5_1(signal)["authority"] == AUTHORITY_FORMAL
    with pytest.raises(ValueError, match="checksum"):
        validate_signal_v5_1({**signal, "contract_checksum": "bad"})


def test_historical_projector_excludes_late_event_and_detects_tamper():
    e1 = ClaimStateEvent(
        "c1", "VERIFICATION", {"verification_status": "SUPPORTED"}, dt(1), source_available_from=dt(1)
    )
    e2 = ClaimStateEvent(
        "c1",
        "LIFECYCLE",
        {"status": "WITHDRAWN", "artifact_id": "life-1"},
        dt(3),
        source_available_from=dt(3),
        previous_event_hash=e1.event_hash,
    )
    projector = HistoricalClaimProjector([e1, e2], membership=lambda snapshot, claim: snapshot == "snap-1")
    early = projector.project(
        "c1", business_as_of=dt(4), knowledge_as_of=dt(2), availability_as_of=dt(2), content_snapshot_id="snap-1"
    )
    assert early["verification_status"] == "SUPPORTED"
    assert (
        projector.project(
            "c1", business_as_of=dt(4), knowledge_as_of=dt(4), availability_as_of=dt(4), content_snapshot_id="snap-2"
        )
        is None
    )
    with pytest.raises(ValueError, match="chain"):
        validate_event_chain(
            [
                e1,
                ClaimStateEvent(
                    "c1",
                    "LIFECYCLE",
                    {"status": "X"},
                    dt(3),
                    source_available_from=dt(3),
                    previous_event_hash="tampered",
                ),
            ]
        )


def test_historical_projector_requires_real_lifecycle_lineage():
    event = ClaimStateEvent("c1", "VERIFICATION", {"status": "SUPPORTED"}, dt(1), source_available_from=dt(1))
    projector = HistoricalClaimProjector([event], membership=lambda snapshot, claim: True)
    projection = projector.project(
        "c1", business_as_of=dt(2), knowledge_as_of=dt(2), availability_as_of=dt(2), content_snapshot_id="snap"
    )
    assert projection is not None and "lifecycle_as_of" not in projection


def test_formal_signal_id_is_new_and_scoped_to_all_query_identity_fields():
    service = SignalService()
    projection = {
        "signal_id": "legacy-signal-id",
        "claim_id": "c1",
        "occurrence_id": "o1",
        "semantic_segment_id": "seg1",
        "available_from": "2024-01-01T00:00:00Z",
        "producer_commit": "snapshot-commit",
        "asserted_at": None,
        "source_available_at": None,
        "temporal_bindings": [],
        "lifecycle_as_of": {"status": "ACTIVE", "known_from": "2024-01-01T00:00:00Z", "artifact_id": "life1"},
        "evidence_refs": [],
    }
    first = service.build_signal_v5_1(projection, query())
    assert first["signal_id"] != "legacy-signal-id"
    assert first["signal_id"] != service.build_signal_v5_1(projection, query(knowledge_as_of=dt(3)))["signal_id"]
    assert first["signal_id"] != service.build_signal_v5_1(
        projection, query(signal_policy_version="signal-policy.v2")
    )["signal_id"]

    incomplete = dict(projection)
    incomplete["lifecycle_as_of"] = {"status": "ACTIVE", "known_from": "2024-01-01T00:00:00Z"}
    with pytest.raises(ValueError, match="historical lifecycle"):
        service.build_signal_v5_1(incomplete, query())


def test_history_migration_marks_pre_event_claims_incomplete():
    migration = (Path(__file__).parents[1] / "migrations" / "026_claim_state_events_publication.sql").read_text()
    assert "ADD COLUMN IF NOT EXISTS legacy_history_incomplete" in migration
    assert "SET legacy_history_incomplete = TRUE" in migration


@pytest.mark.parametrize(
    ("support_status", "expected_count"),
    [("SUPPORTED", 2), ("PARTIALLY_SUPPORTED", 1), ("UNSUPPORTED", 0), ("AMBIGUOUS", 0)],
)
def test_claim_state_payload_captures_formal_support_and_lineage(support_status, expected_count):
    times = SimpleNamespace(
        asserted_at=dt(1),
        source_available_at=dt(2),
        available_from=dt(3),
        source_availability_quality=AvailabilityQuality.PUBLISHED_TIME_PROXY,
    )
    claim = SimpleNamespace(
        claim_id="c1",
        subject_id="600000",
        source_support_status=support_status,
        temporal_bindings=[],
    )
    occurrence = SimpleNamespace(
        occurrence_id="o1",
        semantic_segment_id="seg1",
        times=times,
        evidence_refs=["ev1"],
    )
    payload = _claim_state_payload(claim, occurrence, None, "snap", "snapshot-commit")
    assert payload["support_count"] == expected_count
    assert payload["asserted_at"] == dt(1).isoformat()
    assert payload["source_availability_quality"] == "PUBLISHED_TIME_PROXY"
    assert payload["producer_commit"] == "snapshot-commit"


def test_formal_projection_is_snapshot_event_only_and_min_support_is_historical(tmp_path):
    e1 = ClaimStateEvent(
        "c1", "VERIFICATION_INITIAL", {
            "claim_id": "c1", "occurrence_id": "o1", "semantic_segment_id": "seg1",
            "available_from": "2024-01-01T00:00:00Z", "source_availability_quality": "EXACT",
            "temporal_bindings": [], "evidence_refs": [], "support_count": 3, "symbol": "600000.SH",
            "producer_commit": "snapshot-commit",
            "status": "VERIFIED",
        }, dt(1), source_available_from=dt(1)
    )
    e2 = ClaimStateEvent(
        "c1", "LIFECYCLE", {"status": "ACTIVE", "artifact_id": "life1"}, dt(1),
        source_available_from=dt(1), previous_event_hash=e1.event_hash,
    )
    projector = HistoricalClaimProjector(
        [e1, e2], membership=lambda snapshot, claim: snapshot == "snap",
        snapshot_claim_ids=lambda snapshot: ["c1"],
    )
    state = projector.project(
        "c1", business_as_of=dt(3), knowledge_as_of=dt(3), availability_as_of=dt(3), content_snapshot_id="snap"
    )
    assert state["support_count"] == 3
    immutable = dict(state)
    mutable_knowledge_row = {"support_count": 3, "lifecycle_status": "ACTIVE"}
    assert state == immutable

    application = ContentApplication.__new__(ContentApplication)
    application._snapshots = SimpleNamespace(get=lambda snapshot_id: object())
    application._publication_uow = SimpleNamespace(
        repository=SimpleNamespace(is_ready=lambda snapshot_id: snapshot_id == "snap")
    )
    application._historical_projector = projector
    application._signal_service = SignalService()
    application._knowledge = SimpleNamespace(
        factor_signals_v5=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mutable knowledge queried"))
    )
    formal_query = query(content_snapshot_id="snap", min_support=3)
    formal = application.formal_factor_signals(formal_query)
    assert len(formal) == 1
    before = json.dumps(formal, sort_keys=True, separators=(",", ":"))
    mutable_knowledge_row.update(support_count=99, lifecycle_status="WITHDRAWN")
    after = json.dumps(application.formal_factor_signals(formal_query), sort_keys=True, separators=(",", ":"))
    assert after == before
    assert application.formal_factor_signals(query(content_snapshot_id="snap", min_support=4)) == []

    database = Database(f"sqlite:///{tmp_path / 'event-fork.db'}")
    database.create_schema()
    events = ClaimStateEventRepository(database.session_factory)
    events.append(e1)
    events.append(e2)
    fork = ClaimStateEvent(
        "c1", "LIFECYCLE", {"status": "WITHDRAWN", "artifact_id": "life2"}, dt(1),
        source_available_from=dt(1), previous_event_hash=e1.event_hash,
    )
    with pytest.raises(Exception):
        events.append(fork)

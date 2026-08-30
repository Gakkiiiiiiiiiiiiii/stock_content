import json
from datetime import date, datetime, timezone

import httpx
import pytest

from stock_content.adapters.reference.quant_temporal_reference import QuantTemporalReferenceAdapter
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.ports.temporal_reference import (
    ExchangeCalendarRef,
    FiscalCalendarRef,
    ResolvedPeriod,
    TemporalReferenceAsOfViolationError,
    TemporalReferenceNotFoundError,
    TemporalReferenceProviderUnavailableError,
)
from stock_content.ports.temporal_reference_snapshot import (
    PinnedTemporalReferenceProvider,
    TemporalReferenceSnapshotMismatchError,
    TemporalReferenceSnapshotNotFoundError,
)

AVAILABLE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class SnapshotFixture:
    def __init__(self, *, latest="R1"):
        self.latest = latest
        self.calls = []

    def resolve_exchange_calendar(self, subject_key, as_of):
        return ExchangeCalendarRef("XSHG", "Asia/Shanghai", self.latest, "v1", AVAILABLE, subject_key)

    def resolve_fiscal_calendar(self, subject_key, as_of):
        return FiscalCalendarRef("issuer-fy", self.latest, "v1", AVAILABLE, subject_key)

    def resolve_period(self, subject_key, period_label, as_of):
        return ResolvedPeriod(
            date(2027, 4, 1), date(2027, 6, 30), period_label, "FISCAL", "issuer-fy",
            self.latest, "v1", AVAILABLE, subject_key,
        )

    def get_exchange_calendar_snapshot(self, reference_snapshot_id):
        self.calls.append(("exchange_calendar", reference_snapshot_id))
        if reference_snapshot_id != "R1":
            raise TemporalReferenceSnapshotNotFoundError(reference_snapshot_id)
        return ExchangeCalendarRef("XSHG", "Asia/Shanghai", "R1", "v1", AVAILABLE, "ABC")

    def get_fiscal_calendar_snapshot(self, reference_snapshot_id):
        raise TemporalReferenceSnapshotNotFoundError(reference_snapshot_id)

    def get_period_snapshot(self, reference_snapshot_id, *, subject_key, period_label):
        self.calls.append(("fiscal_period", reference_snapshot_id, subject_key, period_label))
        if reference_snapshot_id != "R1":
            raise TemporalReferenceSnapshotNotFoundError(reference_snapshot_id)
        return ResolvedPeriod(
            date(2027, 4, 1), date(2027, 6, 30), period_label, "FISCAL", "issuer-fy",
            "R1", "v1", AVAILABLE, subject_key,
        )


def test_pinned_provider_reads_r1_after_resolver_moves_to_r2():
    fixture = SnapshotFixture(latest="R2")
    pinned = PinnedTemporalReferenceProvider(fixture, {"fiscal_period|ABC|FY2027Q2": "R1"})
    result = pinned.resolve_period("ABC", "FY2027Q2", datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert result.reference_snapshot_id == "R1"
    assert fixture.calls == [("fiscal_period", "R1", "ABC", "FY2027Q2")]


def test_pinned_provider_does_not_fallback_when_snapshot_is_missing():
    fixture = SnapshotFixture()
    pinned = PinnedTemporalReferenceProvider(fixture, {"fiscal_period|ABC|FY2027Q2": "R2"})
    with pytest.raises(TemporalReferenceSnapshotNotFoundError):
        pinned.resolve_period("ABC", "FY2027Q2", AVAILABLE)


def test_pinned_provider_rejects_snapshot_subject_mismatch():
    class Mismatch(SnapshotFixture):
        def get_period_snapshot(self, reference_snapshot_id, *, subject_key, period_label):
            return ResolvedPeriod(
                date(2027, 4, 1), date(2027, 6, 30), period_label, "FISCAL", "issuer-fy",
                "R1", "v1", AVAILABLE, "OTHER",
            )

    pinned = PinnedTemporalReferenceProvider(Mismatch(), {"fiscal_period|ABC|FY2027Q2": "R1"})
    with pytest.raises(TemporalReferenceSnapshotMismatchError):
        pinned.resolve_period("ABC", "FY2027Q2", AVAILABLE)


def test_http_adapter_cache_isolated_by_reference_type_and_subject():
    responses = {
        "ABC": {"reference_snapshot_id": "R-ABC", "data_version": "v1", "available_at": "2026-01-01T00:00:00Z",
                 "calendar_id": "XSHG", "timezone": "Asia/Shanghai"},
        "XYZ": {"reference_snapshot_id": "R-XYZ", "data_version": "v1", "available_at": "2026-01-01T00:00:00Z",
                 "calendar_id": "XNYS", "timezone": "America/New_York"},
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=responses[json.loads(request.content)["subject_key"]])
    )
    with httpx.Client(transport=transport) as client:
        adapter = QuantTemporalReferenceAdapter("http://reference", client=client)
        first = adapter.resolve_exchange_calendar("ABC", AVAILABLE)
        second = adapter.resolve_exchange_calendar("XYZ", AVAILABLE)
    assert (first.reference_snapshot_id, second.reference_snapshot_id) == ("R-ABC", "R-XYZ")


def test_replay_missing_reference_is_stable():
    snapshots = SnapshotService()
    snapshot = snapshots.record_from_artifacts(
        source_type="fixture", source_ref="ref", source_content_hash="hash",
        producer_manifest={"reference_data": [{
            "reference_type": "fiscal_period", "subject_key": "ABC", "period_label": "FY2027Q2",
            "binding_key": "fiscal_period|ABC|FY2027Q2", "reference_snapshot_id": "R1",
            "data_version": "v1", "available_at": AVAILABLE.isoformat(),
        }]},
    )
    class MissingSnapshots(SnapshotFixture):
        def get_period_snapshot(self, reference_snapshot_id, *, subject_key, period_label):
            raise TemporalReferenceSnapshotNotFoundError(reference_snapshot_id)

    result = ReplayService(snapshots, temporal_reference_snapshot_provider=MissingSnapshots()).replay(
        snapshot.content_snapshot_id
    )
    assert result["error"] == "REPLAY_REFERENCE_SNAPSHOT_MISSING"


def test_not_found_is_a_business_error_not_provider_failure():
    class Missing:
        def resolve_period(self, subject_key, period_label, as_of):
            raise TemporalReferenceNotFoundError("none")

    from stock_content.domain.temporal_normalizer import TemporalNormalizer
    result = TemporalNormalizer(reference_provider=Missing()).normalize(
        "FY2027Q2", subject_key="ABC", anchor=AVAILABLE,
    )
    assert result.normalization_status == "PARTIAL"
    assert result.normalization_reason == "REFERENCE_NOT_FOUND"


def test_http_provider_unavailable_is_not_converted_to_not_found():
    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "unavailable"}))
    with httpx.Client(transport=transport) as client:
        adapter = QuantTemporalReferenceAdapter("http://reference", client=client)
        with pytest.raises(TemporalReferenceProviderUnavailableError):
            adapter.resolve_exchange_calendar("ABC", AVAILABLE)


def test_http_reference_available_after_as_of_fails_closed():
    response = {
        "reference_snapshot_id": "R1", "data_version": "v1",
        "available_at": "2026-02-01T00:00:00Z", "calendar_id": "XSHG", "timezone": "Asia/Shanghai",
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    with httpx.Client(transport=transport) as client:
        adapter = QuantTemporalReferenceAdapter("http://reference", client=client)
        with pytest.raises(TemporalReferenceAsOfViolationError, match="after as_of"):
            adapter.resolve_exchange_calendar("ABC", AVAILABLE)


@pytest.mark.parametrize("payload", [[], {"reference_snapshot_id": "R1"}])
def test_http_invalid_object_or_missing_fields_is_provider_failure(payload):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with httpx.Client(transport=transport) as client:
        adapter = QuantTemporalReferenceAdapter("http://reference", client=client)
        with pytest.raises(TemporalReferenceProviderUnavailableError):
            adapter.resolve_exchange_calendar("ABC", AVAILABLE)

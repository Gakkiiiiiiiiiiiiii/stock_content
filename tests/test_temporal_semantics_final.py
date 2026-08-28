from datetime import date, datetime, timezone

import pytest

from stock_content.domain.temporal_normalizer import TemporalNormalizer
from stock_content.domain.temporal_semantics import (
    CalendarType,
    ClaimTemporalBinding,
    ClaimTemporalRelation,
    MetricTemporalNature,
    TemporalAssertionStatus,
    TemporalRole,
    TemporalScope,
    TemporalValueType,
)
from stock_content.ports.temporal_reference import ExchangeCalendarRef, FiscalCalendarRef, ResolvedPeriod


class ReferenceData:
    def resolve_exchange_calendar(self, subject_key, as_of):
        return ExchangeCalendarRef(
            calendar_id="XSHG",
            timezone="Asia/Shanghai",
            reference_snapshot_id="exchange-snap-1",
            data_version="exchange-v1",
            available_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

    def resolve_fiscal_calendar(self, subject_key, as_of):
        return FiscalCalendarRef(
            calendar_id="issuer-fy",
            reference_snapshot_id="fiscal-snap-1",
            data_version="fiscal-v1",
            available_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )

    def resolve_period(self, subject_key, period_label, as_of):
        return ResolvedPeriod(
            start_date=date(2026, 5, 1),
            end_date=date(2026, 7, 31),
            period_label=period_label,
            calendar_id="issuer-fy",
            reference_snapshot_id="fiscal-snap-1",
            data_version="fiscal-v1",
            available_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )


def test_final_temporal_scopes_and_financial_horizons_are_deterministic():
    normalizer = TemporalNormalizer()
    timeless = normalizer.normalize("timeless")
    assert (timeless.scope, timeless.value_type) == (TemporalScope.TIMELESS, TemporalValueType.NONE)

    open_ended = normalizer.normalize("自2026年开始")
    assert open_ended.scope is TemporalScope.OPEN_ENDED
    assert open_ended.start_date == date(2026, 1, 1)
    assert open_ended.end_date is None

    recurring = normalizer.normalize("每年 Q4")
    assert recurring.scope is TemporalScope.RECURRING
    assert recurring.recurrence.quarters == [4]

    anchored = datetime(2026, 2, 15, tzinfo=timezone.utc)
    assert normalizer.normalize("YTD", anchor=anchored).metric_temporal_nature is MetricTemporalNature.CUMULATIVE
    assert normalizer.normalize("TTM", anchor=anchored).metric_temporal_nature is MetricTemporalNature.TRAILING
    assert normalizer.normalize("NTM", anchor=anchored).metric_temporal_nature is MetricTemporalNature.FORWARD


def test_fiscal_and_market_reference_provenance_is_retained():
    normalizer = TemporalNormalizer(reference_provider=ReferenceData())
    fiscal = normalizer.normalize("FY2027Q2", anchor=datetime(2026, 8, 1, tzinfo=timezone.utc), subject_key="issuer")
    assert fiscal.calendar_type is CalendarType.FISCAL
    assert fiscal.start_date == date(2026, 5, 1)
    assert fiscal.reference_snapshot_id == "fiscal-snap-1"
    assert fiscal.reference_data_version == "fiscal-v1"
    assert fiscal.reference_available_at == datetime(2026, 1, 3, tzinfo=timezone.utc)

    close = normalizer.normalize("2026-08-20 收盘", subject_key="issuer")
    assert close.market_session == "REGULAR"
    assert close.calendar_type is None  # no anchor: do not invent exchange lookup

    anchored_close = normalizer.normalize(
        "今天收盘", anchor=datetime(2026, 8, 20, tzinfo=timezone.utc), subject_key="issuer"
    )
    assert anchored_close.market_session == "REGULAR"
    assert anchored_close.calendar_type is CalendarType.EXCHANGE
    assert anchored_close.timezone == "Asia/Shanghai"
    assert anchored_close.reference_snapshot_id == "exchange-snap-1"


def test_approximate_horizon_keeps_uncertainty_without_fabricating_exact_dates():
    anchor = datetime(2026, 1, 1, tzinfo=timezone.utc)
    binding = TemporalNormalizer().normalize("未来两三个季度", anchor=anchor)
    assert binding.scope is TemporalScope.FORECAST
    assert binding.precision.value == "APPROXIMATE"
    assert binding.start_date is None and binding.end_date is None
    assert binding.earliest_end_date == date(2026, 7, 1)
    assert binding.latest_end_date == date(2026, 10, 1)

    unanchored = TemporalNormalizer().normalize("未来两三个季度")
    assert unanchored.normalization_status == "PARTIAL"
    assert unanchored.value_type is TemporalValueType.NONE
    assert unanchored.expression_key


def test_assertion_markers_override_role_defaults_and_recompute_identity():
    normalizer = TemporalNormalizer()
    planned = normalizer.normalize("计划2026Q2")
    expected = normalizer.normalize("预计 2026-06-30")
    revised = normalizer.normalize("2026Q3 revised")

    assert planned.scope is TemporalScope.INTERVAL
    assert planned.assertion_status is TemporalAssertionStatus.PLANNED
    assert planned.start_date == date(2026, 4, 1)
    assert expected.assertion_status is TemporalAssertionStatus.EXPECTED
    assert expected.start_date == date(2026, 6, 30)
    assert revised.assertion_status is TemporalAssertionStatus.REVISED
    assert revised.start_date == date(2026, 7, 1)

    actual = normalizer.normalize("2026Q2")
    assert planned.temporal_binding_id != actual.temporal_binding_id
    assert normalizer.normalize(
        "预计2026Q2", assertion_status=TemporalAssertionStatus.ACTUAL
    ).assertion_status is TemporalAssertionStatus.ACTUAL


def test_caller_metric_nature_and_assertion_status_are_identity_inputs():
    normalizer = TemporalNormalizer()
    instant = normalizer.normalize(
        "2026-06-30",
        metric_temporal_nature=MetricTemporalNature.INSTANT,
        assertion_status=TemporalAssertionStatus.ACTUAL,
    )
    duration = normalizer.normalize(
        "2026-06-30",
        metric_temporal_nature=MetricTemporalNature.DURATION,
        assertion_status=TemporalAssertionStatus.EXPECTED,
    )
    assert instant.metric_temporal_nature is MetricTemporalNature.INSTANT
    assert duration.metric_temporal_nature is MetricTemporalNature.DURATION
    assert duration.assertion_status is TemporalAssertionStatus.EXPECTED
    assert instant.temporal_binding_id != duration.temporal_binding_id


def test_normalizer_fail_closed_role_scope_invariants():
    with pytest.raises(ValueError, match="FORECAST_TARGET"):
        TemporalNormalizer().normalize(
            "timeless", role=TemporalRole.FORECAST_TARGET, scope_hint=TemporalScope.TIMELESS
        )
    with pytest.raises(ValueError, match="CONDITION_PERIOD"):
        TemporalNormalizer().normalize(
            "timeless", role=TemporalRole.CONDITION_PERIOD, scope_hint=TemporalScope.TIMELESS
        )


def test_temporal_family_and_causal_lag_invariants_fail_closed():
    with pytest.raises(ValueError, match="DATE temporal binding"):
        ClaimTemporalBinding(role="VALID_AT", scope="POINT", value_type="DATE", start_time=datetime.now())
    with pytest.raises(ValueError, match="TIMELESS"):
        ClaimTemporalBinding(role="VALID_AT", scope="TIMELESS", value_type="DATE", start_date="2026-01-01")
    with pytest.raises(ValueError, match="lag_min"):
        ClaimTemporalRelation(
            relation_type="LAG",
            from_binding_id="tb-a",
            to_binding_id="tb-b",
            lag_min=3,
            lag_max=2,
        )

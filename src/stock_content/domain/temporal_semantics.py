"""Financial temporal semantics owned by canonical claims."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .artifacts import canonical_json


class _ValueEnum(str, Enum):
    @classmethod
    def values(cls) -> set[str]:
        return {x.value for x in cls}


class TemporalRole(_ValueEnum):
    VALID_AT = "VALID_AT"
    OBSERVED_AT = "OBSERVED_AT"
    EVENT_AT = "EVENT_AT"
    ANNOUNCED_AT = "ANNOUNCED_AT"
    EFFECTIVE_AT = "EFFECTIVE_AT"
    REPORTING_PERIOD = "REPORTING_PERIOD"
    REPORTED_AT = "REPORTED_AT"
    FORECAST_TARGET = "FORECAST_TARGET"
    CONDITION_PERIOD = "CONDITION_PERIOD"
    INVALIDATION_PERIOD = "INVALIDATION_PERIOD"
    RECORD_DATE = "RECORD_DATE"
    EX_DATE = "EX_DATE"
    PAYMENT_DATE = "PAYMENT_DATE"
    EXECUTION_PERIOD = "EXECUTION_PERIOD"


class TemporalScope(_ValueEnum):
    TIMELESS = "TIMELESS"
    POINT = "POINT"
    INTERVAL = "INTERVAL"
    OPEN_ENDED = "OPEN_ENDED"
    FORECAST = "FORECAST"
    RECURRING = "RECURRING"
    UNKNOWN = "UNKNOWN"


class TemporalPrecision(_ValueEnum):
    EXACT = "EXACT"
    DAY = "DAY"
    MONTH = "MONTH"
    QUARTER = "QUARTER"
    HALF_YEAR = "HALF_YEAR"
    YEAR = "YEAR"
    APPROXIMATE = "APPROXIMATE"
    UNKNOWN = "UNKNOWN"


class TemporalGranularity(_ValueEnum):
    TICK = "TICK"
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"
    EVENT = "EVENT"
    UNKNOWN = "UNKNOWN"


class TemporalAssertionStatus(_ValueEnum):
    ACTUAL = "ACTUAL"
    PLANNED = "PLANNED"
    EXPECTED = "EXPECTED"
    ESTIMATED = "ESTIMATED"
    REVISED = "REVISED"
    GUIDED = "GUIDED"
    UNKNOWN = "UNKNOWN"


class MetricTemporalNature(_ValueEnum):
    INSTANT = "INSTANT"
    DURATION = "DURATION"
    CUMULATIVE = "CUMULATIVE"
    TRAILING = "TRAILING"
    FORWARD = "FORWARD"
    SNAPSHOT = "SNAPSHOT"
    UNKNOWN = "UNKNOWN"


class CalendarType(_ValueEnum):
    CALENDAR = "CALENDAR"
    FISCAL = "FISCAL"
    EXCHANGE = "EXCHANGE"
    CUSTOM = "CUSTOM"


class TemporalValueType(_ValueEnum):
    NONE = "NONE"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"


class AvailabilityQuality(_ValueEnum):
    EXACT = "EXACT"
    PUBLISHED_TIME_PROXY = "PUBLISHED_TIME_PROXY"
    INGEST_TIME_UPPER_BOUND = "INGEST_TIME_UPPER_BOUND"
    UNKNOWN = "UNKNOWN"


class RecurrencePattern(BaseModel):
    frequency: str
    months: list[int] = Field(default_factory=list)
    quarters: list[int] = Field(default_factory=list)
    weekdays: list[int] = Field(default_factory=list)
    market_session: str | None = None
    rule_text: str | None = None


class ClaimTemporalBinding(BaseModel):
    temporal_binding_id: str = ""
    role: TemporalRole
    scope: TemporalScope
    value_type: TemporalValueType = TemporalValueType.NONE
    start_time: datetime | None = None
    end_time: datetime | None = None
    earliest_start_time: datetime | None = None
    latest_start_time: datetime | None = None
    earliest_end_time: datetime | None = None
    latest_end_time: datetime | None = None
    start_date: date | None = None
    end_date: date | None = None
    earliest_start_date: date | None = None
    latest_start_date: date | None = None
    earliest_end_date: date | None = None
    latest_end_date: date | None = None
    period_label: str | None = None
    raw_expression: str | None = None
    expression_key: str | None = None
    precision: TemporalPrecision = TemporalPrecision.UNKNOWN
    granularity: TemporalGranularity | None = None
    assertion_status: TemporalAssertionStatus = TemporalAssertionStatus.UNKNOWN
    metric_temporal_nature: MetricTemporalNature | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    timezone: str | None = None
    calendar_type: CalendarType | None = None
    calendar_id: str | None = None
    market_session: str | None = None
    recurrence: RecurrencePattern | None = None
    normalization_status: str = "UNRESOLVED"
    normalization_reason: str | None = None
    normalization_version: str = "normalization.v1"
    source_evidence_refs: list[str] = Field(default_factory=list)
    reference_snapshot_id: str | None = None
    reference_data_version: str | None = None
    reference_available_at: datetime | None = None

    @model_validator(mode="after")
    def _families(self) -> "ClaimTemporalBinding":
        date_fields = (
            self.start_date,
            self.end_date,
            self.earliest_start_date,
            self.latest_start_date,
            self.earliest_end_date,
            self.latest_end_date,
        )
        time_fields = (
            self.start_time,
            self.end_time,
            self.earliest_start_time,
            self.latest_start_time,
            self.earliest_end_time,
            self.latest_end_time,
        )
        if self.value_type is TemporalValueType.DATE and any(time_fields):
            raise ValueError("DATE temporal binding cannot contain timestamp fields")
        if self.value_type is TemporalValueType.TIMESTAMP and any(date_fields):
            raise ValueError("TIMESTAMP temporal binding cannot contain date fields")
        if self.value_type is TemporalValueType.NONE and (any(date_fields) or any(time_fields)):
            raise ValueError("NONE temporal binding cannot contain date/time fields")
        if self.scope is TemporalScope.TIMELESS:
            if self.value_type is not TemporalValueType.NONE or self.recurrence is not None:
                raise ValueError("TIMELESS temporal binding must use NONE and have no recurrence")
        if self.scope is TemporalScope.RECURRING and self.recurrence is None:
            raise ValueError("RECURRING temporal binding requires a recurrence pattern")
        if self.scope is TemporalScope.OPEN_ENDED:
            endpoints = [
                self.start_time,
                self.end_time,
                self.start_date,
                self.end_date,
            ]
            if sum(value is not None for value in endpoints) != 1:
                raise ValueError("OPEN_ENDED temporal binding must have exactly one endpoint")
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise ValueError("temporal start_time must not be after end_time")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError("temporal start_date must not be after end_date")
        if (
            self.normalization_status.upper() in {"PARTIAL", "UNRESOLVED"}
            and not self.expression_key
            and self.raw_expression
        ):
            object.__setattr__(self, "expression_key", expression_key_of(self.raw_expression, self.role, self.scope))
        if not self.temporal_binding_id:
            object.__setattr__(self, "temporal_binding_id", temporal_binding_id_of(self))
        return self


class ClaimTemporalRelation(BaseModel):
    temporal_relation_id: str = ""
    relation_type: str
    from_binding_id: str
    to_binding_id: str
    lag_value: float | None = None
    lag_unit: str | None = None
    lag_min: float | None = None
    lag_max: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _id(self) -> "ClaimTemporalRelation":
        for name in ("lag_value", "lag_min", "lag_max"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError("causal lag cannot be negative")
        if self.lag_min is not None and self.lag_max is not None and self.lag_min > self.lag_max:
            raise ValueError("lag_min must not be greater than lag_max")
        if not self.temporal_relation_id:
            object.__setattr__(self, "temporal_relation_id", temporal_relation_id_of(self))
        return self


class OccurrenceTimes(BaseModel):
    asserted_at: datetime | None = None
    source_published_at: datetime | None = None
    source_available_at: datetime | None = None
    source_availability_quality: AvailabilityQuality = AvailabilityQuality.UNKNOWN
    ingested_at: datetime
    extraction_completed_at: datetime
    snapshot_committed_at: datetime
    available_from: datetime

    @model_validator(mode="after")
    def _availability(self) -> "OccurrenceTimes":
        if self.available_from < max(self.ingested_at, self.extraction_completed_at, self.snapshot_committed_at):
            raise ValueError("available_from must be no earlier than required dependencies")
        if self.available_from != self.snapshot_committed_at:
            raise ValueError("available_from must equal snapshot_committed_at")
        return self


def expression_key_of(raw_expression: str, role: TemporalRole | str, scope: TemporalScope | str) -> str:
    payload = {
        "raw_expression": " ".join(raw_expression.strip().split()).casefold(),
        "role": str(getattr(role, "value", role)),
        "scope": str(getattr(scope, "value", scope)),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _enum(value: Any) -> Any:
    return getattr(value, "value", value)


def temporal_binding_identity_payload(binding: ClaimTemporalBinding) -> dict[str, Any]:
    data = binding.model_dump(
        mode="json",
        exclude={
            "temporal_binding_id",
            "raw_expression",
            "confidence",
            "source_evidence_refs",
            "normalization_reason",
            "reference_snapshot_id",
            "reference_data_version",
            "reference_available_at",
            # Normalization metadata describes the parser/provenance, not
            # the semantic time value.  Claim identity carries its own
            # normalization version separately.
            "normalization_status",
            "normalization_version",
        },
    )
    if binding.normalization_status.upper() not in {"PARTIAL", "UNRESOLVED"}:
        data.pop("expression_key", None)
    return data


def temporal_binding_id_of(binding: ClaimTemporalBinding) -> str:
    return "tb_" + hashlib.sha256(canonical_json(temporal_binding_identity_payload(binding)).encode()).hexdigest()


def temporal_relation_id_of(relation: ClaimTemporalRelation) -> str:
    data = {
        "relation_type": relation.relation_type,
        "from_binding_id": relation.from_binding_id,
        "to_binding_id": relation.to_binding_id,
        "lag_value": relation.lag_value,
        "lag_unit": relation.lag_unit,
        "lag_min": relation.lag_min,
        "lag_max": relation.lag_max,
    }
    return "tr_" + hashlib.sha256(canonical_json(data).encode()).hexdigest()


binding_id_of = temporal_binding_id_of
relation_id_of = temporal_relation_id_of


__all__ = [name for name in globals() if not name.startswith("_")]

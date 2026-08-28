"""Port for replayable fiscal/exchange reference data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class ExchangeCalendarRef:
    calendar_id: str
    timezone: str
    reference_snapshot_id: str = ""
    data_version: str = ""
    available_at: datetime | None = None


@dataclass(frozen=True)
class FiscalCalendarRef:
    calendar_id: str
    reference_snapshot_id: str = ""
    data_version: str = ""
    available_at: datetime | None = None


@dataclass(frozen=True)
class ResolvedPeriod:
    start_date: date
    end_date: date
    period_label: str
    calendar_type: str = "FISCAL"
    calendar_id: str | None = None
    reference_snapshot_id: str | None = None
    data_version: str | None = None
    available_at: datetime | None = None


class TemporalReferenceProvider(Protocol):
    def resolve_exchange_calendar(self, subject_key: str, as_of: datetime) -> ExchangeCalendarRef | None: ...
    def resolve_fiscal_calendar(self, subject_key: str, as_of: datetime) -> FiscalCalendarRef | None: ...
    def resolve_period(self, subject_key: str, period_label: str, as_of: datetime) -> ResolvedPeriod | None: ...


__all__ = ["ExchangeCalendarRef", "FiscalCalendarRef", "ResolvedPeriod", "TemporalReferenceProvider"]

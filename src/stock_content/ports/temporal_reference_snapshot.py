"""Exact historical temporal-reference retrieval port.

The resolve port deliberately chooses a reference by ``as_of``.  This port is
different: callers provide an immutable snapshot id and no latest/current
selection is permitted.
"""

from __future__ import annotations

from typing import Protocol

from .temporal_reference import (
    ExchangeCalendarRef,
    FiscalCalendarRef,
    ResolvedPeriod,
    TemporalReferenceError,
)


class TemporalReferenceSnapshotError(TemporalReferenceError):
    """Base error for snapshot-by-id retrieval."""


class TemporalReferenceSnapshotNotFoundError(TemporalReferenceSnapshotError):
    """The immutable snapshot id is not retained by the provider."""


class TemporalReferenceSnapshotMismatchError(TemporalReferenceSnapshotError):
    """The returned snapshot does not match the pinned lookup contract."""


# Short aliases keep the error vocabulary convenient for adapters and callers.
ReferenceSnapshotNotFoundError = TemporalReferenceSnapshotNotFoundError
ReferenceSnapshotMismatchError = TemporalReferenceSnapshotMismatchError
ReferenceSnapshotNotFound = TemporalReferenceSnapshotNotFoundError
ReferenceSnapshotMismatch = TemporalReferenceSnapshotMismatchError


class TemporalReferenceSnapshotProvider(Protocol):
    def get_exchange_calendar_snapshot(self, reference_snapshot_id: str) -> ExchangeCalendarRef: ...

    def get_fiscal_calendar_snapshot(self, reference_snapshot_id: str) -> FiscalCalendarRef: ...

    def get_period_snapshot(
        self,
        reference_snapshot_id: str,
        *,
        subject_key: str,
        period_label: str,
    ) -> ResolvedPeriod: ...


class PinnedTemporalReferenceProvider:
    """Resolve temporal references only through snapshot ids fixed by history."""

    def __init__(self, snapshot_provider: TemporalReferenceSnapshotProvider, pinned_reference_ids: dict[str, str]):
        self._snapshots = snapshot_provider
        self._pins = {str(key): str(value) for key, value in pinned_reference_ids.items() if value}

    def _lookup(self, reference_type: str, subject_key: str, period_label: str = "") -> str:
        candidates = (
            f"{reference_type}|{subject_key}|{period_label}",
            f"{reference_type}:{subject_key}:{period_label}",
            f"{reference_type}|{subject_key}",
            f"{reference_type}:{subject_key}",
            f"{reference_type}|{period_label}",
            reference_type,
        )
        for key in candidates:
            if key in self._pins:
                return self._pins[key]
        raise TemporalReferenceSnapshotNotFoundError(
            f"no pinned reference snapshot for {reference_type}/{subject_key}/{period_label}"
        )

    @staticmethod
    def _check(value, expected_id: str, *, reference_type: str, subject_key: str = "", period_label: str = ""):
        if value is None:
            raise TemporalReferenceSnapshotNotFoundError(expected_id)
        expected_class = {
            "exchange_calendar": ExchangeCalendarRef,
            "fiscal_calendar": FiscalCalendarRef,
            "fiscal_period": ResolvedPeriod,
        }.get(reference_type)
        if expected_class is None or not isinstance(value, expected_class):
            raise TemporalReferenceSnapshotMismatchError(
                f"snapshot type mismatch for {reference_type}"
            )
        actual = str(getattr(value, "reference_snapshot_id", "") or "")
        if actual != expected_id:
            raise TemporalReferenceSnapshotMismatchError(
                f"snapshot id mismatch for {reference_type}: expected {expected_id}, got {actual or '<missing>'}"
            )
        if reference_type == "fiscal_period":
            actual_label = str(getattr(value, "period_label", "") or "")
            if actual_label != period_label:
                raise TemporalReferenceSnapshotMismatchError(
                    f"period label mismatch: expected {period_label}, got {actual_label}"
                )
        actual_subject = str(getattr(value, "subject_key", "") or "")
        if actual_subject and actual_subject != subject_key:
            raise TemporalReferenceSnapshotMismatchError(
                f"subject mismatch: expected {subject_key}, got {actual_subject}"
            )
        return value

    def _optional_lookup(self, reference_type: str, subject_key: str) -> str | None:
        for key in (f"{reference_type}|{subject_key}", f"{reference_type}:{subject_key}", reference_type):
            if key in self._pins:
                return self._pins[key]
        # Manifest records may retain a period label even when the pinned
        # payload is only a fiscal/exchange calendar.  Match that exact
        # subject prefix without falling back to another subject.
        prefix = f"{reference_type}|{subject_key}|"
        for key in sorted(self._pins):
            if key.startswith(prefix):
                return self._pins[key]
        return None

    def resolve_exchange_calendar(self, subject_key, as_of):
        reference_id = self._optional_lookup("exchange_calendar", str(subject_key))
        if reference_id is None:
            return None
        return self._check(
            self._snapshots.get_exchange_calendar_snapshot(reference_id), reference_id,
            reference_type="exchange_calendar", subject_key=str(subject_key),
        )

    def resolve_fiscal_calendar(self, subject_key, as_of):
        reference_id = self._optional_lookup("fiscal_calendar", str(subject_key))
        if reference_id is None:
            return None
        return self._check(
            self._snapshots.get_fiscal_calendar_snapshot(reference_id), reference_id,
            reference_type="fiscal_calendar", subject_key=str(subject_key),
        )

    def resolve_period(self, subject_key, period_label, as_of):
        reference_id = self._lookup("fiscal_period", str(subject_key), str(period_label))
        return self._check(
            self._snapshots.get_period_snapshot(
                reference_id, subject_key=str(subject_key), period_label=str(period_label)
            ), reference_id, reference_type="fiscal_period", subject_key=str(subject_key),
            period_label=str(period_label),
        )


__all__ = [
    "TemporalReferenceSnapshotProvider", "TemporalReferenceSnapshotError",
    "TemporalReferenceSnapshotNotFoundError", "TemporalReferenceSnapshotMismatchError",
    "ReferenceSnapshotNotFoundError", "ReferenceSnapshotMismatchError",
    "ReferenceSnapshotNotFound", "ReferenceSnapshotMismatch",
    "PinnedTemporalReferenceProvider",
]

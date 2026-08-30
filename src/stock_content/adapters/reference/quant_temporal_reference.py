"""HTTP adapter for Quant's immutable fiscal/exchange reference data."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from stock_content.ports.temporal_reference import (
    ExchangeCalendarRef,
    FiscalCalendarRef,
    ResolvedPeriod,
    TemporalReferenceAsOfViolationError,
    TemporalReferenceNotFoundError,
    TemporalReferenceProvider,
    TemporalReferenceProviderUnavailableError,
)
from stock_content.ports.temporal_reference_snapshot import (
    TemporalReferenceSnapshotMismatchError,
    TemporalReferenceSnapshotNotFoundError,
    TemporalReferenceSnapshotProvider,
)


def _datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"reference response missing {field}")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"reference response {field} must include timezone")
    return parsed.astimezone(timezone.utc)


class QuantTemporalReferenceAdapter(TemporalReferenceProvider, TemporalReferenceSnapshotProvider):
    """Resolve and retrieve immutable temporal references over explicit HTTP."""

    resolve_path = "/api/v1/reference/temporal/resolve"
    snapshot_path = "/api/v1/reference/temporal/snapshots"

    def __init__(self, base_url: str, *, api_key: str | None = None, timeout: float = 10.0,
                 caller_service: str = "stock_content", caller_headers: dict[str, str] | None = None,
                 client: Any | None = None) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("temporal reference base_url is required")
        if timeout <= 0:
            raise ValueError("temporal reference timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = float(timeout)
        self._caller_headers = {"X-Caller-Service": caller_service, **dict(caller_headers or {})}
        self._client = client
        self._cache: dict[tuple[Any, ...], Any] = {}

    @staticmethod
    def _unwrap(body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("reference response must be a JSON object")
        if set(body) == {"data"}:
            body = body["data"]
        if not isinstance(body, dict):
            raise ValueError("reference response data must be an object")
        return body

    @staticmethod
    def _error_code(body: Any) -> str:
        if not isinstance(body, dict):
            return ""
        value = body.get("code") or body.get("error")
        if isinstance(value, dict):
            value = value.get("code") or value.get("type")
        return str(value or "").upper()

    def _request(self, path: str, payload: dict[str, Any], *, snapshot_id: str | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers = {"Accept": "application/json", "Content-Type": "application/json", **self._caller_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["X-API-Key"] = self._api_key
        try:
            if self._client is not None:
                response = self._client.post(url, json=payload, headers=headers, timeout=self._timeout)
            else:
                response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise TemporalReferenceProviderUnavailableError("temporal reference provider unavailable") from exc
        if response.status_code == 404:
            if snapshot_id:
                raise TemporalReferenceSnapshotNotFoundError(f"reference snapshot {snapshot_id} not found")
            raise TemporalReferenceNotFoundError("temporal reference not found")
        if response.status_code >= 500:
            raise TemporalReferenceProviderUnavailableError(
                f"temporal reference provider returned HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except (ValueError, TypeError) as exc:
            raise TemporalReferenceProviderUnavailableError("temporal reference returned invalid JSON") from exc
        if response.status_code >= 400:
            if self._error_code(body) in {"NOT_FOUND", "REFERENCE_NOT_FOUND"}:
                if snapshot_id:
                    raise TemporalReferenceSnapshotNotFoundError(f"reference snapshot {snapshot_id} not found")
                raise TemporalReferenceNotFoundError("temporal reference not found")
            raise TemporalReferenceProviderUnavailableError(
                f"temporal reference provider returned HTTP {response.status_code}"
            )
        try:
            body = self._unwrap(body)
        except (TypeError, ValueError) as exc:
            raise TemporalReferenceProviderUnavailableError(
                "temporal reference returned an invalid object"
            ) from exc
        if self._error_code(body) in {"NOT_FOUND", "REFERENCE_NOT_FOUND"}:
            if snapshot_id:
                raise TemporalReferenceSnapshotNotFoundError(f"reference snapshot {snapshot_id} not found")
            raise TemporalReferenceNotFoundError("temporal reference not found")
        return body

    @staticmethod
    def _common(body: dict[str, Any]) -> tuple[str, str, datetime]:
        snapshot_id = body.get("reference_snapshot_id")
        version = body.get("data_version")
        if not isinstance(snapshot_id, str) or not snapshot_id or not isinstance(version, str) or not version:
            raise ValueError("reference response requires reference_snapshot_id and data_version")
        return snapshot_id, version, _datetime(body.get("available_at"), "available_at")

    def _resolve(self, reference_type: str, subject_key: str, as_of: datetime, period_label: str | None = None):
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        cache_key = ("resolve", reference_type, subject_key, period_label or "", as_of.isoformat())
        if cache_key in self._cache:
            return self._cache[cache_key]
        payload = {"type": reference_type, "subject_key": subject_key, "as_of": as_of.isoformat()}
        if period_label is not None:
            payload["period_label"] = period_label
        body = self._request(self.resolve_path, payload)
        try:
            snapshot_id, version, available_at = self._common(body)
        except (TypeError, ValueError) as exc:
            raise TemporalReferenceProviderUnavailableError(
                "temporal reference response has invalid fields"
            ) from exc
        if available_at > as_of:
            raise TemporalReferenceAsOfViolationError(
                f"reference {snapshot_id} becomes available after as_of"
            )
        try:
            if reference_type == "exchange_calendar":
                calendar_id, tz = body.get("calendar_id"), body.get("timezone")
                if not isinstance(calendar_id, str) or not isinstance(tz, str) or not calendar_id or not tz:
                    raise ValueError("exchange reference requires calendar_id and timezone")
                result = ExchangeCalendarRef(calendar_id, tz, snapshot_id, version, available_at, subject_key)
            elif reference_type == "fiscal_calendar":
                calendar_id = body.get("calendar_id")
                if not isinstance(calendar_id, str) or not calendar_id:
                    raise ValueError("fiscal reference requires calendar_id")
                result = FiscalCalendarRef(calendar_id, snapshot_id, version, available_at, subject_key)
            else:
                start, end, label = body.get("start_date"), body.get("end_date"), body.get("period_label")
                if not all(isinstance(item, str) and item for item in (start, end, label)):
                    raise ValueError("period reference requires start_date, end_date and period_label")
                result = ResolvedPeriod(date.fromisoformat(start), date.fromisoformat(end), label,
                                        "FISCAL", body.get("calendar_id"), snapshot_id, version, available_at,
                                        subject_key)
        except (TypeError, ValueError) as exc:
            raise TemporalReferenceProviderUnavailableError(
                "temporal reference response has invalid fields"
            ) from exc
        self._cache[cache_key] = result
        return result

    def resolve_exchange_calendar(self, subject_key: str, as_of: datetime):
        return self._resolve("exchange_calendar", str(subject_key), as_of)

    def resolve_fiscal_calendar(self, subject_key: str, as_of: datetime):
        return self._resolve("fiscal_calendar", str(subject_key), as_of)

    def resolve_period(self, subject_key: str, period_label: str, as_of: datetime):
        return self._resolve("period", str(subject_key), as_of, str(period_label))

    def _snapshot(self, reference_type: str, reference_snapshot_id: str, *, subject_key: str = "",
                  period_label: str = ""):
        cache_key = ("snapshot", reference_type, subject_key, period_label, reference_snapshot_id)
        if cache_key in self._cache:
            return self._cache[cache_key]
        body = self._request(
            f"{self.snapshot_path}/{quote(str(reference_snapshot_id), safe='')}",
            {"reference_snapshot_id": str(reference_snapshot_id), "type": reference_type,
             **({"subject_key": subject_key} if subject_key else {}),
             **({"period_label": period_label} if period_label else {})},
            snapshot_id=str(reference_snapshot_id),
        )
        try:
            returned_id, version, available_at = self._common(body)
        except (TypeError, ValueError) as exc:
            raise TemporalReferenceProviderUnavailableError(
                "temporal reference snapshot has invalid fields"
            ) from exc
        if returned_id != str(reference_snapshot_id):
            raise TemporalReferenceSnapshotMismatchError(
                f"snapshot id mismatch: expected {reference_snapshot_id}, got {returned_id}"
            )
        try:
            if reference_type == "exchange_calendar":
                calendar_id, tz = body.get("calendar_id"), body.get("timezone")
                if not isinstance(calendar_id, str) or not isinstance(tz, str) or not calendar_id or not tz:
                    raise ValueError("exchange snapshot requires calendar_id and timezone")
                result = ExchangeCalendarRef(calendar_id, tz, returned_id, version, available_at,
                                             str(body.get("subject_key") or subject_key))
            elif reference_type == "fiscal_calendar":
                calendar_id = body.get("calendar_id")
                if not isinstance(calendar_id, str) or not calendar_id:
                    raise ValueError("fiscal snapshot requires calendar_id")
                result = FiscalCalendarRef(calendar_id, returned_id, version, available_at,
                                           str(body.get("subject_key") or subject_key))
            else:
                start, end, label = body.get("start_date"), body.get("end_date"), body.get("period_label")
                if not all(isinstance(item, str) and item for item in (start, end, label)):
                    raise ValueError("period snapshot requires start_date, end_date and period_label")
                if period_label and label != period_label:
                    raise TemporalReferenceSnapshotMismatchError("period snapshot label mismatch")
                returned_subject = str(body.get("subject_key") or subject_key)
                if subject_key and returned_subject != subject_key:
                    raise TemporalReferenceSnapshotMismatchError("period snapshot subject mismatch")
                result = ResolvedPeriod(date.fromisoformat(start), date.fromisoformat(end), label,
                                        "FISCAL", body.get("calendar_id"), returned_id, version, available_at,
                                        returned_subject)
        except TemporalReferenceSnapshotMismatchError:
            raise
        except (TypeError, ValueError) as exc:
            raise TemporalReferenceProviderUnavailableError(
                "temporal reference snapshot has invalid fields"
            ) from exc
        self._cache[cache_key] = result
        return result

    def get_exchange_calendar_snapshot(self, reference_snapshot_id: str):
        return self._snapshot("exchange_calendar", reference_snapshot_id)

    def get_fiscal_calendar_snapshot(self, reference_snapshot_id: str):
        return self._snapshot("fiscal_calendar", reference_snapshot_id)

    def get_period_snapshot(self, reference_snapshot_id: str, *, subject_key: str, period_label: str):
        return self._snapshot(
            "period", reference_snapshot_id, subject_key=str(subject_key), period_label=str(period_label)
        )


__all__ = ["QuantTemporalReferenceAdapter"]

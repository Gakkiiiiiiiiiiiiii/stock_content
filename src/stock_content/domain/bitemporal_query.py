"""Immutable, canonical formal PIT query identity.

This module is deliberately independent of Pydantic, FastAPI and SQL.  The
formal route can therefore validate requests identically in API and replay
workers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from stock_content.domain.artifacts import canonical_json

FORMAL_CONTRACT = "content-factor-signal.v5.1"
PUBLIC_STRICT = "PUBLIC_STRICT"
QUERY_POLICY_VERSION = "formal-query.v2"


def require_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware UTC datetime")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must use UTC timezone")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class FormalContentSignalQueryV2:
    contract: str
    request_id: str
    symbols: tuple[str, ...]
    business_as_of: datetime
    knowledge_as_of: datetime
    availability_as_of: datetime
    content_snapshot_id: str
    pit_mode: str = PUBLIC_STRICT
    min_support: int = 1
    signal_policy_version: str = "signal-policy.v1"
    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if self.contract != FORMAL_CONTRACT:
            raise ValueError(f"contract must be {FORMAL_CONTRACT}")
        for name in ("request_id", "content_snapshot_id", "signal_policy_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.pit_mode != PUBLIC_STRICT:
            raise ValueError("formal queries require pit_mode=PUBLIC_STRICT")
        if isinstance(self.symbols, str):
            raise ValueError("symbols must be an array")
        symbols = tuple(sorted({str(s).strip() for s in self.symbols if str(s).strip()}))
        if not symbols:
            raise ValueError("symbols must not be empty")
        object.__setattr__(self, "symbols", symbols)
        if isinstance(self.min_support, bool) or not isinstance(self.min_support, int) or self.min_support < 1:
            raise ValueError("min_support must be a positive integer")
        for name in ("business_as_of", "knowledge_as_of", "availability_as_of"):
            object.__setattr__(self, name, require_utc(getattr(self, name), name))
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_utc(value, name))
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FormalContentSignalQueryV2":
        if not isinstance(value, Mapping):
            raise ValueError("formal query must be an object")
        required = (
            "contract",
            "request_id",
            "symbols",
            "business_as_of",
            "knowledge_as_of",
            "availability_as_of",
            "content_snapshot_id",
            "pit_mode",
            "min_support",
        )
        missing = [key for key in required if key not in value or value[key] is None]
        if missing:
            raise ValueError(f"formal query missing {', '.join(missing)}")
        return cls(
            contract=str(value["contract"]),
            request_id=str(value["request_id"]),
            symbols=tuple(value["symbols"]),
            business_as_of=_parse_dt(value["business_as_of"], "business_as_of"),
            knowledge_as_of=_parse_dt(value["knowledge_as_of"], "knowledge_as_of"),
            availability_as_of=_parse_dt(value["availability_as_of"], "availability_as_of"),
            content_snapshot_id=str(value["content_snapshot_id"]),
            pit_mode=str(value["pit_mode"]),
            min_support=value["min_support"],
            signal_policy_version=str(value.get("signal_policy_version") or "signal-policy.v1"),
            start=_parse_optional_dt(value.get("start"), "start"),
            end=_parse_optional_dt(value.get("end"), "end"),
        )

    def canonical_request(self) -> dict[str, Any]:
        payload = {
            "contract": self.contract,
            "request_id": self.request_id,
            "symbols": list(self.symbols),
            "business_as_of": self.business_as_of.isoformat().replace("+00:00", "Z"),
            "knowledge_as_of": self.knowledge_as_of.isoformat().replace("+00:00", "Z"),
            "availability_as_of": self.availability_as_of.isoformat().replace("+00:00", "Z"),
            "content_snapshot_id": self.content_snapshot_id,
            "pit_mode": self.pit_mode,
            "min_support": self.min_support,
            "signal_policy_version": self.signal_policy_version,
        }
        if self.start is not None:
            payload["start"] = self.start.isoformat().replace("+00:00", "Z")
        if self.end is not None:
            payload["end"] = self.end.isoformat().replace("+00:00", "Z")
        return payload

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.canonical_request())

    @property
    def query_id(self) -> str:
        return "query_" + hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _parse_dt(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an RFC3339 datetime")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC3339 datetime") from exc


def _parse_optional_dt(value: Any, field: str) -> datetime | None:
    return None if value is None else _parse_dt(value, field)


def query_id_for(value: FormalContentSignalQueryV2 | Mapping[str, Any]) -> str:
    return (
        value.query_id
        if isinstance(value, FormalContentSignalQueryV2)
        else FormalContentSignalQueryV2.from_mapping(value).query_id
    )


__all__ = ["FORMAL_CONTRACT", "PUBLIC_STRICT", "FormalContentSignalQueryV2", "query_id_for", "require_utc"]

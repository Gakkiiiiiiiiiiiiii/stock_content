"""Append-only bitemporal Claim/Verification/Lifecycle state events."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from stock_content.domain.artifacts import canonical_json


@dataclass(frozen=True)
class ClaimStateEvent:
    claim_id: str
    event_type: str
    payload: dict[str, Any]
    known_from: datetime | None
    business_valid_from: datetime | None = None
    business_valid_to: datetime | None = None
    known_to: datetime | None = None
    source_available_from: datetime | None = None
    previous_event_hash: str | None = None
    event_id: str = ""
    event_hash: str = ""
    legacy_history_incomplete: bool = False

    def __post_init__(self) -> None:
        if not self.claim_id or not self.event_type:
            raise ValueError("claim_id and event_type are required")
        if self.known_from is None and not self.legacy_history_incomplete:
            raise ValueError("known_from is required unless legacy_history_incomplete=true")
        for name in ("known_from", "known_to", "business_valid_from", "business_valid_to", "source_available_from"):
            value = getattr(self, name)
            if value is not None:
                _require_utc(value, name)
        if self.known_from and self.known_to and self.known_from > self.known_to:
            raise ValueError("known_from must not be after known_to")
        if self.business_valid_from and self.business_valid_to and self.business_valid_from > self.business_valid_to:
            raise ValueError("business_valid_from must not be after business_valid_to")
        identity = self.identity_payload()
        expected_event_id = "cse_" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:32]
        expected_hash = hashlib.sha256(canonical_json({"event_id": expected_event_id, **identity}).encode()).hexdigest()
        if self.event_id and self.event_id != expected_event_id:
            raise ValueError("event_id does not match canonical event identity")
        if self.event_hash and self.event_hash != expected_hash:
            raise ValueError("event_hash does not match canonical event identity")
        object.__setattr__(self, "event_id", self.event_id or expected_event_id)
        object.__setattr__(self, "event_hash", self.event_hash or expected_hash)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "known_from": _text(self.known_from),
            "business_valid_from": _text(self.business_valid_from),
            "business_valid_to": _text(self.business_valid_to),
            "known_to": _text(self.known_to),
            "source_available_from": _text(self.source_available_from),
            "previous_event_hash": self.previous_event_hash,
            "legacy_history_incomplete": self.legacy_history_incomplete,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "event_id": self.event_id, "event_hash": self.event_hash}

    def visible_at(
        self,
        *,
        business_as_of: datetime,
        knowledge_as_of: datetime,
        availability_as_of: datetime,
        public_strict: bool = True,
    ) -> bool:
        if self.legacy_history_incomplete or self.known_from is None:
            return False
        if self.known_from > knowledge_as_of or (self.known_to is not None and knowledge_as_of >= self.known_to):
            return False
        if self.source_available_from is None or self.source_available_from > availability_as_of:
            return False
        if self.business_valid_from and business_as_of < self.business_valid_from:
            return False
        if self.business_valid_to and business_as_of >= self.business_valid_to:
            return False
        return True


def validate_event_chain(events: Iterable[ClaimStateEvent]) -> tuple[ClaimStateEvent, ...]:
    # ``previous_event_hash`` describes append order, not business or
    # knowledge time.  A late-known event may legitimately have an older
    # business timestamp, so sorting by a PIT clock would validate the wrong
    # chain and make replay depend on mutable ordering.
    pending = tuple(events)
    by_previous: dict[str | None, ClaimStateEvent] = {}
    for event in pending:
        if event.previous_event_hash in by_previous:
            raise ValueError(f"claim state event chain has multiple successors at {event.event_id}")
        by_previous[event.previous_event_hash] = event
    ordered: list[ClaimStateEvent] = []
    previous: str | None = None
    while by_previous:
        event = by_previous.pop(previous, None)
        if event is None:
            raise ValueError("claim state event chain is broken or has an unreachable event")
        # Reconstructing validates event_hash and catches post-load tampering.
        ClaimStateEvent(
            claim_id=event.claim_id,
            event_type=event.event_type,
            payload=dict(event.payload),
            known_from=event.known_from,
            business_valid_from=event.business_valid_from,
            business_valid_to=event.business_valid_to,
            known_to=event.known_to,
            source_available_from=event.source_available_from,
            previous_event_hash=event.previous_event_hash,
            event_id=event.event_id,
            event_hash=event.event_hash,
            legacy_history_incomplete=event.legacy_history_incomplete,
        )
        previous = event.event_hash
        ordered.append(event)
    return tuple(ordered)


def _require_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")


def _text(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _sort_time(value: datetime | None) -> datetime:
    return value or datetime.min.replace(tzinfo=UTC)


__all__ = ["ClaimStateEvent", "validate_event_chain"]

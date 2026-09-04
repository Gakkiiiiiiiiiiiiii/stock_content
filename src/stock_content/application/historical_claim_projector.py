"""Single historical projection entry point for formal queries and replay."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from stock_content.domain.claim_state_event import ClaimStateEvent, validate_event_chain


class HistoricalLineageIncompleteError(ValueError):
    """The migration marker explicitly denies historical formal use."""


class HistoricalClaimProjector:
    def __init__(
        self,
        events: Iterable[ClaimStateEvent] | None = None,
        *,
        event_loader: Callable[[str], Iterable[ClaimStateEvent]] | None = None,
        membership: Callable[[str, str], bool] | None = None,
        history_incomplete: Callable[[str], bool] | None = None,
        snapshot_claim_ids: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        self._events = tuple(events or ())
        self._loader = event_loader
        self._membership = membership
        self._history_incomplete = history_incomplete
        self._snapshot_claim_ids = snapshot_claim_ids

    def claims_for_snapshot(self, content_snapshot_id: str) -> tuple[str, ...]:
        """Return immutable claim membership, never a latest/current query."""
        if self._snapshot_claim_ids is None:
            return ()
        return tuple(sorted({str(item) for item in self._snapshot_claim_ids(content_snapshot_id) if str(item)}))

    def events_for(self, claim_id: str) -> tuple[ClaimStateEvent, ...]:
        events = tuple(
            self._loader(claim_id) if self._loader else (event for event in self._events if event.claim_id == claim_id)
        )
        return validate_event_chain(events)

    def is_history_incomplete(self, claim_id: str) -> bool:
        """Report a legacy marker without consulting a mutable state row."""
        if self._history_incomplete is not None and self._history_incomplete(claim_id):
            return True
        return any(event.legacy_history_incomplete for event in self.events_for(claim_id))

    def project(
        self,
        claim_id: str,
        *,
        business_as_of: datetime,
        knowledge_as_of: datetime,
        availability_as_of: datetime,
        content_snapshot_id: str | None = None,
    ) -> dict[str, Any] | None:
        if content_snapshot_id and (self._membership is None or not self._membership(content_snapshot_id, claim_id)):
            return None
        # A migration marker is an explicit deny, not an unknown state that a
        # caller may turn into a successful empty formal response.
        if self.is_history_incomplete(claim_id):
            return {"claim_id": claim_id, "content_snapshot_id": content_snapshot_id,
                    "legacy_history_incomplete": True}
        candidates = [
            event
            for event in self.events_for(claim_id)
            if event.visible_at(
                business_as_of=business_as_of, knowledge_as_of=knowledge_as_of, availability_as_of=availability_as_of
            )
        ]
        if not candidates:
            return None
        # Fold every visible event. Verification and lifecycle are separate
        # streams and selecting one "last" event loses one of their states.
        state: dict[str, Any] = {"claim_id": claim_id, "content_snapshot_id": content_snapshot_id}
        lifecycle_event = None
        for event in candidates:
            state.update(dict(event.payload))
            state["event_type"] = event.event_type
            state["event_id"] = event.event_id
            state["event_hash"] = event.event_hash
            if "LIFECYCLE" in event.event_type.upper():
                lifecycle_event = event
        if lifecycle_event is None:
            # Do not derive lifecycle from the mutable knowledge row or invent
            # a status from the query clock.  The raw historical state remains
            # useful to compatibility/replay diagnostics; the formal signal
            # builder rejects this projection because lifecycle_as_of is
            # absent.
            return {
                **state, "payload": dict(state),
                "legacy_history_incomplete": False,
            }
        lifecycle_status = lifecycle_event.payload.get("status") or lifecycle_event.payload.get("to_status")
        lifecycle_artifact_id = (
            lifecycle_event.payload.get("artifact_id")
            or lifecycle_event.payload.get("lifecycle_artifact_id")
        )
        if not lifecycle_status or not lifecycle_artifact_id or lifecycle_event.known_from is None:
            return {**state, "payload": dict(state), "legacy_history_incomplete": True}
        state["lifecycle_as_of"] = {
            "status": str(lifecycle_status),
            "known_from": lifecycle_event.known_from.isoformat().replace("+00:00", "Z"),
            "artifact_id": str(lifecycle_artifact_id),
        }
        state["status"] = state["lifecycle_as_of"]["status"]
        event = candidates[-1]
        return {
            **state, "payload": dict(state),
            "legacy_history_incomplete": any(item.legacy_history_incomplete for item in candidates),
        }

    def project_claim(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return self.project(*args, **kwargs)


def _sort(value: datetime | None) -> datetime:
    from datetime import UTC

    return value or datetime.min.replace(tzinfo=UTC)


__all__ = ["HistoricalClaimProjector", "HistoricalLineageIncompleteError"]

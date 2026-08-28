from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, model_validator

from .artifacts import LifecycleArtifact, canonical_json


class LifecycleTargetType(str, Enum):
    CLAIM = "CLAIM"
    OCCURRENCE = "OCCURRENCE"


class KnowledgeLifecycleEvent(BaseModel):
    lifecycle_event_id: str = ""
    target_type: LifecycleTargetType
    target_id: str
    from_status: str | None = None
    to_status: str
    effective_at: datetime
    recorded_at: datetime
    reason_code: str
    policy_version: str
    supersedes_event_id: str | None = None

    @model_validator(mode="after")
    def _identity(self) -> "KnowledgeLifecycleEvent":
        if not self.lifecycle_event_id:
            payload = {
                "target_type": self.target_type.value,
                "target_id": self.target_id,
                "from_status": self.from_status,
                "to_status": self.to_status,
                "effective_at": self.effective_at,
                "reason_code": self.reason_code,
                "policy_version": self.policy_version,
            }
            object.__setattr__(
                self, "lifecycle_event_id", "le_" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
            )
        return self


def lifecycle_event_id_of(event: KnowledgeLifecycleEvent) -> str:
    return event.lifecycle_event_id


def select_lifecycle_event(
    events: Iterable[KnowledgeLifecycleEvent],
    *,
    target_type: LifecycleTargetType | str,
    target_id: str,
    business_as_of: datetime,
    knowledge_as_of: datetime,
) -> KnowledgeLifecycleEvent | None:
    """Select one deterministic event from a bitemporal event stream.

    Both target dimensions are exact filters.  The remaining event ordering
    is deliberately explicit so replay does not depend on input order.
    """
    requested_type = LifecycleTargetType(target_type)
    candidates = [
        event
        for event in events
        if event.target_type is requested_type
        and event.target_id == target_id
        and event.effective_at <= business_as_of
        and event.recorded_at <= knowledge_as_of
    ]
    return max(
        candidates,
        key=lambda event: (event.effective_at, event.recorded_at, event.lifecycle_event_id),
        default=None,
    )


select_bitemporal_lifecycle_event = select_lifecycle_event


__all__ = [
    "LifecycleTargetType",
    "KnowledgeLifecycleEvent",
    "LifecycleArtifact",
    "lifecycle_event_id_of",
    "select_lifecycle_event",
    "select_bitemporal_lifecycle_event",
]

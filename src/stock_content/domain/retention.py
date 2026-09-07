"""Dry-run retention planning with immutable, locator-free tombstones."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from stock_content.domain.security_redaction import contains_sensitive_value


class RetentionClass(StrEnum):
    RAW_MEDIA = "raw_media"
    SUBTITLE = "subtitle"
    TRANSCRIPT = "transcript"
    KNOWLEDGE = "knowledge"


_ENV = {
    RetentionClass.RAW_MEDIA: "CONTENT_RAW_MEDIA_RETENTION_DAYS",
    RetentionClass.SUBTITLE: "CONTENT_SUBTITLE_RETENTION_DAYS",
    RetentionClass.TRANSCRIPT: "CONTENT_TRANSCRIPT_RETENTION_DAYS",
    RetentionClass.KNOWLEDGE: "CONTENT_KNOWLEDGE_RETENTION_DAYS",
}
_DEFAULTS = {
    RetentionClass.RAW_MEDIA: 7,
    RetentionClass.SUBTITLE: 30,
    RetentionClass.TRANSCRIPT: 90,
    RetentionClass.KNOWLEDGE: 3650,
}


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    days: dict[RetentionClass, int]

    def __post_init__(self) -> None:
        valid_days = all(isinstance(day, int) and 1 <= day <= 36500 for day in self.days.values())
        if set(self.days) != set(RetentionClass) or not valid_days:
            raise ValueError("retention days must be configured as positive values no greater than 36500")

    @classmethod
    def from_environment(cls) -> "RetentionPolicy":
        values: dict[RetentionClass, int] = {}
        for artifact_class, default in _DEFAULTS.items():
            try:
                values[artifact_class] = int(os.getenv(_ENV[artifact_class], str(default)))
            except ValueError as exc:
                raise ValueError("retention days must be integers") from exc
        return cls(values)

    def expires_at(self, artifact_class: RetentionClass, created_at: datetime) -> datetime:
        created = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
        return created + timedelta(days=self.days[artifact_class])


@dataclass(frozen=True, slots=True)
class RetentionCandidate:
    artifact_id: str
    artifact_class: RetentionClass
    content_hash: str
    source_identity_hash: str
    audit_lineage_id: str
    created_at: datetime
    legal_hold: bool = False

    def __post_init__(self) -> None:
        values = (self.artifact_id, self.content_hash, self.source_identity_hash, self.audit_lineage_id)
        if not all(values) or contains_sensitive_value(values):
            raise ValueError("retention candidate must contain stable, non-secret lineage only")


@dataclass(frozen=True, slots=True)
class Tombstone:
    tombstone_id: str
    artifact_id: str
    artifact_class: RetentionClass
    content_hash: str
    source_identity_hash: str
    audit_lineage_id: str
    reason: str
    expired_at: datetime
    recorded_at: datetime

    @classmethod
    def for_candidate(cls, candidate: RetentionCandidate, *, expired_at: datetime, now: datetime) -> "Tombstone":
        identity = "|".join(
            (candidate.artifact_id, candidate.artifact_class.value, candidate.content_hash, candidate.audit_lineage_id)
        )
        return cls(
            tombstone_id="ts_" + hashlib.sha256(identity.encode()).hexdigest(),
            artifact_id=candidate.artifact_id,
            artifact_class=candidate.artifact_class, content_hash=candidate.content_hash,
            source_identity_hash=candidate.source_identity_hash, audit_lineage_id=candidate.audit_lineage_id,
            reason="RETENTION_EXPIRED", expired_at=expired_at, recorded_at=now,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "tombstone_id": self.tombstone_id,
            "artifact_id": self.artifact_id,
            "artifact_class": self.artifact_class.value,
            "content_hash": self.content_hash, "source_identity_hash": self.source_identity_hash,
            "audit_lineage_id": self.audit_lineage_id, "reason": self.reason,
            "expired_at": self.expired_at.isoformat(), "recorded_at": self.recorded_at.isoformat(),
        }


class RetentionExecutionState(StrEnum):
    DELETE_PENDING = "DELETE_PENDING"
    DELETED = "DELETED"
    DELETE_FAILED = "DELETE_FAILED"


@dataclass(frozen=True, slots=True)
class RetentionExecution:
    """Mutable execution state around an immutable tombstone identity."""

    tombstone: Tombstone
    state: RetentionExecutionState
    attempt_count: int = 0
    last_error_code: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_count < 0:
            raise ValueError("retention attempt count cannot be negative")


__all__ = [
    "RetentionCandidate",
    "RetentionClass",
    "RetentionExecution",
    "RetentionExecutionState",
    "RetentionPolicy",
    "Tombstone",
]

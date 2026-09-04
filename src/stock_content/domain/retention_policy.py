"""Retention, derived-data and tombstone rules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .source_policy import AccessClassification


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    retention_class: str
    retain_for: timedelta
    derived_retain_for: timedelta | None = None
    access_classification: AccessClassification = AccessClassification.INTERNAL
    policy_version: str = "retention-policy.v1"

    def expires_at(self, created_at: datetime) -> datetime:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return created_at + self.retain_for

    def is_expired(self, created_at: datetime, *, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at(created_at)

    def derived_expires_at(self, created_at: datetime) -> datetime:
        return created_at + (self.derived_retain_for or self.retain_for)


@dataclass(frozen=True, slots=True)
class Tombstone:
    artifact_id: str
    reason: str
    deleted_at: datetime
    policy_version: str
    actor: str


__all__ = ["RetentionPolicy", "Tombstone"]

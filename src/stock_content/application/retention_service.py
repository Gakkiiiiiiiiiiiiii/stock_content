"""Application port for auditable retention and tombstoning."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from stock_content.domain.retention_policy import RetentionPolicy, Tombstone


class TombstoneRepository(Protocol):
    def append(self, tombstone: Tombstone) -> Tombstone: ...
    def get(self, artifact_id: str) -> Tombstone | None: ...


class InMemoryTombstoneRepository:
    def __init__(self) -> None:
        self._rows: dict[str, Tombstone] = {}

    def append(self, tombstone: Tombstone) -> Tombstone:
        existing = self._rows.get(tombstone.artifact_id)
        if existing is not None:
            return existing
        self._rows[tombstone.artifact_id] = tombstone
        return tombstone

    def get(self, artifact_id: str) -> Tombstone | None:
        return self._rows.get(artifact_id)


class RetentionService:
    def __init__(self, repository: TombstoneRepository | None = None) -> None:
        self.repository = repository or InMemoryTombstoneRepository()

    def should_tombstone(self, created_at: datetime, policy: RetentionPolicy, *, now: datetime | None = None) -> bool:
        return policy.is_expired(created_at, now=now)

    def tombstone(
        self, artifact_id: str, policy: RetentionPolicy, *, reason: str, actor: str,
        now: datetime | None = None,
    ) -> Tombstone:
        event = Tombstone(artifact_id, reason, now or datetime.now(UTC), policy.policy_version, actor)
        return self.repository.append(event)

    def audit(self, artifact_id: str) -> Tombstone | None:
        return self.repository.get(artifact_id)


__all__ = ["InMemoryTombstoneRepository", "RetentionService", "TombstoneRepository"]

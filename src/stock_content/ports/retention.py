"""Retention persistence and byte-deletion boundaries."""
from __future__ import annotations

from typing import Protocol

from stock_content.domain.retention import RetentionExecution, Tombstone


class TombstoneRepository(Protocol):
    def get(self, tombstone_id: str) -> Tombstone | None: ...
    def insert_immutable(self, tombstone: Tombstone) -> Tombstone: ...


class RetentionExecutionRepository(TombstoneRepository, Protocol):
    def get_execution(self, tombstone_id: str) -> RetentionExecution | None: ...
    def prepare_execution(self, tombstone: Tombstone) -> RetentionExecution: ...
    def mark_deleted(self, tombstone_id: str) -> RetentionExecution: ...
    def mark_failed(self, tombstone_id: str, error_code: str) -> RetentionExecution: ...


class ArtifactByteDeleter(Protocol):
    """Private adapter keyed by artifact id; durable state never sees locators."""

    def delete(self, artifact_id: str, *, idempotency_key: str) -> None: ...


__all__ = ["ArtifactByteDeleter", "RetentionExecutionRepository", "TombstoneRepository"]

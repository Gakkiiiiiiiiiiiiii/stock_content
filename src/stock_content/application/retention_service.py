"""Fail-closed retention planning and auditable byte deletion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from stock_content.domain.retention import RetentionCandidate, RetentionExecutionState, RetentionPolicy, Tombstone
from stock_content.ports.retention import ArtifactByteDeleter, TombstoneRepository


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    candidate: RetentionCandidate
    tombstone: Tombstone | None
    dry_run: bool
    action: str


class RetentionService:
    def __init__(self, policy: RetentionPolicy, tombstones: TombstoneRepository) -> None:
        self._policy = policy
        self._tombstones = tombstones

    def plan(
        self, candidate: RetentionCandidate, *, now: datetime | None = None, dry_run: bool = True
    ) -> RetentionPlan:
        now = now or datetime.now(UTC)
        if candidate.legal_hold:
            return RetentionPlan(candidate, None, dry_run, "LEGAL_HOLD")
        expires_at = self._policy.expires_at(candidate.artifact_class, candidate.created_at)
        if now < expires_at:
            return RetentionPlan(candidate, None, dry_run, "KEEP")
        tombstone = Tombstone.for_candidate(candidate, expired_at=expires_at, now=now)
        if dry_run:
            return RetentionPlan(candidate, tombstone, True, "AUDIT_TOMBSTONE_PLANNED")
        existing = self._tombstones.get(tombstone.tombstone_id)
        persisted = existing or self._tombstones.insert_immutable(tombstone)
        return RetentionPlan(candidate, persisted, False, "AUDIT_TOMBSTONE_RECORDED")

    def execute(
        self,
        candidate: RetentionCandidate,
        deleter: ArtifactByteDeleter,
        *,
        now: datetime | None = None,
        dry_run: bool = True,
    ) -> RetentionPlan:
        """Delete only after a durable intent exists; retries use its stable id.

        An adapter failure leaves a durable DELETE_FAILED record and raises no
        success-looking result.  Legal holds and non-expired data never touch
        the byte adapter.
        """
        planned = self.plan(candidate, now=now, dry_run=dry_run)
        if planned.tombstone is None or dry_run:
            return planned
        repository = self._tombstones
        if not all(hasattr(repository, name) for name in ("prepare_execution", "mark_deleted", "mark_failed")):
            raise RuntimeError("RETENTION_EXECUTION_REPOSITORY_REQUIRED")
        execution = repository.prepare_execution(planned.tombstone)
        if execution.state is RetentionExecutionState.DELETED:
            return RetentionPlan(candidate, planned.tombstone, False, "RETENTION_DELETED")
        try:
            deleter.delete(candidate.artifact_id, idempotency_key=planned.tombstone.tombstone_id)
        except Exception as exc:  # adapters must not leak private locators
            repository.mark_failed(planned.tombstone.tombstone_id, _safe_error_code(exc))
            return RetentionPlan(candidate, planned.tombstone, False, "RETENTION_DELETE_FAILED")
        repository.mark_deleted(planned.tombstone.tombstone_id)
        return RetentionPlan(candidate, planned.tombstone, False, "RETENTION_DELETED")


def _safe_error_code(exc: Exception) -> str:
    # Error class names are safe diagnostics; never persist adapter messages
    # because they can include signed URLs, object keys, or credential values.
    return (type(exc).__name__.upper() or "DELETE_FAILED")[:64]


__all__ = ["RetentionPlan", "RetentionService"]

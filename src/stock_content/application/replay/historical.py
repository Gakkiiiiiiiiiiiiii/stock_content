"""Historical replay closure validation."""
from __future__ import annotations

from typing import Any

from stock_content.application.replay.errors import ReplayIntegrityError
from stock_content.domain.initial_verification import TERMINAL_STATUSES


class ReplayHistoricalMixin:
    def _verify_snapshot_ancestry(self, snapshot: Any) -> dict[str, Any]:
        """Validate refresh/replay snapshot parents as a finite deterministic DAG."""
        if self._snapshots is None:
            return {"checked": False, "snapshot_ids": []}
        visiting: set[str] = set()
        visited: set[str] = set()

        def walk(item: Any) -> None:
            identifier = str(item.content_snapshot_id)
            if identifier in visiting:
                raise ReplayIntegrityError("REPLAY_LINEAGE_CYCLE", f"snapshot parent cycle includes {identifier}")
            if identifier in visited:
                return
            visiting.add(identifier)
            for parent_id in (item.parent_snapshot_id, item.supersedes_snapshot_id):
                if not parent_id:
                    continue
                parent = self._snapshots.get(str(parent_id))
                if parent is None:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"snapshot {identifier} references missing parent {parent_id}")
                walk(parent)
            visiting.remove(identifier)
            visited.add(identifier)

        walk(snapshot)
        return {"checked": True, "snapshot_ids": sorted(visited)}

    def _validate_verification_closure(self, snapshot: Any, verification_artifact: Any) -> None:
        """Validate fixed historical Job/Result references without latest reads."""
        entries = [
            item for item in (getattr(verification_artifact, "results", ()) or ())
            if hasattr(item, "provider")
        ]
        if not entries:
            return  # legacy bare VerificationResult artifact
        if self._verification is None:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_MISSING",
                "verification repository is unavailable for exact closure",
            )
        candidate = snapshot.created_at
        if candidate.tzinfo is None:
            from datetime import UTC
            candidate = candidate.replace(tzinfo=UTC)
        for entry in entries:
            if entry.status == "VERIFICATION_PENDING":
                job_id = str(entry.verification_job_id or "")
                job = self._verification.get_job(job_id) if job_id else None
                if job is None:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_MISSING",
                        f"verification job row {job_id} is missing",
                    )
                created_at = job.created_at
                if created_at is None:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", f"job {job_id} has no created_at")
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=candidate.tzinfo)
                if str(job.claim_id) != str(entry.claim_id) or str(job.provider) != str(entry.provider):
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", f"job {job_id} lineage mismatch")
                if created_at > candidate:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        f"job {job_id} was created after snapshot",
                    )
                # Intentionally do not inspect job.status: current execution
                # state is not part of historical Snapshot truth.
                continue
            verification_id = str(entry.verification_id or "")
            row = self._verification.get_result(verification_id) if verification_id else None
            if row is None:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_MISSING",
                    f"verification result row {verification_id} is missing",
                )
            if (
                str(row.claim_id) != str(entry.claim_id)
                or str(row.provider) != str(entry.provider)
                or str(row.status) != str(entry.status)
                or str(row.status) not in TERMINAL_STATUSES
                or (
                    entry.result is not None
                    and dict(row.result_payload or {}) != entry.result.model_dump(mode="json")
                )
            ):
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID", f"verification result {verification_id} lineage mismatch"
                )
            available_at = row.available_at
            if available_at is None:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID", f"verification result {verification_id} has no available_at"
                )
            if available_at.tzinfo is None:
                available_at = available_at.replace(tzinfo=candidate.tzinfo)
            if available_at > candidate:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID", f"verification result {verification_id} is future-dated"
                )

__all__ = ["ReplayHistoricalMixin"]

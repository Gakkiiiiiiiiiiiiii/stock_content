"""ContentSnapshot Replay V2 and immutable lineage verification.

This module is the stable public façade. Implementation concerns live in the
``replay`` package so callers retain the historical import path while the
application layer has explicit integrity and reprocessing seams.
"""
from __future__ import annotations

from typing import Any

from stock_content.application.replay.errors import ReplayIntegrityError
from stock_content.application.replay.historical import ReplayHistoricalMixin
from stock_content.application.replay.integrity import ReplayIntegrityMixin
from stock_content.application.replay.reprocess import ReplayReprocessMixin
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.lineage import lineage_of


class ReplayService(ReplayIntegrityMixin, ReplayHistoricalMixin, ReplayReprocessMixin):
    """Run VERIFY_LINEAGE, REPROCESS, or MIGRATION_REPLAY."""

    _MODES = ("VERIFY_LINEAGE", "REPROCESS", "MIGRATION_REPLAY")
    _RUNTIME_OPTIONS = frozenset(
        {
            "idempotency_key", "trace_id", "decision_id", "replay_raw_storage_uri",
            "replay_snapshot_kind", "replay_parent_snapshot_id", "replay_supersedes_snapshot_id",
            "replay_pipeline_version", "replay_lifecycle_timestamp",
            "temporal_reference_provider",
        }
    )

    def __init__(self, snapshots: SnapshotService, *, artifact_repository: Any | None = None,
                 signal_outbox: Any | None = None, task_repository: Any | None = None,
                 pipeline: Any | None = None, claim_repository: Any | None = None,
                 occurrence_repository: Any | None = None,
                 lifecycle_repository: Any | None = None,
                 verification_repository: Any | None = None,
                 historical_projector: Any | None = None,
                 temporal_reference_snapshot_provider: Any | None = None) -> None:
        self._snapshots = snapshots
        self._artifacts = artifact_repository
        self._signal_outbox = signal_outbox
        self._tasks = task_repository
        self._pipeline = pipeline
        self._claims = claim_repository
        self._occurrences = occurrence_repository
        self._lifecycle = lifecycle_repository
        self._verification = verification_repository
        self._historical_projector = historical_projector
        self._reference_snapshots = temporal_reference_snapshot_provider

    def replay(self, content_snapshot_id: str, *, mode: str | None = None,
               pipeline_version: str | None = None,
               overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        snapshot = self._snapshots.get(content_snapshot_id)
        if snapshot is None:
            return {"error": "SNAPSHOT_NOT_FOUND", "content_snapshot_id": content_snapshot_id}
        requested = str(mode or "VERIFY_LINEAGE").upper()
        if requested == "EXACT":
            requested = "VERIFY_LINEAGE"
        if requested not in self._MODES:
            return {"error": "INVALID_REPLAY_MODE", "content_snapshot_id": content_snapshot_id,
                    "supported_modes": list(self._MODES)}
        try:
            lineage = self._verify_lineage(snapshot)
            self._validate_historical_projection(snapshot)
            if requested == "VERIFY_LINEAGE":
                result = {
                    "content_snapshot_id": content_snapshot_id, "mode": "VERIFY_LINEAGE",
                    "replay_mode": "EXACT" if mode is None else "VERIFY_LINEAGE",
                    "identity_match": lineage["identity_match"],
                    "recomputed_snapshot_id": lineage["recomputed_snapshot_id"],
                    "artifact_ids": dict(snapshot.artifact_ids),
                    "lineage": lineage_of(snapshot).to_dict(),
                    "artifact_validation": lineage["artifact_validation"],
                }
                if not lineage["identity_match"]:
                    result.update({"error": "REPLAY_IDENTITY_MISMATCH",
                                   "detail": "recomputed snapshot identity differs from persisted id"})
                return result
            if not lineage["identity_match"]:
                return {
                    "error": "REPLAY_IDENTITY_MISMATCH",
                    "detail": "recomputed snapshot identity differs from persisted id",
                    "content_snapshot_id": content_snapshot_id,
                    "mode": requested,
                    "recomputed_snapshot_id": lineage["recomputed_snapshot_id"],
                }
            return self._reprocess(snapshot, requested, pipeline_version, overrides)
        except ReplayIntegrityError as exc:
            return exc.to_dict(content_snapshot_id=content_snapshot_id)

    def check_current_verification_liveness(self, snapshot: Any) -> dict[str, Any]:
        """Check only the current source head, never historical closure."""
        verification = self._verification
        if verification is None or self._artifacts is None:
            return {"checked": False}
        if hasattr(self._snapshots, "list_for_source"):
            siblings = self._snapshots.list_for_source(snapshot.source_type, snapshot.source_ref)
            if siblings:
                current = max(siblings, key=lambda item: (item.created_at, item.content_snapshot_id))
                if current.content_snapshot_id != snapshot.content_snapshot_id:
                    return {"checked": False, "status": "NOT_CURRENT_HEAD"}
        verification_id = str((snapshot.artifact_ids or {}).get("verification") or "")
        artifact = self._artifacts.get(verification_id) if verification_id else None
        pending = [
            item for item in (getattr(artifact, "results", ()) or ())
            if getattr(item, "status", None) == "VERIFICATION_PENDING"
        ]
        if not pending:
            return {"checked": True, "status": "NOT_PENDING"}
        for item in pending:
            job = verification.get_job(str(getattr(item, "verification_job_id", "")))
            if job is None:
                return {"checked": True, "error": "DANGLING_CURRENT_VERIFICATION_PENDING"}
            if str(job.status) in {"VERIFICATION_PENDING", "PENDING", "LEASED", "RETRYABLE"}:
                continue
            return {"checked": True, "error": "DANGLING_CURRENT_VERIFICATION_PENDING"}
        return {"checked": True, "status": "LIVE"}

    def _validate_historical_projection(self, snapshot: Any) -> None:
        """Replay claim state through the same append-only projector as formal queries."""
        projector = self._historical_projector
        if projector is None or self._artifacts is None:
            return
        mapping = dict(snapshot.artifact_ids or {})
        claims_artifact = self._artifacts.get(str(mapping.get("claims") or ""))
        lifecycle_artifact = self._artifacts.get(str(mapping.get("lifecycle") or ""))
        claim_ids = {
            str(item.claim_id if hasattr(item, "claim_id") else item)
            for item in (getattr(claims_artifact, "claims", ()) or ())
        }
        if not claim_ids:
            return
        business_as_of = getattr(lifecycle_artifact, "lifecycle_business_as_of", None)
        knowledge_as_of = getattr(lifecycle_artifact, "lifecycle_knowledge_as_of", None)
        availability_as_of = getattr(snapshot, "created_at", None)
        if business_as_of is None or knowledge_as_of is None or availability_as_of is None:
            raise ReplayIntegrityError(
                "REPLAY_HISTORICAL_CLOCK_MISSING",
                "snapshot lifecycle closure lacks immutable replay clocks",
            )
        for claim_id in sorted(claim_ids):
            try:
                projection = projector.project(
                    claim_id,
                    business_as_of=business_as_of,
                    knowledge_as_of=knowledge_as_of,
                    availability_as_of=availability_as_of,
                    content_snapshot_id=snapshot.content_snapshot_id,
                )
            except ValueError as exc:
                raise ReplayIntegrityError(
                    "REPLAY_HISTORICAL_LINEAGE_INVALID",
                    f"claim {claim_id} append-only history failed validation",
                    claim_id=claim_id,
                ) from exc
            if projection is None or projection.get("legacy_history_incomplete"):
                raise ReplayIntegrityError(
                    "REPLAY_HISTORICAL_LINEAGE_INCOMPLETE",
                    f"claim {claim_id} cannot be projected from append-only history",
                    claim_id=claim_id,
                )


__all__ = ["ReplayIntegrityError", "ReplayService"]

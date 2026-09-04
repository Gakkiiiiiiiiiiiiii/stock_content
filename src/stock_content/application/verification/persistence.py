"""Persistence and lineage lookup helpers for verification refresh."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy import select

from stock_content.adapters.postgres.models import (
    ClaimArtifactMemberRow,
    ClaimVerificationResultRow,
    ContentArtifactEdgeRow,
    ContentArtifactRow,
    ContentSnapshotArtifactRow,
    ContentSnapshotRow,
    SignalOutboxRow,
)
from stock_content.adapters.postgres.repositories.snapshot_repository import (
    _compare_candidate_to_existing_row,
    _compare_existing_identity,
    _validate_snapshot_artifacts,
    _validate_snapshot_candidate,
    _validate_snapshot_row,
)
from stock_content.application.verification.closure import _validate_refresh_verification_closure
from stock_content.domain.artifacts import (
    VerificationArtifact,
    canonical_json,
    serialize_artifact,
)


class VerificationPersistenceMixin:
    def _load_parent_snapshot(self, session, snapshot_id: str | None):
        if snapshot_id:
            row = session.get(ContentSnapshotRow, snapshot_id)
            if row is None:
                raise KeyError("parent content snapshot not found")
            return self._row_to_snapshot(row, session)
        raise ValueError("parent_snapshot_id is required for verification refresh")

    @staticmethod
    def _row_to_snapshot(row: ContentSnapshotRow, session=None):
        from stock_content.adapters.postgres.repositories.snapshot_repository import _row_to_snapshot

        return _row_to_snapshot(row, session)

    def resolve_parent_snapshot_id(self, claim_id: str) -> str:
        """Resolve the latest snapshot containing this claim's persisted artifact.

        This deliberately joins claim-artifact membership to snapshot-artifact
        membership; a globally newest snapshot from another source is never a
        valid parent.
        """
        with self._sessions() as session:
            rows = session.scalars(
                select(ContentSnapshotRow)
                .join(
                    ContentSnapshotArtifactRow,
                    ContentSnapshotArtifactRow.content_snapshot_id == ContentSnapshotRow.content_snapshot_id,
                )
                .join(
                    ClaimArtifactMemberRow,
                    ClaimArtifactMemberRow.artifact_id == ContentSnapshotArtifactRow.artifact_id,
                )
                .where(ClaimArtifactMemberRow.claim_id == claim_id)
                .order_by(ContentSnapshotRow.created_at.desc(), ContentSnapshotRow.content_snapshot_id.desc())
            ).all()
            unique = {row.content_snapshot_id: row for row in rows}
            if not unique:
                raise KeyError(f"no snapshot lineage for claim: {claim_id}")
            return next(iter(unique.values())).content_snapshot_id

    @staticmethod
    def _snapshot_contains_claim(session, snapshot, claim_id: str) -> bool:
        claim_artifact_id = str((snapshot.artifact_ids or {}).get("claims") or "")
        if not claim_artifact_id:
            return False
        member = session.scalar(
            select(ClaimArtifactMemberRow).where(
                ClaimArtifactMemberRow.artifact_id == claim_artifact_id,
                ClaimArtifactMemberRow.claim_id == claim_id,
            )
        )
        return member is not None

    @staticmethod
    def _find_snapshot_for_result(session, verification_id: str, claim_id: str):
        """Return (snapshot_id, signal_id) for an already-applied result."""
        from stock_content.adapters.postgres.repositories.snapshot_repository import _row_to_snapshot

        result_row = session.get(ClaimVerificationResultRow, verification_id)
        if result_row is None or result_row.claim_id != claim_id:
            return None
        expected_result = dict(result_row.result_payload or {})
        rows = session.scalars(select(ContentSnapshotRow).order_by(ContentSnapshotRow.created_at.desc())).all()
        for row in rows:
            snapshot = _row_to_snapshot(row, session)
            if not VerificationPersistenceMixin._snapshot_contains_claim(session, snapshot, claim_id):
                continue
            verification_id_in_snapshot = str((snapshot.artifact_ids or {}).get("verification") or "")
            if not verification_id_in_snapshot:
                continue
            artifact_row = session.get(ContentArtifactRow, verification_id_in_snapshot)
            if artifact_row is None:
                continue
            payload = dict(artifact_row.payload or {})
            results = payload.get("results") or []
            if not any(
                isinstance(item, dict)
                and str(item.get("claim_id")) == claim_id
                and canonical_json({
                    key: value for key, value in item.items()
                    if key not in {"provider", "verification_id", "verification_job_id"}
                }) == canonical_json(expected_result)
                for item in results
            ):
                continue
            outbox = session.scalar(
                select(SignalOutboxRow).where(
                    SignalOutboxRow.content_snapshot_id == snapshot.content_snapshot_id,
                    SignalOutboxRow.claim_id == claim_id,
                )
            )
            return snapshot.content_snapshot_id, outbox.signal_id if outbox else None
        return None

    @staticmethod
    def _persist_artifact(session, artifact: VerificationArtifact) -> None:
        payload = json.loads(canonical_json(serialize_artifact(artifact)))
        row = session.get(ContentArtifactRow, artifact.artifact_id)
        if row is not None:
            if row.content_hash != artifact.content_hash:
                raise ValueError("verification artifact id conflict")
            return
        session.add(
            ContentArtifactRow(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                schema_version=artifact.schema_version,
                producer_stage=artifact.producer_stage,
                producer_version=artifact.producer_version,
                content_hash=artifact.content_hash,
                parent_artifact_ids=list(artifact.parent_artifact_ids),
                payload=payload,
                created_at=artifact.created_at,
            )
        )
        for parent_id in artifact.parent_artifact_ids:
            session.add(
                ContentArtifactEdgeRow(
                    edge_id=hashlib.sha256(f"{artifact.artifact_id}:{parent_id}".encode()).hexdigest(),
                    artifact_id=artifact.artifact_id,
                    parent_artifact_id=parent_id,
                )
            )

    @staticmethod
    def _persist_snapshot(session, snapshot) -> None:
        payload = _validate_snapshot_candidate(snapshot)
        _validate_snapshot_artifacts(
            session,
            artifact_ids=dict(snapshot.artifact_ids),
            schema_version=snapshot.schema_version,
            snapshot_id=snapshot.content_snapshot_id,
        )
        existing = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
        if existing is not None:
            _validate_snapshot_row(session, existing)
            existing_identity = dict(existing.identity or {})
            _compare_existing_identity(existing_identity, payload, snapshot_id=snapshot.content_snapshot_id)
            _compare_candidate_to_existing_row(snapshot, existing, existing_identity)
            return
        session.add(
            ContentSnapshotRow(
                content_snapshot_id=snapshot.content_snapshot_id,
                source_type=snapshot.source_type,
                source_ref=snapshot.source_ref,
                source_content_hash=snapshot.source_content_hash,
                identity=payload,
                artifact_ids=dict(snapshot.artifact_ids),
                quant_market_snapshot_ids=list(snapshot.quant_market_snapshot_ids),
                pipeline_version=snapshot.pipeline_version,
                schema_version=snapshot.schema_version,
                code_sha=snapshot.code_sha,
                config_hash=snapshot.config_hash,
                source_artifact_id=snapshot.source_artifact_id,
                artifact_root_hash=snapshot.artifact_root_hash,
                snapshot_kind=snapshot.snapshot_kind,
                parent_snapshot_id=snapshot.parent_snapshot_id,
                supersedes_snapshot_id=snapshot.supersedes_snapshot_id,
                producer_manifest=dict(snapshot.producer_manifest),
                created_at=snapshot.created_at,
            )
        )
        for slot, artifact_id in snapshot.artifact_ids.items():
            session.add(
                ContentSnapshotArtifactRow(
                    member_id=hashlib.sha256(
                        f"{snapshot.content_snapshot_id}:{slot}:{artifact_id}".encode()
                    ).hexdigest(),
                    content_snapshot_id=snapshot.content_snapshot_id,
                    artifact_id=artifact_id,
                    slot=slot,
                )
            )
        session.flush()
        _validate_refresh_verification_closure(session, snapshot)

    @staticmethod
    def _persist_outbox(session, payload: dict, now: datetime) -> None:
        signal_id = str(payload["signal_id"])
        existing = session.scalar(select(SignalOutboxRow).where(SignalOutboxRow.signal_id == signal_id))
        if existing is not None:
            if dict(existing.payload or {}) != payload:
                raise ValueError("signal outbox payload conflict")
            return
        session.add(
            SignalOutboxRow(
                outbox_id="outbox-" + hashlib.sha256(signal_id.encode()).hexdigest()[:32],
                signal_id=signal_id,
                content_snapshot_id=payload.get("content_snapshot_id"),
                claim_id=payload.get("claim_id"),
                schema_version=payload.get("signal_schema_version", "content-factor-signal.v4"),
                payload=payload,
                status="PENDING",
                next_attempt_at=now,
                created_at=now,
            )
        )




__all__ = ["VerificationPersistenceMixin"]

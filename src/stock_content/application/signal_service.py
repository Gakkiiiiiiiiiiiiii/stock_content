"""The sole production caller of the pure SignalPolicy."""
from __future__ import annotations

from typing import Any

from stock_content.domain.signal_contract import validate_signal_v4
from stock_content.domain.signal_policy import SignalPolicy


class SignalService:
    def __init__(self, policy: SignalPolicy | None = None) -> None:
        self._policy = policy or SignalPolicy()

    @property
    def policy(self) -> SignalPolicy:
        return self._policy

    def build_signal(self, snapshot: Any, claim: Any, verification: Any, **kwargs: Any) -> dict[str, Any]:
        return self._policy.build_signal(snapshot, claim, verification, **kwargs)

    def build_initial_signals(
        self, snapshot: Any, claims: list[Any], verification: Any,
        *, trace_id: str | None = None, decision_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Build all deterministic eligible projections for a snapshot."""
        return self._policy.build_initial_signals(snapshot, claims, verification,
                                                   trace_id=trace_id, decision_id=decision_id)

    def rebuild_signals(
        self, snapshot: Any, claims: list[Any], verification: Any,
        *, trace_id: str | None = None, decision_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Alias used by the Postgres rebuild path; IDs remain stable."""
        return self._policy.build_initial_signals(snapshot, claims, verification,
                                                   trace_id=trace_id, decision_id=decision_id)

    # Compatibility with callers that use the shorter names.
    build_initial = build_initial_signals
    rebuild = rebuild_signals

    def enqueue_initial(
        self,
        outbox: Any,
        snapshot: Any,
        claim: Any,
        verification: Any,
        *,
        verification_artifact_id: str,
        trace_id: str | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Create eligible non-FACT signals through the durable outbox."""
        payload = self.build_signal(
            snapshot,
            claim,
            verification,
            verification_artifact_id=verification_artifact_id,
            trace_id=trace_id,
            decision_id=decision_id,
        )
        if self._policy.evaluate(claim, verification, snapshot=snapshot).allowed:
            validate_signal_v4(payload)
            outbox.enqueue(payload)
            return payload
        return None

    def signals_for_snapshot(
        self, outbox: Any, snapshot_id: str, *, claim_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Read v4 signals from PostgreSQL outbox/projection, never KnowledgeUnit."""
        return [dict(row.payload or {}) for row in outbox.list_for_snapshot(snapshot_id, claim_id=claim_id)]

    def rebuild_from_postgres(self, database: Any, *, snapshot_policy: str = "latest-verified") -> int:
        """Rebuild outbox rows from relational authority (Qdrant is ignored)."""
        from sqlalchemy import select

        from stock_content.adapters.postgres.models import (
            ClaimArtifactMemberRow,
            ClaimVerificationResultRow,
            ContentArtifactRow,
            ContentSnapshotRow,
            ContentSourceHeadRow,
        )
        from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
        from stock_content.adapters.postgres.repositories.signal_outbox_repository import SignalOutboxRepository
        from stock_content.adapters.postgres.repositories.snapshot_repository import _row_to_snapshot
        from stock_content.domain.artifacts import deserialize_artifact

        claims = SqlClaimRepository(database.session_factory)
        outbox = SignalOutboxRepository(database.session_factory)
        count = 0
        with database.session_factory() as session:
            if snapshot_policy == "latest-verified":
                ids = list(session.scalars(select(ContentSourceHeadRow.latest_verified_snapshot_id)).all())
            elif snapshot_policy == "latest":
                ids = list(session.scalars(select(ContentSourceHeadRow.latest_snapshot_id)).all())
            else:
                ids = [snapshot_policy]
            snapshots = session.scalars(
                select(ContentSnapshotRow).where(ContentSnapshotRow.content_snapshot_id.in_(ids))
            ).all()
            for row in snapshots:
                snapshot = _row_to_snapshot(row)
                verification_id = str((snapshot.artifact_ids or {}).get("verification") or "")
                claims_artifact_id = str((snapshot.artifact_ids or {}).get("claims") or "")
                artifact_row = session.get(ContentArtifactRow, verification_id)
                if artifact_row is None:
                    continue
                verification = deserialize_artifact(dict(artifact_row.payload or {}))
                members = session.scalars(
                    select(ClaimArtifactMemberRow).where(ClaimArtifactMemberRow.artifact_id == claims_artifact_id)
                ).all()
                for member in members:
                    claim = claims.get(member.claim_id)
                    result = next((item for item in verification.results if item.claim_id == member.claim_id), None)
                    if claim is None or result is None:
                        continue
                    result_row = session.scalar(
                        select(ClaimVerificationResultRow)
                        .where(ClaimVerificationResultRow.claim_id == member.claim_id)
                        .order_by(ClaimVerificationResultRow.created_at.desc())
                    )
                    view = result.model_dump(mode="json") | {"provider": result_row.provider if result_row else "quant"}
                    payload = self.build_signal(snapshot, claim, view, verification_artifact_id=verification_id)
                    if self._policy.evaluate(claim, view, snapshot=snapshot).allowed:
                        validate_signal_v4(payload)
                        outbox.enqueue(payload)
                        count += 1
        return count


__all__ = ["SignalService"]

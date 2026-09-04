"""The sole production caller of the pure SignalPolicy."""

from __future__ import annotations

from typing import Any

from stock_content.domain.bitemporal_query import FormalContentSignalQueryV2
from stock_content.domain.signal_contract import upgrade_signal_v5, validate_signal_v4, validate_signal_v5
from stock_content.domain.signal_contract_v5_1 import (
    AUTHORITY_FORMAL,
    CONTRACT_CHECKSUM,
    CONTRACT_NAME,
    formal_signal_id,
    validate_signal_v5_1,
)
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
        self,
        snapshot: Any,
        claims: list[Any],
        verification: Any,
        *,
        trace_id: str | None = None,
        decision_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build all deterministic eligible projections for a snapshot."""
        return self._policy.build_initial_signals(
            snapshot, claims, verification, trace_id=trace_id, decision_id=decision_id
        )

    def rebuild_signals(
        self,
        snapshot: Any,
        claims: list[Any],
        verification: Any,
        *,
        trace_id: str | None = None,
        decision_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Alias used by the Postgres rebuild path; IDs remain stable."""
        return self._policy.build_initial_signals(
            snapshot, claims, verification, trace_id=trace_id, decision_id=decision_id
        )

    def build_signal_v5(self, projection: dict[str, Any]) -> dict[str, Any]:
        """Validate/build the lineage-only v5 projection from SQL data."""
        return upgrade_signal_v5(projection)

    def build_signal_v5_1(self, projection: dict[str, Any], query: FormalContentSignalQueryV2) -> dict[str, Any]:
        """Build a complete formal signal from an immutable historical projection.

        The projection's legacy signal id is intentionally ignored.  Formal
        identity is scoped to the complete query (all three clocks, snapshot,
        query and policy), so a materialized v4/v5 row can never be promoted
        into a v5.1 fact by reusing its id.
        """
        lifecycle = dict(projection.get("lifecycle_as_of") or projection.get("lifecycle") or {})
        lifecycle_status = lifecycle.get("status") or lifecycle.get("to_status")
        lifecycle_known_from = _as_rfc3339(lifecycle.get("known_from"))
        lifecycle_artifact_id = lifecycle.get("artifact_id") or lifecycle.get("lifecycle_artifact_id")
        if (
            not all((lifecycle_status, lifecycle_known_from, lifecycle_artifact_id))
            or any(str(value).strip().lower() == "unknown" for value in (
                lifecycle_status, lifecycle_known_from, lifecycle_artifact_id
            ))
        ):
            raise ValueError("formal signal requires a complete historical lifecycle lineage")
        occurrence_id = str(projection.get("occurrence_id") or "")
        semantic_segment_id = str(projection.get("semantic_segment_id") or "")
        claim_id = str(projection.get("claim_id") or "")
        available_from = _as_rfc3339(projection.get("available_from"))
        if not claim_id or not occurrence_id or not semantic_segment_id or not available_from:
            raise ValueError("formal signal requires immutable claim occurrence and availability lineage")
        clocks = {
            field: getattr(query, field).isoformat().replace("+00:00", "Z")
            for field in ("business_as_of", "knowledge_as_of", "availability_as_of")
        }
        signal = {
            key: projection[key]
            for key in (
            "claim_id", "occurrence_id", "semantic_segment_id", "asserted_at",
                "source_available_at", "available_from", "temporal_bindings",
            )
            if key in projection
        }
        signal.setdefault("asserted_at", projection.get("asserted_at"))
        signal.setdefault("source_available_at", projection.get("source_available_at"))
        signal["available_from"] = available_from
        signal["asserted_at"] = _as_rfc3339(signal.get("asserted_at"))
        signal["source_available_at"] = _as_rfc3339(signal.get("source_available_at"))
        producer_commit = str(projection.get("producer_commit") or "")
        if not producer_commit:
            raise ValueError("formal signal requires snapshot producer_commit lineage")
        signal.update({
            "contract": CONTRACT_NAME,
            "contract_checksum": CONTRACT_CHECKSUM,
            "authority": AUTHORITY_FORMAL,
            "formal_eligible": True,
            "content_snapshot_id": query.content_snapshot_id,
            **clocks,
            "lifecycle_as_of": {
                "status": lifecycle_status,
                "known_from": lifecycle_known_from,
                "artifact_id": lifecycle_artifact_id,
            },
            "producer_commit": producer_commit,
            # The request policy is authoritative.  A stale materialized row
            # must not change the identity of a formal query.
            "signal_policy_version": query.signal_policy_version,
            "temporal_bindings": list(projection.get("temporal_bindings") or []),
            "evidence_refs": list(projection.get("evidence_refs") or []),
            "source_availability_quality": str(projection.get("source_availability_quality") or "EXACT"),
        })
        signal["signal_id"] = formal_signal_id(
            claim_id=claim_id,
            occurrence_id=occurrence_id,
            semantic_segment_id=semantic_segment_id,
            content_snapshot_id=query.content_snapshot_id,
            business_as_of=clocks["business_as_of"],
            knowledge_as_of=clocks["knowledge_as_of"],
            availability_as_of=clocks["availability_as_of"],
            query_id=query.query_id,
            signal_policy_version=signal["signal_policy_version"],
        )
        return validate_signal_v5_1(signal)

    @staticmethod
    def validate_signal_v5(signal: dict[str, Any]) -> dict[str, Any]:
        return validate_signal_v5(signal)

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


def _as_rfc3339(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    return str(value).replace("+00:00", "Z")

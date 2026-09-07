"""PostgreSQL persistence and SQL-authoritative formal bundle reads."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ClaimArtifactMemberRow,
    ClaimOccurrenceEvidenceRow,
    ClaimOccurrenceRow,
    ClaimStateEventRow,
    ContentArtifactRow,
    ContentKnowledgeBundleRow,
    ContentSnapshotRow,
    FinancialClaimRow,
    SourceArtifactMetadataRow,
)
from stock_content.application.historical_claim_projector import HistoricalClaimProjector
from stock_content.domain.claim_state_event import ClaimStateEvent
from stock_content.domain.knowledge_bundle import KnowledgeBundleRequest, sha256


class PostgresKnowledgeBundleRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def insert(self, bundle: dict[str, Any]) -> dict[str, Any]:
        with self._sessions.begin() as session:
            inserted = self._insert_ignore_conflict(session, bundle)
            if inserted:
                return dict(bundle)
            row = session.scalar(
                select(ContentKnowledgeBundleRow).where(
                    or_(
                        ContentKnowledgeBundleRow.bundle_id == bundle["bundle_id"],
                        ContentKnowledgeBundleRow.bundle_hash == bundle["bundle_hash"],
                    )
                )
            )
            if row is None:
                # The conflict was not one of our immutable keys.  Do not
                # convert an unrelated database failure into idempotency.
                raise ValueError("bundle insert conflict without immutable row")
            if row.bundle_hash != bundle["bundle_hash"] or dict(row.payload) != bundle:
                raise ValueError("immutable bundle id collision")
            return dict(row.payload)
        return dict(bundle)

    @staticmethod
    def _values(bundle: dict[str, Any]) -> dict[str, Any]:
        return {
            "bundle_id": bundle["bundle_id"],
            "bundle_hash": bundle["bundle_hash"],
            "content_snapshot_id": bundle["content_snapshot_id"],
            "request_hash": bundle["request_hash"],
            "contract_version": bundle["contract"],
            "payload": bundle,
            "producer_git_commit": bundle["producer"]["git_commit"],
            "pipeline_version": bundle["producer"]["pipeline_version"],
        }

    def _insert_ignore_conflict(self, session, bundle: dict[str, Any]) -> bool:
        """Atomically create an immutable Bundle or identify its unique race.

        PostgreSQL and SQLite both use their native ``ON CONFLICT DO
        NOTHING`` form.  The savepoint path is intentionally narrow for other
        SQLAlchemy test dialects and re-raises every non-unique failure.
        """
        values = self._values(bundle)
        dialect = session.bind.dialect.name
        if dialect == "postgresql":
            result = session.execute(
                postgresql_insert(ContentKnowledgeBundleRow)
                .values(**values)
                .on_conflict_do_nothing()
                # psycopg may expose an indeterminate rowcount for conflict
                # ignoring inserts.  The immutable bundle id is returned
                # only when this transaction won the first write.
                .returning(ContentKnowledgeBundleRow.bundle_id)
            )
            return result.scalar_one_or_none() is not None
        if dialect == "sqlite":
            result = session.execute(sqlite_insert(ContentKnowledgeBundleRow).values(**values).on_conflict_do_nothing())
            return bool(result.rowcount)
        try:
            with session.begin_nested():
                session.add(ContentKnowledgeBundleRow(**values))
                session.flush()
            return True
        except IntegrityError:
            return False

    def get(self, bundle_id: str) -> dict[str, Any] | None:
        with self._sessions() as session:
            row = session.get(ContentKnowledgeBundleRow, bundle_id)
        return None if row is None else dict(row.payload)


class PostgresKnowledgeBundleAuthority:
    """Formal data comes from SQL snapshot membership, never the search index."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def read_bundle_source(self, request: KnowledgeBundleRequest) -> dict[str, Any] | None:
        with self._sessions() as session:
            snapshot = session.get(ContentSnapshotRow, request.content_snapshot_id)
            if snapshot is None:
                return None
            snapshot_available = _utc(snapshot.created_at) <= request.availability_as_of
            source_artifact = session.get(
                ContentArtifactRow, snapshot.source_artifact_id or (snapshot.artifact_ids or {}).get("source")
            )
            metadata = session.get(SourceArtifactMetadataRow, source_artifact.artifact_id) if source_artifact else None
            source = {
                "source_type": metadata.source_type if metadata else None,
                "source_identity_hash": metadata.source_identity_hash if metadata else None,
                "source_version_id": metadata.source_version_id if metadata else None,
                "canonical_url": metadata.canonical_url if metadata else None,
                "author": metadata.author if metadata else None,
                "published_at": _iso(metadata.published_at) if metadata and metadata.published_at else None,
                "source_content_hash": metadata.source_content_hash if metadata else None,
            }
            claim_artifact_id = (snapshot.artifact_ids or {}).get("claims")
            evidence_artifact = session.get(ContentArtifactRow, (snapshot.artifact_ids or {}).get("evidence"))
            if not claim_artifact_id or evidence_artifact is None:
                return {"snapshot_available": snapshot_available, "source": source, "items": []}
            evidence_map = {
                str(item.get("evidence_id")): item
                for item in dict(evidence_artifact.payload or {}).get("evidences", [])
                if isinstance(item, dict)
            }
            rows = session.execute(
                select(FinancialClaimRow, ClaimOccurrenceRow)
                .join(ClaimArtifactMemberRow, ClaimArtifactMemberRow.claim_id == FinancialClaimRow.claim_id)
                .join(ClaimOccurrenceRow, ClaimOccurrenceRow.claim_id == FinancialClaimRow.claim_id)
                .where(
                    ClaimArtifactMemberRow.artifact_id == claim_artifact_id,
                    FinancialClaimRow.subject_id == request.symbol,
                    FinancialClaimRow.claim_schema_version == "claim.atomic.v1",
                    FinancialClaimRow.grounding_status == "GROUNDED",
                    FinancialClaimRow.legacy_grounding_incomplete.is_(False),
                    ClaimOccurrenceRow.claim_schema_version == "claim.atomic.v1",
                    ClaimOccurrenceRow.grounding_status == "GROUNDED",
                    ClaimOccurrenceRow.legacy_grounding_incomplete.is_(False),
                    ClaimOccurrenceRow.available_from <= request.availability_as_of,
                    ClaimOccurrenceRow.snapshot_committed_at <= request.knowledge_as_of,
                ).order_by(
                    FinancialClaimRow.subject_type,
                    FinancialClaimRow.subject_id,
                    FinancialClaimRow.predicate,
                    ClaimOccurrenceRow.occurrence_id,
                    FinancialClaimRow.claim_id,
                )
            ).all()
            items = []
            for claim, occurrence in rows:
                projection = _historical_projection(session, claim, request)
                if projection is None:
                    # The claim did not exist at one of the requested clocks.
                    # It is not a current-data fallback candidate.
                    continue
                if (
                    projection.get("snapshot_id") != request.content_snapshot_id
                    or projection.get("occurrence_id") != occurrence.occurrence_id
                ):
                    # A claim can recur in another snapshot.  The immutable
                    # event payload, rather than the current occurrence row,
                    # selects the occurrence for this exact snapshot.
                    continue
                lifecycle = projection.get("lifecycle_as_of")
                if not isinstance(lifecycle, dict):
                    raise ValueError("HISTORICAL_LIFECYCLE_AUTHORITY_MISSING")
                if lifecycle.get("status") != "ACTIVE":
                    # A retraction/supersession visible at the requested clocks
                    # excludes the occurrence; it must not be relabelled ACTIVE.
                    continue
                support_status = projection.get("support_status")
                verification_status = projection.get("verification_status")
                if not isinstance(support_status, str) or not support_status:
                    raise ValueError("HISTORICAL_SUPPORT_AUTHORITY_MISSING")
                if not isinstance(verification_status, str) or not verification_status:
                    raise ValueError("HISTORICAL_VERIFICATION_AUTHORITY_MISSING")
                links = session.scalars(
                    select(ClaimOccurrenceEvidenceRow).where(
                        ClaimOccurrenceEvidenceRow.occurrence_id == occurrence.occurrence_id,
                        ClaimOccurrenceEvidenceRow.evidence_role.in_(("PRIMARY", "SECONDARY")),
                    ).order_by(
                        ClaimOccurrenceEvidenceRow.evidence_role,
                        ClaimOccurrenceEvidenceRow.ordinal,
                        ClaimOccurrenceEvidenceRow.evidence_id,
                    )
                ).all()
                evidence = [
                    {
                        "evidence_id": link.evidence_id,
                        "ownership": link.evidence_role,
                        **_citation(evidence_map[link.evidence_id]),
                    }
                    for link in links
                    if link.evidence_id in evidence_map
                ]
                if not any(entry["ownership"] == "PRIMARY" for entry in evidence):
                    continue
                items.append(
                    {
                        "knowledge_id": occurrence.occurrence_id,
                        "claim_id": claim.claim_id,
                        "occurrence_id": occurrence.occurrence_id,
                        "statement": claim.normalized_statement,
                        "subject": {"type": claim.subject_type, "key": claim.subject_id},
                        "predicate": claim.predicate,
                        "object": {"value": claim.value, "unit": claim.unit},
                        "condition": claim.condition_text,
                        "invalidation": claim.invalidation_text,
                        # These three status dimensions are bitemporal
                        # projections from the append-only state ledger.  The
                        # mutable convenience rows are never an authority for
                        # formal Bundle policy.
                        "support_status": support_status,
                        "lifecycle_status": lifecycle["status"],
                        "confidence": claim.source_confidence,
                        "temporal": {
                            "target_start": _iso(claim.period_start or claim.fact_time),
                            "target_end": _iso(claim.period_end or claim.fact_time),
                            "precision": "EXACT",
                        },
                        "evidence": evidence,
                        "verification": {
                            "status": verification_status,
                            "reason_codes": list(projection.get("verification_reason_codes") or []),
                        },
                        "contradiction_group_id": claim.contradiction_group_id,
                        "known_from": lifecycle.get("known_from"),
                        "available_from": projection.get("available_from"),
                        "claim_schema_version": claim.claim_schema_version,
                        "grounding_status": claim.grounding_status,
                        "legacy_grounding_incomplete": claim.legacy_grounding_incomplete,
                    }
                )
            return {
                "snapshot_available": snapshot_available,
                "source_available_from": (
                    _iso(metadata.source_available_from)
                    if metadata and metadata.source_available_from
                    else None
                ),
                "source": source,
                "items": items,
            }


def _historical_projection(session, claim: FinancialClaimRow, request: KnowledgeBundleRequest) -> dict[str, Any] | None:
    """Project one snapshot member through its immutable claim-state chain.

    Bundle reads deliberately fail closed when the historical chain is absent,
    malformed, legacy-marked, or lacks the verification/lifecycle data needed
    for formal use.  A current ``FinancialClaimRow`` may supply immutable claim
    text, but never the as-of support, verification, or lifecycle decision.
    """
    if claim.legacy_history_incomplete:
        raise ValueError("HISTORICAL_CLAIM_LINEAGE_INCOMPLETE")
    rows = session.scalars(
        select(ClaimStateEventRow).where(ClaimStateEventRow.claim_id == claim.claim_id)
    ).all()
    if not rows:
        raise ValueError("HISTORICAL_CLAIM_AUTHORITY_MISSING")
    events = tuple(_event_from_row(row) for row in rows)
    projector = HistoricalClaimProjector(
        events,
        membership=lambda snapshot_id, claim_id: (
            snapshot_id == request.content_snapshot_id and claim_id == claim.claim_id
        ),
        history_incomplete=lambda claim_id: claim_id == claim.claim_id and bool(claim.legacy_history_incomplete),
    )
    projection = projector.project(
        claim.claim_id,
        business_as_of=request.business_as_of,
        knowledge_as_of=request.knowledge_as_of,
        availability_as_of=request.availability_as_of,
        content_snapshot_id=request.content_snapshot_id,
    )
    if projection is not None and projection.get("legacy_history_incomplete"):
        raise ValueError("HISTORICAL_CLAIM_LINEAGE_INCOMPLETE")
    return projection


def _event_from_row(row: ClaimStateEventRow) -> ClaimStateEvent:
    return ClaimStateEvent(
        claim_id=row.claim_id,
        event_type=row.event_type,
        payload=dict(row.payload or {}),
        known_from=_utc(row.known_from) if row.known_from is not None else None,
        business_valid_from=_utc(row.business_valid_from) if row.business_valid_from is not None else None,
        business_valid_to=_utc(row.business_valid_to) if row.business_valid_to is not None else None,
        known_to=_utc(row.known_to) if row.known_to is not None else None,
        source_available_from=_utc(row.source_available_from) if row.source_available_from is not None else None,
        previous_event_hash=row.previous_event_hash,
        event_id=row.claim_state_event_id,
        event_hash=row.event_hash,
        legacy_history_incomplete=row.legacy_history_incomplete,
    )


def _utc(value):
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, time.min, tzinfo=UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value):
    return None if value is None else _utc(value).isoformat().replace("+00:00", "Z")


def _citation(value: dict[str, Any]) -> dict[str, Any]:
    """Project a stored :class:`EvidenceItem` into an immutable public citation.

    EvidenceItem intentionally has no item-level ``content_hash``.  The public
    quote is the provenance primitive, so its hash must be computed from the
    exact canonical JSON representation that the consumer verifies.  Never
    silently substitute an artifact hash: it would authenticate a different
    object than the quote shown to the consumer.
    """
    quote = value.get("normalized_text") or value.get("evidence_text")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("BUNDLE_EVIDENCE_QUOTE_MISSING")
    return {
        "modality": value.get("source_type"),
        "artifact_id": value.get("source_artifact_id"),
        "segment_id": (value.get("locator") or {}).get("segment_id"),
        "start_ms": value.get("start_ms"),
        "end_ms": value.get("end_ms"),
        "quote": quote,
        "quote_hash": sha256(quote),
    }

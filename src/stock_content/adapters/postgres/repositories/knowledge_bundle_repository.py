"""PostgreSQL persistence and SQL-authoritative formal bundle reads."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any

from sqlalchemy import or_, select, text
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
    ContentKnowledgeBundleIdempotencyRow,
    ContentKnowledgeBundleRow,
    ContentSnapshotRow,
    FinancialClaimRow,
    SourceArtifactMetadataRow,
)
from stock_content.application.historical_claim_projector import HistoricalClaimProjector
from stock_content.domain.claim_state_event import ClaimStateEvent
from stock_content.domain.knowledge_bundle import V2_CONTRACT, KnowledgeBundleRequest, sha256
from stock_content.ports.repositories import IdempotencyConflict


class PostgresKnowledgeBundleRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def insert(
        self,
        bundle: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        idempotency_request_hash: str | None = None,
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            if not idempotency_request_hash:
                raise ValueError("idempotency request hash is required")
            return self._insert_idempotent(bundle, idempotency_key, idempotency_request_hash)
        with self._sessions.begin() as session:
            return self._insert_or_get_immutable(session, bundle)

    @staticmethod
    def _idempotency_key_hash(key: str) -> str:
        return sha256({"endpoint": "content-knowledge-bundle", "idempotency_key": key})

    def _insert_idempotent(
        self, bundle: dict[str, Any], idempotency_key: str, idempotency_request_hash: str
    ) -> dict[str, Any]:
        key_hash = self._idempotency_key_hash(idempotency_key)
        with self._sessions.begin() as session:
            # The key-scoped PostgreSQL advisory lock makes the check, Bundle
            # insert and retry binding one serializable critical section. It
            # prevents a losing different-request race from materializing an
            # otherwise unreachable immutable Bundle. SQLite's unique mapping
            # remains a correct development fallback; production requires PG.
            if session.bind.dialect.name == "postgresql":
                session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": key_hash})
            existing = session.get(ContentKnowledgeBundleIdempotencyRow, key_hash)
            if existing is not None:
                return self._resolve_idempotent(session, existing, idempotency_request_hash)

            result = self._insert_or_get_immutable(session, bundle)
            mapping = ContentKnowledgeBundleIdempotencyRow(
                idempotency_key_hash=key_hash,
                idempotency_request_hash=idempotency_request_hash,
                bundle_id=result["bundle_id"],
            )
            try:
                with session.begin_nested():
                    session.add(mapping)
                    session.flush()
            except IntegrityError:
                # SQLite and non-PostgreSQL dialects do not have the advisory
                # lock. Their unique durable mapping still determines the only
                # permitted result after a concurrent insert race.
                existing = session.get(ContentKnowledgeBundleIdempotencyRow, key_hash)
                if existing is None:
                    raise
                return self._resolve_idempotent(session, existing, idempotency_request_hash)
            return result

    def get_idempotent(self, *, idempotency_key: str, idempotency_request_hash: str) -> dict[str, Any] | None:
        key_hash = self._idempotency_key_hash(idempotency_key)
        with self._sessions() as session:
            existing = session.get(ContentKnowledgeBundleIdempotencyRow, key_hash)
            if existing is None:
                return None
            return self._resolve_idempotent(session, existing, idempotency_request_hash)

    @staticmethod
    def _resolve_idempotent(session, mapping, expected_request_hash: str) -> dict[str, Any]:
        if mapping.idempotency_request_hash != expected_request_hash:
            raise IdempotencyConflict()
        row = session.get(ContentKnowledgeBundleRow, mapping.bundle_id)
        if row is None:
            # The FK should make this impossible. Do not report a successful
            # retry when durable state has been tampered with.
            raise ValueError("idempotency mapping references missing bundle")
        return dict(row.payload)

    def _insert_or_get_immutable(self, session, bundle: dict[str, Any]) -> dict[str, Any]:
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
            # The conflict was not one of our immutable keys.  Do not convert
            # an unrelated database failure into idempotency.
            raise ValueError("bundle insert conflict without immutable row")
        if row.bundle_hash != bundle["bundle_hash"] or dict(row.payload) != bundle:
            raise ValueError("immutable bundle id collision")
        return dict(row.payload)

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
            predicates = [
                ClaimArtifactMemberRow.artifact_id == claim_artifact_id,
                FinancialClaimRow.claim_schema_version == "claim.atomic.v1",
                FinancialClaimRow.grounding_status == "GROUNDED",
                FinancialClaimRow.legacy_grounding_incomplete.is_(False),
                ClaimOccurrenceRow.claim_schema_version == "claim.atomic.v1",
                ClaimOccurrenceRow.grounding_status == "GROUNDED",
                ClaimOccurrenceRow.legacy_grounding_incomplete.is_(False),
                ClaimOccurrenceRow.available_from <= request.availability_as_of,
                ClaimOccurrenceRow.snapshot_committed_at <= request.knowledge_as_of,
            ]
            # UNSPECIFIED is an explicit v2 *scope* selector.  It is never
            # written back as a common subject for a multi-topic video.
            if request.symbol.upper() != "UNSPECIFIED":
                predicates.append(FinancialClaimRow.subject_id == request.symbol)
            rows = session.execute(
                select(FinancialClaimRow, ClaimOccurrenceRow)
                .join(ClaimArtifactMemberRow, ClaimArtifactMemberRow.claim_id == FinancialClaimRow.claim_id)
                .join(ClaimOccurrenceRow, ClaimOccurrenceRow.claim_id == FinancialClaimRow.claim_id)
                .where(*predicates).order_by(
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
                if request.contract_version == V2_CONTRACT:
                    semantic = _v2_semantics(claim, occurrence)
                    artifact_ids = {
                        str(evidence_map[link.evidence_id].get("source_artifact_id") or "")
                        for link in links
                        if link.evidence_id in evidence_map
                    }
                    artifact_rows = {
                        row.artifact_id: row
                        for row in session.scalars(
                            select(ContentArtifactRow).where(ContentArtifactRow.artifact_id.in_(artifact_ids - {""}))
                        )
                    }
                    frame_artifact_ids = {
                        str((row.payload or {}).get("frame_artifact_id") or "")
                        for row in artifact_rows.values()
                    } - {""}
                    if frame_artifact_ids:
                        artifact_rows.update({
                            row.artifact_id: row
                            for row in session.scalars(
                                select(ContentArtifactRow).where(ContentArtifactRow.artifact_id.in_(frame_artifact_ids))
                            )
                        })
                    direct_evidence = _v2_evidence(links, evidence_map, artifact_rows)
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
                            "primary_domain": semantic["primary_domain"],
                            "claim_nature": semantic["claim_nature"],
                            "attribution": semantic["attribution"],
                            "source_grade": semantic["source_grade"],
                            "detail": semantic["detail"],
                            "temporal": semantic["temporal"],
                            "occurrence_review": semantic["occurrence_review"],
                            "support_status": support_status,
                            "lifecycle_status": lifecycle["status"],
                            "confidence": claim.source_confidence,
                            "evidence": direct_evidence,
                            "verification": {
                                "status": (
                                    semantic["external_truth_status"]
                                    if semantic["attribution"].get("attributed")
                                    else verification_status
                                ),
                                "reason_codes": list(projection.get("verification_reason_codes") or []),
                            },
                            "external_truth_status": semantic["external_truth_status"],
                            "contradiction_group_id": claim.contradiction_group_id,
                            "known_from": lifecycle.get("known_from"),
                            "available_from": projection.get("available_from"),
                            "claim_schema_version": claim.claim_schema_version,
                            "grounding_status": claim.grounding_status,
                            "legacy_grounding_incomplete": claim.legacy_grounding_incomplete,
                        }
                    )
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


def _v2_semantics(claim: FinancialClaimRow, occurrence: ClaimOccurrenceRow) -> dict[str, Any]:
    """Read explicit occurrence semantics from immutable JSON authority.

    The JSON columns are intentionally used for this additive projection: they
    are already snapshot-bound, transactional and available to replayed rows.
    Missing detail is not replaced with a vague sentence; the v2 validator
    fails closed until the producer supplied an auditable structured detail.
    """
    payload = dict(claim.payload or {})
    occurrence_payload = dict(occurrence.provenance or {})
    semantic = dict(payload.get("bundle_v2") or {})
    semantic.update(dict(occurrence_payload.get("bundle_v2") or {}))
    nature = str(semantic.get("claim_nature") or claim.fact_category or "OPINION")
    grounded_reasons = [str(value) for value in (claim.grounding_reason_codes or [])]
    review = dict(semantic.get("occurrence_review") or {})
    if not review:
        conflict_reasons = [value for value in grounded_reasons if "CONFLICT" in value or "REVIEW" in value]
        review = {
            "status": "HUMAN_REVIEW_REQUIRED" if conflict_reasons else "NOT_REQUIRED",
            "reason_codes": conflict_reasons,
        }
    return {
        "primary_domain": str(semantic.get("primary_domain") or "UNKNOWN"),
        "claim_nature": nature,
        "attribution": dict(semantic.get("attribution") or {
            "attributed": nature in {"OPINION", "FORECAST", "CAUSAL_THESIS"},
            "source_label": "source_material" if nature in {"OPINION", "FORECAST", "CAUSAL_THESIS"} else None,
        }),
        "source_grade": str(semantic.get("source_grade") or "UNKNOWN"),
        "detail": dict(semantic.get("detail") or {}),
        "temporal": dict(semantic.get("temporal") or {
            "kind": "UNKNOWN", "start": None, "end": None, "as_of": None,
            "rule": None, "label": None, "precision": "UNKNOWN", "explicitly_unknown": True,
        }),
        "occurrence_review": review,
        "external_truth_status": str(semantic.get("external_truth_status") or "NOT_CHECKED"),
    }


def _v2_evidence(
    links, evidence_map: dict[str, dict[str, Any]], artifact_rows: dict[str, ContentArtifactRow]
) -> list[dict[str, Any]]:
    """Project direct immutable transcript/frame/OCR/vision citations.

    Every evidence record keeps the exact content-addressed artifact hash. OCR
    and vision records also retain their frame locator and model identity.  A
    linked visual record emits its source Frame artifact as a distinct evidence
    member, so consumers do not have to infer cross-modal support from text.
    """
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for link in links:
        value = evidence_map.get(link.evidence_id)
        if value is None:
            continue
        artifact_id = str(value.get("source_artifact_id") or "")
        artifact = artifact_rows.get(artifact_id)
        if artifact is None:
            raise ValueError("BUNDLE_EVIDENCE_ARTIFACT_MISSING")
        modality = _v2_modality(value.get("source_type"))
        locator_data = dict(value.get("locator") or {})
        start_ms = value.get("start_ms")
        end_ms = value.get("end_ms")
        locator = {
            "segment_id": locator_data.get("segment_id") or locator_data.get("segment_index"),
            "frame_id": locator_data.get("frame_id"),
            "start_ms": int(start_ms if start_ms is not None else locator_data.get("timestamp_ms") or 0),
            "end_ms": int(end_ms if end_ms is not None else locator_data.get("timestamp_ms") or 0),
            "bbox": locator_data.get("bbox"),
        }
        content = str(value.get("normalized_text") or value.get("evidence_text") or "")
        if not content.strip():
            raise ValueError("BUNDLE_EVIDENCE_QUOTE_MISSING")
        entry: dict[str, Any] = {
            "evidence_id": str(link.evidence_id),
            "ownership": link.evidence_role,
            "modality": modality,
            "artifact_id": artifact.artifact_id,
            "artifact_hash": "sha256:" + str(artifact.content_hash),
            "locator": locator,
            "content": content,
        }
        if modality in {"ocr", "vision"}:
            entry["model"] = _v2_model_identity(artifact, modality)
        result.append(entry)
        seen.add(entry["evidence_id"])
        frame_artifact_id = str((artifact.payload or {}).get("frame_artifact_id") or "")
        frame = artifact_rows.get(frame_artifact_id)
        if modality in {"ocr", "vision"} and frame is not None:
            frame_id = str(locator.get("frame_id") or (frame.payload or {}).get("frame_id") or "")
            frame_entry_id = f"{link.evidence_id}:frame"
            if frame_entry_id not in seen and frame_id:
                result.append({
                    "evidence_id": frame_entry_id,
                    "ownership": link.evidence_role,
                    "modality": "frame",
                    "artifact_id": frame.artifact_id,
                    "artifact_hash": "sha256:" + str(frame.content_hash),
                    "locator": {**locator, "frame_id": frame_id},
                    "content": f"frame:{frame_id}",
                })
                seen.add(frame_entry_id)
    return result


def _v2_modality(value: Any) -> str:
    source = str(value or "").strip().lower()
    if source in {"ocr", "vision", "frame"}:
        return source
    return "transcript"


def _v2_model_identity(artifact: ContentArtifactRow, modality: str) -> dict[str, Any]:
    payload = dict(artifact.payload or {})
    if modality == "ocr":
        name, version = payload.get("engine"), payload.get("engine_version")
    else:
        name, version = payload.get("model_name") or payload.get("model"), payload.get("model_version")
    if not str(name or "").strip() or not str(version or "").strip():
        raise ValueError("BUNDLE_EVIDENCE_MODEL_IDENTITY_MISSING")
    confidence = payload.get("confidence_score")
    return {"name": str(name), "version": str(version), "confidence": confidence}

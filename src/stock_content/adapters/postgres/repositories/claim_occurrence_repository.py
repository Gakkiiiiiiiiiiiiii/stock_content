"""Immutable persistence for claim occurrences and role relationships."""

from __future__ import annotations

from datetime import UTC

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ClaimOccurrenceEvidenceRow, ClaimOccurrenceRow
from stock_content.domain.artifacts import canonical_json
from stock_content.domain.claim_occurrence import (
    ClaimOccurrence,
    assertion_locator_hash_of,
    occurrence_id_of,
)


class ClaimOccurrenceRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def save(self, occurrence: ClaimOccurrence) -> ClaimOccurrence:
        with self._sessions.begin() as session:
            return self.save_in_session(session, occurrence)

    def save_in_session(self, session, occurrence: ClaimOccurrence) -> ClaimOccurrence:
        """Persist into a caller-owned transaction.

        Snapshot publication uses this hook to make the immutable occurrence,
        lifecycle ledger and snapshot one SQL commit boundary.
        """
        _validate_occurrence_identity(occurrence)
        payload = occurrence.model_dump(mode="python")
        time_payload = occurrence.times.model_dump(mode="python")
        values = {
            key: (time_payload[key] if key in time_payload else payload[key])
            for key in (
                "occurrence_id", "claim_id", "source_artifact_id", "transcript_artifact_id",
                "semantic_segment_id", "assertion_locator_hash", "asserted_at", "source_published_at",
                "source_available_at", "source_availability_quality", "ingested_at",
                "extraction_completed_at", "snapshot_committed_at", "available_from",
                "source_support_status", "source_confidence", "extractor_confidence",
                "raw_temporal_expressions", "provenance",
            )
        }
        # Claim the immutable occurrence row atomically.  A loser must read
        # the committed winner and return it without writing roles or metadata.
        inserted = _insert_ignore(session, ClaimOccurrenceRow, values)
        if not inserted:
            row = session.get(ClaimOccurrenceRow, occurrence.occurrence_id)
            if row is None:
                raise RuntimeError("occurrence disappeared after a unique-key conflict")
            existing = _existing_payload(session, row)
            _compare_occurrence_identity(existing, payload, occurrence.occurrence_id)
            return ClaimOccurrence.model_validate(existing)
        roles = {
            "PRIMARY": occurrence.evidence_refs,
            "CONDITION": occurrence.condition_evidence_refs,
            "INVALIDATION": occurrence.invalidation_evidence_refs,
            "TEMPORAL": occurrence.temporal_evidence_refs,
        }
        for role, refs in roles.items():
            for ordinal, evidence_id in enumerate(sorted(set(refs))):
                _insert_ignore(
                    session,
                    ClaimOccurrenceEvidenceRow,
                    {
                        "occurrence_id": occurrence.occurrence_id,
                        "evidence_id": evidence_id,
                        "evidence_role": role,
                        "ordinal": ordinal,
                    },
                )
        return occurrence

    def validate_immutable(self, occurrence: ClaimOccurrence) -> None:
        """Check an existing row without opening a write transaction."""
        existing = self.get(occurrence.occurrence_id)
        if existing is not None:
            _validate_occurrence_identity(occurrence)
            _compare_occurrence_identity(
                existing.model_dump(mode="python"),
                occurrence.model_dump(mode="python"),
                occurrence.occurrence_id,
            )

    insert = save

    def get(self, occurrence_id: str) -> ClaimOccurrence | None:
        with self._sessions() as session:
            row = session.get(ClaimOccurrenceRow, occurrence_id)
            if row is None:
                return None
            payload = _row_payload(row)
            role_rows = session.scalars(
                select(ClaimOccurrenceEvidenceRow)
                .where(ClaimOccurrenceEvidenceRow.occurrence_id == occurrence_id)
                .order_by(ClaimOccurrenceEvidenceRow.evidence_role, ClaimOccurrenceEvidenceRow.ordinal,
                          ClaimOccurrenceEvidenceRow.evidence_id)
            ).all()
        refs = {"PRIMARY": [], "CONDITION": [], "INVALIDATION": [], "TEMPORAL": []}
        for item in role_rows:
            refs.setdefault(item.evidence_role, []).append(item.evidence_id)
        payload["evidence_refs"] = sorted({*payload.get("evidence_refs", []), *refs["PRIMARY"]})
        payload["condition_evidence_refs"] = refs["CONDITION"]
        payload["invalidation_evidence_refs"] = refs["INVALIDATION"]
        payload["temporal_evidence_refs"] = refs["TEMPORAL"]
        return ClaimOccurrence.model_validate(payload)

    def list_for_claim(self, claim_id: str) -> list[ClaimOccurrence]:
        with self._sessions() as session:
            ids = list(session.scalars(
                select(ClaimOccurrenceRow.occurrence_id)
                .where(ClaimOccurrenceRow.claim_id == claim_id)
                .order_by(ClaimOccurrenceRow.occurrence_id)
            ).all())
        return [item for item in (self.get(identifier) for identifier in ids) if item is not None]

    def list_by_evidence(self, evidence_id: str) -> list[ClaimOccurrence]:
        """Return occurrences owning an evidence item through role membership.

        This is the authoritative reverse lookup for final claims.  It never
        consults the legacy claim_evidence compatibility table.
        """
        with self._sessions() as session:
            ids = list(session.scalars(
                select(ClaimOccurrenceEvidenceRow.occurrence_id)
                .where(ClaimOccurrenceEvidenceRow.evidence_id == evidence_id)
                .order_by(ClaimOccurrenceEvidenceRow.occurrence_id)
                .distinct()
            ).all())
        return [item for item in (self.get(identifier) for identifier in ids) if item is not None]


def _row_payload(row: ClaimOccurrenceRow) -> dict:
    return {
        "occurrence_id": row.occurrence_id,
        "claim_id": row.claim_id,
        "source_artifact_id": row.source_artifact_id,
        "transcript_artifact_id": row.transcript_artifact_id,
        "semantic_segment_id": row.semantic_segment_id,
        "assertion_locator_hash": row.assertion_locator_hash,
        "evidence_refs": [],
        "condition_evidence_refs": [],
        "invalidation_evidence_refs": [],
        "temporal_evidence_refs": [],
        "times": {
            "asserted_at": _utc_time(row.asserted_at),
            "source_published_at": _utc_time(row.source_published_at),
            "source_available_at": _utc_time(row.source_available_at),
            "source_availability_quality": row.source_availability_quality,
            "ingested_at": _utc_time(row.ingested_at),
            "extraction_completed_at": _utc_time(row.extraction_completed_at),
            "snapshot_committed_at": _utc_time(row.snapshot_committed_at),
            "available_from": _utc_time(row.available_from),
        },
        "source_support_status": row.source_support_status,
        "source_confidence": row.source_confidence,
        "extractor_confidence": row.extractor_confidence,
        "raw_temporal_expressions": list(row.raw_temporal_expressions or []),
        "provenance": dict(row.provenance or {}),
    }


def _normalize_times(value):
    if hasattr(value, "value"):
        return _normalize_times(value.value)
    if isinstance(value, dict):
        return {key: _normalize_times(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_times(item) for item in value]
    if hasattr(value, "isoformat"):
        if getattr(value, "tzinfo", None) is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return value


def _utc_time(value):
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _validate_occurrence_identity(occurrence: ClaimOccurrence) -> None:
    """Validate the supplied identity against its authoritative coordinates."""
    expected_locator = assertion_locator_hash_of(
        occurrence.source_artifact_id,
        occurrence.transcript_artifact_id,
        occurrence.semantic_segment_id,
        occurrence.evidence_refs
        + occurrence.condition_evidence_refs
        + occurrence.invalidation_evidence_refs,
        occurrence.temporal_evidence_refs,
    )
    if occurrence.assertion_locator_hash != expected_locator:
        raise ValueError(
            f"occurrence {occurrence.occurrence_id} assertion locator does not match evidence coordinates"
        )
    expected_id = occurrence_id_of(
        occurrence.claim_id,
        occurrence.source_artifact_id,
        expected_locator,
    )
    if occurrence.occurrence_id != expected_id:
        raise ValueError(f"occurrence id {occurrence.occurrence_id} does not match immutable identity")


def _occurrence_identity(value: dict) -> dict:
    """Extract only identity/coordinate fields; exclude first-write metadata."""
    evidence_coordinate_refs = sorted(
        {
            ref
            for field in (
                "evidence_refs",
                "condition_evidence_refs",
                "invalidation_evidence_refs",
                "temporal_evidence_refs",
            )
            for ref in (value.get(field) or [])
        }
    )
    return {
        "claim_id": value.get("claim_id"),
        "source_artifact_id": value.get("source_artifact_id"),
        "transcript_artifact_id": value.get("transcript_artifact_id"),
        "semantic_segment_id": value.get("semantic_segment_id"),
        "assertion_locator_hash": value.get("assertion_locator_hash"),
        # Role assignment is a first-write projection, not occurrence
        # identity.  The locator contract uses the union of all coordinates.
        "evidence_coordinate_refs": evidence_coordinate_refs,
    }


def _compare_occurrence_identity(existing: dict, candidate: dict, occurrence_id: str) -> None:
    if canonical_json(_occurrence_identity(existing)) != canonical_json(_occurrence_identity(candidate)):
        raise ValueError(f"occurrence id {occurrence_id} already stores a different identity")


def _existing_payload(session, row: ClaimOccurrenceRow) -> dict:
    payload = _row_payload(row)
    role_rows = session.scalars(
        select(ClaimOccurrenceEvidenceRow).where(
            ClaimOccurrenceEvidenceRow.occurrence_id == row.occurrence_id
        )
    ).all()
    payload["evidence_refs"] = sorted(
        item.evidence_id for item in role_rows if item.evidence_role == "PRIMARY"
    )
    payload["condition_evidence_refs"] = sorted(
        item.evidence_id for item in role_rows if item.evidence_role == "CONDITION"
    )
    payload["invalidation_evidence_refs"] = sorted(
        item.evidence_id for item in role_rows if item.evidence_role == "INVALIDATION"
    )
    payload["temporal_evidence_refs"] = sorted(
        item.evidence_id for item in role_rows if item.evidence_role == "TEMPORAL"
    )
    return payload


def _insert_ignore(session, model, values: dict) -> bool:
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        result = session.execute(postgres_insert(model).values(**values).on_conflict_do_nothing())
        return result.rowcount == 1
    elif dialect == "sqlite":
        result = session.execute(sqlite_insert(model).values(**values).on_conflict_do_nothing())
        return result.rowcount == 1
    else:
        try:
            with session.begin_nested():
                session.add(model(**values))
                session.flush()
            return True
        except IntegrityError:
            return False


__all__ = ["ClaimOccurrenceRepository"]

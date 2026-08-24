"""Canonical FinancialClaim persistence and evidence reverse lookup."""
from __future__ import annotations

import json
from collections.abc import Iterable

from sqlalchemy import inspect, select, text
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.legacy_ids import (
    legacy_evidence_member_id,
    legacy_verification_id,
)
from stock_content.adapters.postgres.models import (
    ClaimEvidenceRow,
    ClaimVerificationJobRow,
    ClaimVerificationResultRow,
    FinancialClaimRow,
)
from stock_content.domain.claims import FinancialClaim


class SqlClaimRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def save(self, claim: FinancialClaim) -> FinancialClaim:
        payload = claim.model_dump(mode="json")

        with self._sessions.begin() as session:
            row = session.get(FinancialClaimRow, claim.claim_id)
            if row is not None:
                existing_claim = FinancialClaim.model_validate(_payload_from_row(session, row))
                if existing_claim.content_payload() != claim.content_payload():
                    raise ValueError(f"claim id {claim.claim_id} already stores a different payload")
                if not isinstance(row.payload, dict) or not row.payload.get("claim_type"):
                    row.payload = payload
            else:
                values = {
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "fact_category": claim.fact_category,
                    "subject_type": claim.subject_type,
                    "subject_id": claim.subject_id,
                    "predicate": claim.predicate,
                    "value": claim.value,
                    "unit": claim.unit,
                    "currency": claim.currency,
                    "fact_time": claim.fact_time,
                    "period_start": claim.period_start,
                    "period_end": claim.period_end,
                    "published_at": claim.published_at,
                    "source_confidence": claim.source_confidence,
                    "extractor_confidence": claim.extractor_confidence,
                    "extraction_model_id": claim.extraction_model_id,
                    "extraction_prompt_version": claim.extraction_prompt_version,
                    "condition_text": claim.condition_text,
                    "invalidation_text": claim.invalidation_text,
                    "claim_schema_version": claim.claim_schema_version,
                    "normalization_version": claim.normalization_version,
                    "source_support_status": claim.source_support_status,
                    "payload": payload,
                }
                _insert_ignore(session, FinancialClaimRow, values, [FinancialClaimRow.claim_id])
                row = session.get(FinancialClaimRow, claim.claim_id)
                if row is None:
                    raise RuntimeError("claim disappeared after a unique-key conflict")
                existing_claim = FinancialClaim.model_validate(_payload_from_row(session, row))
                if existing_claim.content_payload() != claim.content_payload():
                    raise ValueError(f"claim id {claim.claim_id} already stores a different payload")
            for evidence_id in claim.evidence_refs:
                member_id = legacy_evidence_member_id(claim.claim_id, evidence_id)
                _insert_ignore(
                    session,
                    ClaimEvidenceRow,
                    {"member_id": member_id, "claim_id": claim.claim_id, "evidence_id": evidence_id},
                    [ClaimEvidenceRow.claim_id, ClaimEvidenceRow.evidence_id],
                )
        return claim

    def get(self, claim_id: str) -> FinancialClaim | None:
        with self._sessions() as session:
            row = session.get(FinancialClaimRow, claim_id)
            if row is None:
                return None
            payload = _payload_from_row(session, row)
            refs = _membership_evidence_refs(session, claim_id)
        payload["evidence_refs"] = refs or list(payload.get("evidence_refs") or [])
        return FinancialClaim.model_validate(payload)

    def evidence(self, claim_id: str) -> list[str]:
        with self._sessions() as session:
            return _membership_evidence_refs(session, claim_id)

    def verifications(self, claim_id: str) -> list[dict]:
        """Return immutable verification results in deterministic order."""
        with self._sessions() as session:
            rows = session.scalars(
                select(ClaimVerificationResultRow)
                .where(ClaimVerificationResultRow.claim_id == claim_id)
                .order_by(ClaimVerificationResultRow.created_at, ClaimVerificationResultRow.verification_id)
            ).all()
            jobs = session.scalars(
                select(ClaimVerificationJobRow)
                .where(ClaimVerificationJobRow.claim_id == claim_id)
                .order_by(ClaimVerificationJobRow.created_at, ClaimVerificationJobRow.job_id)
            ).all()
            legacy_rows = _legacy_lifecycle_rows(session, claim_id)
            result_ids = {row.verification_id for row in rows}
        items = [
            {
                "verification_id": row.verification_id,
                "claim_id": row.claim_id,
                "provider": row.provider,
                "status": row.status,
                "market_snapshot_id": row.market_snapshot_id,
                "market_data_version": row.market_data_version,
                "result": dict(row.result_payload or {}),
                "trace_id": row.trace_id,
                "verified_at": row.verified_at.isoformat() if row.verified_at else None,
            }
            for row in rows
        ]
        for legacy in legacy_rows:
            verification_id = legacy_verification_id(legacy["claim_id"])
            if verification_id in result_ids:
                continue
            result = _json_value(legacy.get("result")) or {}
            if not isinstance(result, dict):
                result = {"legacy_result": result}
            items.append(
                {
                    "verification_id": verification_id,
                    "claim_id": legacy["claim_id"],
                    "provider": "legacy_lifecycle",
                    "status": legacy["status"],
                    "market_snapshot_id": legacy.get("market_snapshot_id"),
                    "market_data_version": legacy.get("market_data_version"),
                    "result": result,
                    "trace_id": None,
                    "verified_at": legacy.get("verification_timestamp").isoformat()
                    if legacy.get("verification_timestamp")
                    else None,
                }
            )
        items.extend([
            {
                "job_id": row.job_id,
                "claim_id": row.claim_id,
                "provider": row.provider,
                "status": row.status,
                "retry_count": row.retry_count,
                "next_retry_at": row.next_retry_at.isoformat() if row.next_retry_at else None,
                "trace_id": row.trace_id,
            }
            for row in jobs
        ])
        return items

    def claims_for_evidence(self, evidence_id: str) -> list[FinancialClaim]:
        with self._sessions() as session:
            ids = list(
                session.scalars(
                    select(ClaimEvidenceRow.claim_id).where(ClaimEvidenceRow.evidence_id == evidence_id)
                ).all()
            )
        return [claim for claim_id in ids if (claim := self.get(claim_id)) is not None]

    def enqueue_verification_jobs(self, claims: Iterable[FinancialClaim], trace_id: str | None = None) -> None:
        from stock_content.adapters.postgres.repositories.verification_job_repository import (
            PostgresVerificationJobRepository,
        )

        PostgresVerificationJobRepository(self._sessions).enqueue(
            claims, provider="quant", trace_id=trace_id
        )


ClaimRepository = SqlClaimRepository
PostgresClaimRepository = SqlClaimRepository

__all__ = ["ClaimRepository", "PostgresClaimRepository", "SqlClaimRepository"]


def _insert_ignore(session, model, values: dict, conflict_columns: list) -> bool:
    """Insert once while preserving the surrounding transaction on a race."""
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        statement = postgres_insert(model).values(**values)
    elif dialect == "sqlite":
        statement = sqlite_insert(model).values(**values)
    else:
        try:
            with session.begin_nested():
                session.add(model(**values))
                session.flush()
            return True
        except IntegrityError:
            return False
    result = session.execute(statement.on_conflict_do_nothing(index_elements=conflict_columns))
    return result.rowcount == 1


def _json_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _legacy_evidence_refs(session, claim_id: str) -> list[str]:
    """Read the denormalized evidence column retained by migration 013."""
    bind = session.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("financial_claim"):
        return []
    columns = {column["name"] for column in inspector.get_columns("financial_claim")}
    if "evidence_refs" not in columns:
        return []
    value = session.execute(
        text("SELECT evidence_refs FROM financial_claim WHERE claim_id = :claim_id"),
        {"claim_id": claim_id},
    ).scalar_one_or_none()
    value = _json_value(value) or []
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def _membership_evidence_refs(session, claim_id: str) -> list[str]:
    """Prefer normalized memberships, with a direct 013 fallback."""
    if inspect(session.get_bind()).has_table("claim_evidence"):
        refs = list(
            session.scalars(
                select(ClaimEvidenceRow.evidence_id).where(ClaimEvidenceRow.claim_id == claim_id)
                .order_by(ClaimEvidenceRow.evidence_id)
            ).all()
        )
        if refs:
            return refs
    return _legacy_evidence_refs(session, claim_id)


def _payload_from_row(session, row: FinancialClaimRow) -> dict:
    """Construct a complete domain payload for both 013 and 015 rows."""
    payload = _json_value(row.payload)
    if not isinstance(payload, dict):
        payload = {}
    values = {
        "claim_id": row.claim_id,
        "claim_type": row.claim_type,
        "fact_category": row.fact_category or "FACT",
        "subject_type": row.subject_type,
        "subject_id": row.subject_id,
        "predicate": row.predicate,
        "value": _json_value(row.value),
        "unit": row.unit,
        "currency": row.currency,
        "fact_time": row.fact_time,
        "period_start": row.period_start,
        "period_end": row.period_end,
        "published_at": row.published_at,
        "source_support_status": row.source_support_status or "UNSUPPORTED",
        "source_confidence": row.source_confidence,
        "extractor_confidence": row.extractor_confidence,
        "extraction_model_id": row.extraction_model_id or "unknown",
        "extraction_prompt_version": row.extraction_prompt_version or "unknown",
        "condition_text": row.condition_text,
        "invalidation_text": row.invalidation_text,
        "claim_schema_version": row.claim_schema_version or "claim.v2",
        "normalization_version": row.normalization_version or "normalization.v1",
    }
    for key, value in values.items():
        payload.setdefault(key, value)
    payload.setdefault("evidence_refs", _legacy_evidence_refs(session, row.claim_id))
    return payload


def _legacy_lifecycle_rows(session, claim_id: str) -> list[dict]:
    """Compatibility view when migration 015 has not copied lifecycle rows."""
    bind = session.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("claim_verification_lifecycle"):
        return []
    rows = session.execute(
        text("SELECT * FROM claim_verification_lifecycle WHERE claim_id = :claim_id"),
        {"claim_id": claim_id},
    ).mappings()
    return list(rows)

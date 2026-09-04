"""Verification closure and monotonic source-head helpers."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import and_, case, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from stock_content.adapters.postgres.models import (
    ClaimVerificationJobRow,
    ClaimVerificationResultRow,
    ContentArtifactRow,
    ContentSourceHeadRow,
)
from stock_content.domain.artifacts import deserialize_artifact
from stock_content.domain.claims import VerificationArtifactEntry
from stock_content.domain.initial_verification import verify_initial_verification_closure


def _validate_refresh_verification_closure(session, snapshot) -> None:
    """Validate every exact job/result reference in a refresh artifact."""
    verification_id = str((snapshot.artifact_ids or {}).get("verification") or "")
    if not verification_id:
        return
    row = session.get(ContentArtifactRow, verification_id)
    if row is None:
        return
    artifact = deserialize_artifact(dict(row.payload or {}))
    entries = [
        item for item in (getattr(artifact, "results", ()) or ())
        if isinstance(item, VerificationArtifactEntry)
    ]
    if not entries:
        return
    job_ids = {str(item.verification_job_id) for item in entries if item.verification_job_id}
    result_ids = {str(item.verification_id) for item in entries if item.verification_id}
    jobs = {
        item.job_id: item
        for item in session.scalars(
            select(ClaimVerificationJobRow).where(ClaimVerificationJobRow.job_id.in_(job_ids))
        ).all()
    } if job_ids else {}
    results = {
        item.verification_id: item
        for item in session.scalars(
            select(ClaimVerificationResultRow).where(
                ClaimVerificationResultRow.verification_id.in_(result_ids)
            )
        ).all()
    } if result_ids else {}
    verify_initial_verification_closure(
        artifact_results=entries, jobs=jobs, results=results,
        snapshot_committed_at=snapshot.created_at,
    )


def _upsert_source_head(
    session,
    *,
    source_identity_hash: str,
    snapshot_id: str,
    verified_snapshot_id: str | None,
    updated_at: datetime,
) -> None:
    """Create/advance a source head without a first-writer race."""
    values = {
        "source_identity_hash": source_identity_hash,
        "latest_snapshot_id": snapshot_id,
        "latest_verified_snapshot_id": verified_snapshot_id,
        "updated_at": updated_at,
    }
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        statement = postgres_insert(ContentSourceHeadRow).values(**values)
    elif dialect == "sqlite":
        statement = sqlite_insert(ContentSourceHeadRow).values(**values)
    else:
        try:
            with session.begin_nested():
                session.add(ContentSourceHeadRow(**values))
                session.flush()
            return
        except IntegrityError:
            head = session.get(ContentSourceHeadRow, source_identity_hash)
            if head is None:
                raise RuntimeError("source head disappeared after a unique-key conflict")
            if _head_is_newer(updated_at, snapshot_id, head.updated_at, head.latest_snapshot_id):
                head.latest_snapshot_id = snapshot_id
                head.updated_at = updated_at
                if verified_snapshot_id is not None:
                    head.latest_verified_snapshot_id = verified_snapshot_id
            return

    excluded = statement.excluded
    is_newer = or_(
        excluded.updated_at > ContentSourceHeadRow.updated_at,
        and_(
            excluded.updated_at == ContentSourceHeadRow.updated_at,
            excluded.latest_snapshot_id > ContentSourceHeadRow.latest_snapshot_id,
        ),
    )


    verified_is_newer = and_(
        excluded.latest_verified_snapshot_id.is_not(None),
        or_(ContentSourceHeadRow.latest_verified_snapshot_id.is_(None), is_newer),
    )
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[ContentSourceHeadRow.source_identity_hash],
            set_={
                "latest_snapshot_id": case(
                    (is_newer, excluded.latest_snapshot_id),
                    else_=ContentSourceHeadRow.latest_snapshot_id,
                ),
                "updated_at": case(
                    (is_newer, excluded.updated_at),
                    else_=ContentSourceHeadRow.updated_at,
                ),
                "latest_verified_snapshot_id": case(
                    (verified_is_newer, excluded.latest_verified_snapshot_id),
                    else_=ContentSourceHeadRow.latest_verified_snapshot_id,
                ),
            },
        )
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _head_is_newer(
    updated_at: datetime,
    snapshot_id: str,
    existing_updated_at: datetime,
    existing_snapshot_id: str,
) -> bool:
    def normalized(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    return (normalized(updated_at), snapshot_id) > (
        normalized(existing_updated_at),
        existing_snapshot_id,
    )



__all__ = ["_as_utc", "_head_is_newer", "_upsert_source_head", "_validate_refresh_verification_closure"]

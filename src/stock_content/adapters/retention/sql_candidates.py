"""Authoritative, private retention-candidate enumeration."""

from __future__ import annotations

from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ContentArtifactRow,
    RetentionArtifactLocatorRow,
    SourceArtifactMetadataRow,
)
from stock_content.domain.retention import RetentionCandidate, RetentionClass


class SqlRetentionCandidateRepository:
    """Read retention work only from controlled SQL metadata.

    Relative locators never leave this adapter.  The scheduler receives a
    candidate plus an artifact-id deleter whose map is built from these rows.
    """

    def __init__(self, session_factory: sessionmaker, *, private_root_id: str) -> None:
        if not private_root_id:
            raise ValueError("RETENTION_PRIVATE_ROOT_ID_REQUIRED")
        self._sessions = session_factory
        self._root_id = private_root_id

    def candidates(self) -> tuple[RetentionCandidate, ...]:
        with self._sessions() as session:
            rows = session.execute(
                select(ContentArtifactRow, SourceArtifactMetadataRow, RetentionArtifactLocatorRow)
                .join(
                    SourceArtifactMetadataRow, SourceArtifactMetadataRow.artifact_id == ContentArtifactRow.artifact_id
                )
                .join(
                    RetentionArtifactLocatorRow,
                    RetentionArtifactLocatorRow.artifact_id == ContentArtifactRow.artifact_id,
                )
                .where(RetentionArtifactLocatorRow.private_root_id == self._root_id)
                .order_by(ContentArtifactRow.created_at, ContentArtifactRow.artifact_id)
            ).all()
        result = []
        for artifact, metadata, locator in rows:
            try:
                artifact_class = RetentionClass(str(metadata.retention_class))
            except ValueError:
                continue
            source_identity_hash = str(metadata.source_identity_hash or "")
            is_sha256 = len(source_identity_hash) == 64 and all(
                char in "0123456789abcdef" for char in source_identity_hash.lower()
            )
            if not is_sha256:
                # Historical rows without a true source identity must not be
                # relabelled from content bytes or scheduled for deletion.
                continue
            created = artifact.created_at if artifact.created_at.tzinfo else artifact.created_at.replace(tzinfo=UTC)
            result.append(
                RetentionCandidate(
                    artifact_id=artifact.artifact_id,
                    artifact_class=artifact_class,
                    content_hash=artifact.content_hash,
                    source_identity_hash=source_identity_hash,
                    audit_lineage_id=f"artifact:{artifact.artifact_id}",
                    created_at=created,
                    legal_hold=bool(locator.legal_hold),
                )
            )
        return tuple(result)

    def private_locator_map(self) -> dict[str, str]:
        with self._sessions() as session:
            rows = session.scalars(
                select(RetentionArtifactLocatorRow).where(RetentionArtifactLocatorRow.private_root_id == self._root_id)
            ).all()
        # Absolute locators and traversal are rejected again by the deleter;
        # do not normalize untrusted material into a safe-looking value here.
        return {row.artifact_id: row.relative_locator for row in rows}


__all__ = ["SqlRetentionCandidateRepository"]

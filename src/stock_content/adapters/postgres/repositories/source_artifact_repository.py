from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ArtifactTombstoneRow, SourceArtifactMetadataRow


class SourceArtifactRepository:
    """Persist source governance metadata and append-only tombstones."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def save_metadata(self, artifact_id: str, *, source_policy_version: str, retention_class: str,
                      access_classification: str, source_content_hash: str, content_size: int,
                      mime_type: str, encryption_key_id: str | None = None) -> None:
        if not all((source_policy_version, retention_class, access_classification, source_content_hash, mime_type)):
            raise ValueError("new source artifacts require policy, retention, access, hash and mime metadata")
        with self._sessions.begin() as session:
            row = session.get(SourceArtifactMetadataRow, artifact_id)
            if row is None:
                session.add(SourceArtifactMetadataRow(
                    artifact_id=artifact_id, source_policy_version=source_policy_version,
                    retention_class=retention_class, access_classification=access_classification,
                    source_content_hash=source_content_hash, content_size=content_size,
                    mime_type=mime_type, encryption_key_id=encryption_key_id,
                ))
            elif (row.source_content_hash, row.content_size, row.mime_type) != (
                source_content_hash, content_size, mime_type,
            ):
                raise ValueError("source artifact metadata is immutable")

    def tombstone(self, artifact_id: str, *, reason: str, actor: str, policy_version: str,
                  request_id: str, deleted_at: datetime | None = None) -> None:
        with self._sessions.begin() as session:
            if session.get(ArtifactTombstoneRow, artifact_id) is None:
                session.add(ArtifactTombstoneRow(
                    artifact_id=artifact_id, reason=reason, actor=actor,
                    policy_version=policy_version, request_id=request_id,
                    deleted_at=deleted_at or datetime.now(UTC),
                ))

    def is_tombstoned(self, artifact_id: str) -> bool:
        with self._sessions() as session:
            return session.get(ArtifactTombstoneRow, artifact_id) is not None


__all__ = ["SourceArtifactRepository"]

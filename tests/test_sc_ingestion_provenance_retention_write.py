from __future__ import annotations

from datetime import UTC, datetime

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import RetentionArtifactLocatorRow, SourceArtifactMetadataRow
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.retention.sql_candidates import SqlRetentionCandidateRepository
from stock_content.api.dependencies import build_application
from stock_content.domain.artifacts import SourceArtifact, artifact_id_of


def _source(raw_path: str) -> SourceArtifact:
    source = SourceArtifact(
        artifact_id="source-pending",
        artifact_type="source",
        producer_stage="download",
        source_type="bilibili",
        source_ref="BV1fixture",
        source_content_hash="a" * 64,
        raw_content_hash="a" * 64,
        raw_content_length=3,
        raw_storage_uri=raw_path,
        source_identity_hash="b" * 64,
        source_version_id="source-version-fixture",
        source_metadata={
            "source_policy_version": "source-policy.v1",
            "access_classification": "PUBLIC",
            "mime_type": "video/mp4",
            "canonical_url": "https://www.bilibili.com/video/BV1fixture?secret=never-persist",
            "source_id": "BV1fixture",
            "source_part": "1",
            "author": "fixture-author",
            "published_at": "2026-09-01T00:00:00Z",
            "source_available_from": "2026-09-01T00:00:00Z",
            "business_as_of": "2026-09-01T00:00:00Z",
            "pipeline_version": "pipeline.v3",
        },
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    return SourceArtifact(**{**source.__dict__, "artifact_id": artifact_id_of(source)})


def test_production_artifact_write_normalizes_safe_provenance_and_registers_private_locator(tmp_path, monkeypatch):
    root = tmp_path / "private"
    raw = root / "raw" / "fixture.mp4"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    monkeypatch.setenv("CONTENT_RETENTION_PRIVATE_ROOT", str(root))
    monkeypatch.setenv("CONTENT_RETENTION_PRIVATE_ROOT_ID", "fixture-root")
    monkeypatch.setenv("CONTENT_SERVICE_VERSION", "fixture-service")
    database = Database(f"sqlite:///{tmp_path / 'content.db'}")
    database.create_schema()
    artifact = _source(str(raw))
    SqlArtifactRepository(database.session_factory).put(artifact)
    with database.session_factory() as session:
        metadata = session.get(SourceArtifactMetadataRow, artifact.artifact_id)
        locator = session.get(RetentionArtifactLocatorRow, artifact.artifact_id)
    assert metadata is not None
    assert metadata.canonical_url == "https://www.bilibili.com/video/BV1fixture"
    assert metadata.source_identity_hash == "b" * 64
    assert metadata.source_available_from.replace(tzinfo=UTC) == datetime(2026, 9, 1, tzinfo=UTC)
    assert "secret" not in repr(metadata)
    assert locator is not None
    assert locator.relative_locator == "raw/fixture.mp4"
    candidates = SqlRetentionCandidateRepository(database.session_factory, private_root_id="fixture-root").candidates()
    assert candidates[0].artifact_id == artifact.artifact_id
    assert candidates[0].source_identity_hash == "b" * 64


def test_external_source_ref_never_becomes_retention_locator(tmp_path, monkeypatch):
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setenv("CONTENT_RETENTION_PRIVATE_ROOT", str(root))
    monkeypatch.setenv("CONTENT_RETENTION_PRIVATE_ROOT_ID", "fixture-root")
    database = Database(f"sqlite:///{tmp_path / 'content.db'}")
    database.create_schema()
    source = _source("https://media.example.test/raw.mp4?signature=secret")
    SqlArtifactRepository(database.session_factory).put(source)
    with database.session_factory() as session:
        assert session.get(SourceArtifactMetadataRow, source.artifact_id) is not None
        assert session.get(RetentionArtifactLocatorRow, source.artifact_id) is None


def test_nonexistent_retention_root_is_never_reported_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTENT_RETENTION_ENABLED", "true")
    monkeypatch.setenv("CONTENT_RETENTION_PRIVATE_ROOT", str(tmp_path / "missing"))
    monkeypatch.setenv("CONTENT_RETENTION_PRIVATE_ROOT_ID", "fixture-root")
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    assert application._retention_scheduler is None  # noqa: SLF001 - composition boundary
    assert application._retention_status == "RETENTION_PRIVATE_ROOT_NOT_READY"  # noqa: SLF001

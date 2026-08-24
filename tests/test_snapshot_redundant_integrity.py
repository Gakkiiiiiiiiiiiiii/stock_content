from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from sqlalchemy import delete

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ContentArtifactEdgeRow,
    ContentArtifactRow,
    ContentSnapshotArtifactRow,
    ContentSnapshotRow,
)
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import (
    SnapshotIntegrityError,
    SqlSnapshotStore,
)
from stock_content.application.verification_refresh import VerificationRefreshService
from stock_content.domain.artifacts import ClaimArtifact, SourceArtifact
from stock_content.domain.lineage import build_content_snapshot


def _snapshot():
    return build_content_snapshot(
        source_type="fixture",
        source_ref="redundant-source",
        source_content_hash="content-hash",
        artifact_ids={"source": "source-artifact", "claims": "claims-artifact"},
        code_sha="release-sha",
        config_hash="config-hash",
        producer_manifest={"code_sha": "release-sha"},
    )


def _persist_artifacts(database):
    artifacts = SqlArtifactRepository(database.session_factory)
    artifacts.put(
        SourceArtifact(
            artifact_id="source-artifact",
            artifact_type="source",
            source_type="fixture",
            source_ref="redundant-source",
            source_content_hash="content-hash",
        )
    )
    artifacts.put(
        ClaimArtifact(
            artifact_id="claims-artifact",
            artifact_type="claims",
            parent_artifact_ids=("source-artifact",),
            evidence_artifact_id="source-artifact",
        )
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("source_content_hash", "tampered-content"),
        ("artifact_ids", {"source": "tampered-source"}),
        ("pipeline_version", "tampered-pipeline"),
        ("schema_version", "tampered-schema"),
        ("code_sha", "tampered-code"),
        ("config_hash", "tampered-config"),
        ("source_artifact_id", "tampered-source"),
        ("artifact_root_hash", "tampered-root"),
        ("producer_manifest", {"tampered": True}),
    ],
)
def test_redundant_snapshot_columns_fail_closed_for_save_get_and_list(tmp_path, column, value):
    database = Database(f"sqlite:///{tmp_path / 'snapshot-integrity.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        row = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
        assert row is not None
        setattr(row, column, value)

    with pytest.raises(SnapshotIntegrityError):
        store.get(snapshot.content_snapshot_id)
    with pytest.raises(SnapshotIntegrityError):
        store.list_for_source(snapshot.source_type, snapshot.source_ref)
    with pytest.raises(SnapshotIntegrityError):
        store.save(snapshot)


def test_direct_dataclass_provenance_conflict_is_rejected_before_insert(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'snapshot-provenance.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    forged = replace(
        snapshot,
        producer_manifest={
            "code_sha": "forged-code",
            "configs": {"config_hash": "forged-config"},
        },
    )

    with pytest.raises(SnapshotIntegrityError, match="producer_manifest"):
        store.save(forged)
    with database.session_factory() as session:
        assert session.get(ContentSnapshotRow, snapshot.content_snapshot_id) is None


def test_member_relation_tampering_fails_closed_for_save_get_and_list(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'snapshot-members.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        member = session.query(ContentSnapshotArtifactRow).filter_by(
            content_snapshot_id=snapshot.content_snapshot_id,
        ).first()
        assert member is not None
        member.member_id = hashlib.sha256(b"tampered-member").hexdigest()

    with pytest.raises(SnapshotIntegrityError):
        store.get(snapshot.content_snapshot_id)
    with pytest.raises(SnapshotIntegrityError):
        store.list_for_source(snapshot.source_type, snapshot.source_ref)
    with pytest.raises(SnapshotIntegrityError):
        store.save(snapshot)


def test_verification_refresh_persist_rejects_tampered_existing_snapshot(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'snapshot-refresh-integrity.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        row = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
        assert row is not None
        row.source_content_hash = "tampered-content"

    with database.session_factory.begin() as session:
        with pytest.raises(SnapshotIntegrityError):
            VerificationRefreshService._persist_snapshot(session, snapshot)


@pytest.mark.parametrize(
    "identity_key",
    [
        "source_content_hash",
        "artifact_ids",
        "pipeline_version",
        "schema_version",
        "source_artifact_id",
        "artifact_root_hash",
        "snapshot_kind",
        "parent_snapshot_id",
        "supersedes_snapshot_id",
        "producer_manifest",
        "model_versions",
        "prompt_versions",
        "configuration",
        "external_snapshots",
        "policy_versions",
    ],
)
def test_modern_snapshot_missing_identity_key_fails_closed(tmp_path, identity_key):
    database = Database(f"sqlite:///{tmp_path / 'snapshot-v2-missing.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        row = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
        assert row is not None
        identity = dict(row.identity)
        identity.pop(identity_key)
        row.identity = identity

    with pytest.raises(SnapshotIntegrityError):
        store.get(snapshot.content_snapshot_id)
    with pytest.raises(SnapshotIntegrityError):
        store.list_for_source(snapshot.source_type, snapshot.source_ref)
    with pytest.raises(SnapshotIntegrityError):
        store.save(snapshot)


@pytest.mark.parametrize(
    "tamper", ["content_hash", "payload", "artifact_type", "parent_artifact_ids", "parent_edge"]
)
def test_modern_artifact_row_tampering_fails_on_all_snapshot_entries(tmp_path, tamper):
    database = Database(f"sqlite:///{tmp_path / 'artifact-row-tamper.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        row = session.get(
            ContentArtifactRow,
            "claims-artifact" if tamper in {"parent_artifact_ids", "parent_edge"} else "source-artifact",
        )
        assert row is not None
        if tamper == "content_hash":
            row.content_hash = "tampered-hash"
        elif tamper == "payload":
            payload = dict(row.payload)
            payload["source_ref"] = "tampered-source"
            row.payload = payload
        elif tamper == "artifact_type":
            row.artifact_type = "media"
        elif tamper == "parent_artifact_ids":
            row.parent_artifact_ids = []
        else:
            edge = session.query(ContentArtifactEdgeRow).filter_by(artifact_id="claims-artifact").first()
            assert edge is not None
            edge.parent_artifact_id = "tampered-parent"

    with pytest.raises(SnapshotIntegrityError):
        store.get(snapshot.content_snapshot_id)
    with pytest.raises(SnapshotIntegrityError):
        store.list_for_source(snapshot.source_type, snapshot.source_ref)
    with pytest.raises(SnapshotIntegrityError):
        store.save(snapshot)
    with database.session_factory.begin() as session:
        with pytest.raises(SnapshotIntegrityError):
            VerificationRefreshService._persist_snapshot(session, snapshot)


def test_modern_missing_artifact_fails_on_all_snapshot_entries(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'artifact-missing.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        session.delete(session.get(ContentArtifactRow, "source-artifact"))

    with pytest.raises(SnapshotIntegrityError):
        store.get(snapshot.content_snapshot_id)
    with pytest.raises(SnapshotIntegrityError):
        store.list_for_source(snapshot.source_type, snapshot.source_ref)
    with pytest.raises(SnapshotIntegrityError):
        store.save(snapshot)
    with database.session_factory.begin() as session:
        with pytest.raises(SnapshotIntegrityError):
            VerificationRefreshService._persist_snapshot(session, snapshot)


def test_modern_snapshot_missing_members_fails_closed(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'snapshot-v2-members.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        session.execute(
            delete(ContentSnapshotArtifactRow).where(
                ContentSnapshotArtifactRow.content_snapshot_id == snapshot.content_snapshot_id
            )
        )

    with pytest.raises(SnapshotIntegrityError):
        store.get(snapshot.content_snapshot_id)
    with pytest.raises(SnapshotIntegrityError):
        store.list_for_source(snapshot.source_type, snapshot.source_ref)
    with pytest.raises(SnapshotIntegrityError):
        store.save(snapshot)


def test_legacy_v1_identity_without_additive_keys_remains_readable(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'snapshot-legacy.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    snapshot = _snapshot()
    _persist_artifacts(database)
    store.save(snapshot)

    with database.session_factory.begin() as session:
        row = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
        assert row is not None
        row.schema_version = "content.snapshot.v1"
        row.identity = {
            key: value
            for key, value in dict(row.identity).items()
            if key
            not in {
                "source_artifact_id",
                "artifact_root_hash",
                "snapshot_kind",
                "parent_snapshot_id",
                "supersedes_snapshot_id",
                "producer_manifest",
                "model_versions",
                "prompt_versions",
                "configuration",
                "external_snapshots",
                "policy_versions",
            }
        }
        row.identity["schema_version"] = "content.snapshot.v1"
        session.execute(
            delete(ContentSnapshotArtifactRow).where(
                ContentSnapshotArtifactRow.content_snapshot_id == snapshot.content_snapshot_id
            )
        )

    fetched = store.get(snapshot.content_snapshot_id)
    assert fetched is not None
    assert fetched.artifact_ids == snapshot.artifact_ids
    assert [item.content_snapshot_id for item in store.list_for_source(
        snapshot.source_type, snapshot.source_ref
    )] == [snapshot.content_snapshot_id]

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateTable

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ClaimEvidenceRow,
    ClaimVerificationResultRow,
    ContentArtifactEdgeRow,
    ContentArtifactRow,
    ContentSnapshotArtifactRow,
    ContentSnapshotRow,
    ContentStageCheckpointRow,
    FinancialClaimRow,
    SignalOutboxRow,
)
from stock_content.adapters.postgres.repositories.artifact_repository import (
    ArtifactIntegrityError,
    SqlArtifactRepository,
)
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import SnapshotIntegrityError, SqlSnapshotStore
from stock_content.domain.artifacts import SourceArtifact
from stock_content.domain.lineage import build_content_snapshot


def test_sqlite_upgrade_backfills_013_claim_and_lifecycle(tmp_path):
    url = f"sqlite:///{tmp_path / 'legacy-claim.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE financial_claim (
                claim_id VARCHAR(80) PRIMARY KEY, claim_type VARCHAR(40) NOT NULL,
                fact_category VARCHAR(20) NOT NULL, subject_type VARCHAR(40) NOT NULL,
                subject_id VARCHAR(128) NOT NULL, predicate VARCHAR(255) NOT NULL,
                value JSON, unit VARCHAR(40), fact_time TIMESTAMP, published_at TIMESTAMP,
                evidence_refs JSON NOT NULL, source_confidence FLOAT NOT NULL,
                extractor_confidence FLOAT NOT NULL, video_id VARCHAR(64),
                content_snapshot_id VARCHAR(80), created_at TIMESTAMP NOT NULL
            )
        """))
        connection.execute(text("""
            CREATE TABLE claim_verification_lifecycle (
                claim_id VARCHAR(80) PRIMARY KEY, status VARCHAR(40) NOT NULL,
                retry_count INTEGER NOT NULL, next_retry_at TIMESTAMP,
                market_snapshot_id VARCHAR(128), market_data_version VARCHAR(80),
                fact_date DATE, adjustment VARCHAR(20), verification_timestamp TIMESTAMP,
                verification_rule_version VARCHAR(80) NOT NULL, result JSON NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """))
        connection.execute(
            text("""
                INSERT INTO financial_claim(
                    claim_id, claim_type, fact_category, subject_type, subject_id, predicate,
                    value, evidence_refs, source_confidence, extractor_confidence, created_at
                    ) VALUES ('legacy-claim', 'PRICE', 'FACT', 'EQUITY', '600000.SH', 'price',
                        10, :evidence_refs, 0.8, 0.9, '2026-01-01T00:00:00')
                """),
                {"evidence_refs": json.dumps(["evidence-old"])},
            )
        connection.execute(
            text("""
                INSERT INTO claim_verification_lifecycle(
                    claim_id, status, retry_count, verification_rule_version, result, updated_at
                ) VALUES ('legacy-claim', 'VERIFIED', 2, 'verification_rule.legacy',
                    :result, '2026-01-02T00:00:00')
            """),
            {"result": json.dumps({"reference_value": 10})},
        )

    database = Database(url)
    database.create_schema()
    database.create_schema()  # upgrade is safe to replay
    claims = SqlClaimRepository(database.session_factory)
    claim = claims.get("legacy-claim")
    assert claim is not None
    assert claim.claim_type == "PRICE"
    assert claim.evidence_refs == ["evidence-old"]
    claims.save(claim)  # backfilled membership remains idempotent
    assert claims.evidence("legacy-claim") == ["evidence-old"]
    verifications = claims.verifications("legacy-claim")
    assert any(item["status"] == "VERIFIED" for item in verifications)
    assert any(item.get("provider") == "legacy_lifecycle" for item in verifications)
    legacy_items = [item for item in verifications if item["verification_id"].startswith("legacy-")]
    assert len(legacy_items) == 1

    with database.session_factory() as session:
        assert session.scalar(select(ClaimEvidenceRow.evidence_id)) == "evidence-old"
        assert session.scalar(select(ClaimVerificationResultRow.status)) == "VERIFIED"
        assert session.scalar(select(ClaimEvidenceRow.member_id)) == hashlib.md5(
            b"legacy-claim:evidence-old", usedforsecurity=False
        ).hexdigest()
        assert session.scalar(select(ClaimVerificationResultRow.verification_id)) == (
            "legacy-" + hashlib.md5(b"legacy-claim", usedforsecurity=False).hexdigest()
        )


def test_immutable_conflicts_are_domain_errors_and_members_are_idempotent(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'immutable.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    source = SourceArtifact(
        artifact_id="fixed-source", artifact_type="source", source_type="fixture",
        source_ref="one", raw_content_hash="raw", parent_artifact_ids=(),
    )
    same = SourceArtifact(**{**source.__dict__, "created_at": datetime(2030, 1, 1, tzinfo=UTC)})
    artifacts.put(source)
    artifacts.put(same)
    artifacts.put(same)
    with pytest.raises(ArtifactIntegrityError):
        artifacts.put(SourceArtifact(**{**source.__dict__, "source_ref": "two"}))
    with database.session_factory() as session:
        assert session.scalar(select(ContentArtifactEdgeRow.edge_id)) is None

    snapshots = SqlSnapshotStore(database.session_factory)
    snapshot = build_content_snapshot(
        source_type="fixture", source_ref="one", source_content_hash="raw",
        artifact_ids={"source": "fixed-source"}, code_sha="immutable",
    )
    snapshots.save(snapshot)
    snapshots.save(snapshot)
    conflicting = build_content_snapshot(
        source_type="fixture", source_ref="one", source_content_hash="different",
        artifact_ids={"source": "fixed-source"}, code_sha="immutable",
    )
    conflicting = type(snapshot)(**{**conflicting.__dict__, "content_snapshot_id": snapshot.content_snapshot_id})
    with pytest.raises(SnapshotIntegrityError):
        snapshots.save(conflicting)
    with database.session_factory() as session:
        assert session.scalar(select(ContentSnapshotArtifactRow.member_id)) is not None


def test_artifact_type_and_hash_are_unique_across_legacy_ids(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'artifact-unique.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    first = SourceArtifact(
        artifact_id="legacy-source-a", artifact_type="source", source_type="fixture",
        source_ref="same", source_content_hash="raw", parent_artifact_ids=(),
    )
    second = SourceArtifact(
        artifact_id="legacy-source-b", artifact_type="source", source_type="fixture",
        source_ref="same", source_content_hash="raw", parent_artifact_ids=(),
    )

    stored = artifacts.put(first)
    retry = artifacts.put(second)

    assert stored.artifact_id == first.artifact_id
    assert retry.artifact_id == first.artifact_id
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ContentArtifactRow)) == 1
        # The unique key is enforced by the database, not merely by the
        # repository's preflight lookup.
        assert session.execute(
            text(
                "SELECT COUNT(*) FROM content_artifact "
                "WHERE artifact_type = 'source' AND content_hash = :content_hash"
            ),
            {"content_hash": first.content_hash},
        ).scalar_one() == 1


def test_postgres_migration_contains_legacy_backfill_and_atomic_conflict_guards():
    foundation = Path("migrations/015_content_fact_chain_foundation.sql").read_text(encoding="utf-8")
    refresh = Path("migrations/016_verification_refresh_signal_outbox.sql").read_text(encoding="utf-8")
    claim_table_start = foundation.index("CREATE TABLE IF NOT EXISTS financial_claim")
    claim_table_end = foundation.index("CREATE INDEX IF NOT EXISTS ix_financial_claim_subject")
    claim_table = foundation[claim_table_start:claim_table_end]
    result_table_start = foundation.index("CREATE TABLE IF NOT EXISTS claim_verification_result")
    result_table_end = foundation.index("CREATE TABLE IF NOT EXISTS content_snapshot_artifact")
    result_table = foundation[result_table_start:result_table_end]
    assert "payload JSONB NOT NULL" in claim_table
    assert "result_payload JSONB NOT NULL" in result_table
    payload_create = claim_table_start + claim_table.index("payload JSONB NOT NULL")
    result_create = result_table_start + result_table.index("result_payload JSONB NOT NULL")
    payload_upgrade = foundation.index("ALTER COLUMN payload TYPE JSONB USING payload::jsonb")
    result_upgrade = foundation.index("ALTER COLUMN result_payload TYPE JSONB USING result_payload::jsonb")
    assert payload_create < payload_upgrade
    assert result_create < result_upgrade
    assert "payload JSON NOT NULL" not in claim_table
    history_migration = Path("migrations/026_claim_state_events_publication.sql").read_text(encoding="utf-8")
    assert "SET legacy_history_incomplete = TRUE" in history_migration
    assert "legacy_history_incomplete IS FALSE" in history_migration
    assert "result_payload JSON NOT NULL" not in result_table
    assert "jsonb_build_object" in foundation
    assert "ON CONFLICT(member_id) DO NOTHING" in foundation
    assert "UNIQUE(claim_id, evidence_id)" in foundation
    assert "uq_claim_evidence_claim_evidence" in foundation
    payload_backfill = foundation.index("SET payload = jsonb_build_object")
    assert payload_upgrade < payload_backfill
    lifecycle_copy = refresh.index("INSERT INTO claim_verification_result")
    assert refresh.index("ADD COLUMN IF NOT EXISTS fact_date") < lifecycle_copy
    assert refresh.index("ADD COLUMN IF NOT EXISTS adjustment") < lifecycle_copy
    assert refresh.index("ADD COLUMN IF NOT EXISTS verification_timestamp") < lifecycle_copy
    assert refresh.index("ADD COLUMN IF NOT EXISTS verification_rule_version") < lifecycle_copy
    assert refresh.index("ADD COLUMN IF NOT EXISTS verified_at") < lifecycle_copy
    assert "COALESCE(lifecycle.result, '{}'::jsonb)" in refresh
    assert "|| jsonb_build_object" in refresh
    assert "claim_verification_lifecycle" in refresh


def test_claim_payloads_are_jsonb_on_postgres_and_json_on_sqlite():
    for model, column in (
        (ContentSnapshotRow, "identity"),
        (ContentSnapshotRow, "artifact_ids"),
        (ContentSnapshotRow, "quant_market_snapshot_ids"),
        (ContentSnapshotRow, "producer_manifest"),
        (ContentArtifactRow, "parent_artifact_ids"),
        (ContentArtifactRow, "payload"),
        (ContentStageCheckpointRow, "artifact_ids"),
        (ContentStageCheckpointRow, "artifact_hashes"),
        (ContentStageCheckpointRow, "payload"),
        (FinancialClaimRow, "payload"),
        (FinancialClaimRow, "value"),
        (ClaimVerificationResultRow, "result_payload"),
        (SignalOutboxRow, "payload"),
    ):
        postgres_ddl = str(CreateTable(model.__table__).compile(dialect=postgresql.dialect()))
        sqlite_ddl = str(CreateTable(model.__table__).compile(dialect=sqlite.dialect()))
        assert f"{column} JSONB" in postgres_ddl
        assert f"{column} JSON" in sqlite_ddl
        assert f"{column} JSONB" not in sqlite_ddl


def test_core_json_migrations_use_jsonb_and_upgrade_old_json_columns():
    snapshot = Path("migrations/012_content_snapshot.sql").read_text(encoding="utf-8")
    claims = Path("migrations/013_financial_claims.sql").read_text(encoding="utf-8")
    foundation = Path("migrations/015_content_fact_chain_foundation.sql").read_text(encoding="utf-8")
    for fragment in (
        "identity JSONB",
        "artifact_ids JSONB",
        "quant_market_snapshot_ids JSONB",
    ):
        assert fragment in snapshot
    for fragment in ("value JSONB", "evidence_refs JSONB", "result JSONB"):
        assert fragment in claims
    for fragment in (
        "parent_artifact_ids JSONB",
        "payload JSONB NOT NULL",
        "artifact_ids JSONB",
        "artifact_hashes JSONB",
        "value JSONB",
        "result_payload JSONB NOT NULL",
    ):
        assert fragment in foundation
    for column in (
        "identity",
        "artifact_ids",
        "quant_market_snapshot_ids",
        "producer_manifest",
        "parent_artifact_ids",
        "artifact_hashes",
        "evidence_refs",
        "result_payload",
    ):
        assert f"ALTER COLUMN {column} TYPE JSONB USING {column}::jsonb" in foundation

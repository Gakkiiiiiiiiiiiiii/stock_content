"""Real PostgreSQL N-1 upgrade coverage for the numbered content release."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.migration_ledger import (
    MIGRATION_LEDGER_TABLE,
    apply_migrations,
    expected_migrations,
)
from stock_content.adapters.postgres.models import Base, ContentPublicationRunRow, FinancialClaimRow
from stock_content.adapters.postgres.repositories.claim_event_repository import ClaimStateEventRepository
from stock_content.adapters.postgres.repositories.publication_repository import PublicationRepository
from stock_content.adapters.postgres.repositories.signal_outbox_repository import SignalOutboxRepository
from stock_content.application.publication_unit_of_work import PublicationUnitOfWork
from stock_content.domain.claim_state_event import ClaimStateEvent
from stock_content.domain.publication_run import ContentPublicationRun

POSTGRES_URL = os.getenv("CONTENT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="CONTENT_TEST_POSTGRES_URL is required for real PostgreSQL tests",
)

_N_MINUS_ONE_MIGRATION = "028_source_artifact_governance"
_CURRENT_MIGRATION = "029_publication_sealed_projection"
_N_MINUS_ONE_ABSENT_TABLES = {"content_publication_manifests", "content_sealed_signals"}
_BASELINE_SQL_AUTHORITY = {
    "024_final_claim_evidence_ownership",
    "026_claim_state_events_publication",
}


def _prepare_n_minus_one_release(engine) -> tuple[ClaimStateEvent, dict[str, object]]:
    """Create the release-028 mapped schema and its genuine release ledger.

    This deliberately models a release-028 database, rather than taking a
    current schema and deleting ledger rows: its ledger is complete through
    028, its two 029 tables do not exist, and the SQL-only guards owned by the
    already-recorded migrations are installed from their numbered scripts.
    """
    migrations = expected_migrations()
    assert migrations[-2].migration_id == _N_MINUS_ONE_MIGRATION
    assert migrations[-1].migration_id == _CURRENT_MIGRATION
    n_minus_one = migrations[:-1]

    with engine.begin() as connection:
        Base.metadata.create_all(
            connection,
            tables=[table for table in Base.metadata.sorted_tables if table.name not in _N_MINUS_ONE_ABSENT_TABLES],
        )
        for migration in n_minus_one:
            if migration.migration_id in _BASELINE_SQL_AUTHORITY:
                connection.exec_driver_sql(migration.sql)
        connection.execute(
            text(
                f"CREATE TABLE {MIGRATION_LEDGER_TABLE} ("
                "migration_id VARCHAR(128) PRIMARY KEY, "
                "checksum CHAR(64) NOT NULL, "
                "applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
        )
        for migration in n_minus_one:
            connection.execute(
                text(
                    f"INSERT INTO {MIGRATION_LEDGER_TABLE} (migration_id, checksum) "
                    "VALUES (:migration_id, :checksum)"
                ),
                {"migration_id": migration.migration_id, "checksum": migration.checksum},
            )

    sessions = sessionmaker(engine, expire_on_commit=False)
    root = ClaimStateEvent(
        claim_id="n1-claim",
        event_type="VERIFICATION_INITIAL",
        payload={"release": "028", "immutable": True},
        known_from=datetime(2026, 9, 1, tzinfo=UTC),
        source_available_from=datetime(2026, 9, 1, tzinfo=UTC),
    )
    with sessions.begin() as session:
        session.add(
            FinancialClaimRow(
                claim_id=root.claim_id,
                claim_type="FACT",
                fact_category="FACT",
                subject_type="issuer",
                subject_id="issuer-n1",
                predicate="revenue",
                value={"amount": 42},
                source_confidence=0.9,
                extractor_confidence=0.9,
                extraction_model_id="n1-fixture",
                extraction_prompt_version="n1-fixture",
                claim_schema_version="claim.final.v1",
                normalization_version="n1-fixture",
                source_support_status="SUPPORTED",
                legacy_history_incomplete=False,
                payload={"fixture": "n1"},
            )
        )
    ClaimStateEventRepository(sessions).append(root)
    legacy_identity = ContentPublicationRun(
        content_snapshot_id="n1-snapshot",
        query_hash="n1-query",
        signal_policy_version="signal-policy.v1",
    )
    legacy_publication = {
        "publication_run_id": legacy_identity.publication_run_id,
        "content_snapshot_id": "n1-snapshot",
        "query_hash": "n1-query",
        "signal_policy_version": "signal-policy.v1",
        "state": "PUBLISHED",
        "manifest_hash": "n1-legacy-manifest",
        "version": 7,
    }
    with sessions.begin() as session:
        session.add(ContentPublicationRunRow(**legacy_publication))
    return root, legacy_publication


def _publication_uow(sessions):
    repository = PublicationRepository(sessions)
    outbox = SignalOutboxRepository(sessions)

    def signal_writer(session, rows, *, publication_run_id):
        repository.save_sealed_signals_in_session(session, publication_run_id, tuple(rows))

    return PublicationUnitOfWork(
        repository,
        snapshot_writer=lambda session, snapshot_id, manifest: None,
        signal_writer=signal_writer,
        outbox_writer=lambda session, rows: [outbox.enqueue_in_session(session, row) for row in rows],
    ), repository, outbox


def test_release_028_upgrade_preserves_history_and_publication_replay(postgres_empty_engine):
    root, legacy_publication = _prepare_n_minus_one_release(postgres_empty_engine)
    sessions = sessionmaker(postgres_empty_engine, expire_on_commit=False)
    legacy_publication_statement = text(
        "SELECT publication_run_id, content_snapshot_id, query_hash, signal_policy_version, "
        "state, manifest_hash, version FROM content_publication_runs "
        "WHERE publication_run_id = :publication_run_id"
    )

    with postgres_empty_engine.connect() as connection:
        recorded_before = {
            row.migration_id
            for row in connection.execute(text(f"SELECT migration_id FROM {MIGRATION_LEDGER_TABLE}"))
        }
        assert recorded_before == {migration.migration_id for migration in expected_migrations()[:-1]}
        assert connection.scalar(text("SELECT to_regclass('content_publication_manifests')")) is None
        assert connection.scalar(text("SELECT to_regclass('content_sealed_signals')")) is None
        before_claim = connection.execute(
            text("SELECT claim_id, payload, legacy_history_incomplete FROM financial_claim WHERE claim_id = 'n1-claim'")
        ).one()
        before_event = connection.execute(
            text("SELECT claim_state_event_id, event_hash, payload FROM claim_state_events WHERE claim_id = 'n1-claim'")
        ).one()
        before_publication = connection.execute(
            legacy_publication_statement,
            {"publication_run_id": legacy_publication["publication_run_id"]},
        ).one()

    assert apply_migrations(postgres_empty_engine) == (_CURRENT_MIGRATION,)
    assert apply_migrations(postgres_empty_engine) == ()

    database = Database(POSTGRES_URL)
    database.engine.dispose()
    database.engine = postgres_empty_engine
    database.session_factory = sessions
    database.verify_schema()

    with postgres_empty_engine.connect() as connection:
        assert {
            (row.migration_id, row.checksum)
            for row in connection.execute(text(f"SELECT migration_id, checksum FROM {MIGRATION_LEDGER_TABLE}"))
        } == {(migration.migration_id, migration.checksum) for migration in expected_migrations()}
        assert connection.scalar(text("SELECT to_regclass('content_publication_manifests')")) is not None
        assert connection.scalar(text("SELECT to_regclass('content_sealed_signals')")) is not None
        assert connection.execute(
            text("SELECT claim_id, payload, legacy_history_incomplete FROM financial_claim WHERE claim_id = 'n1-claim'")
        ).one() == before_claim
        assert connection.execute(
            text("SELECT claim_state_event_id, event_hash, payload FROM claim_state_events WHERE claim_id = 'n1-claim'")
        ).one() == before_event
        assert connection.execute(
            legacy_publication_statement,
            {"publication_run_id": legacy_publication["publication_run_id"]},
        ).one() == before_publication

    history = ClaimStateEventRepository(sessions).list_for_claim("n1-claim")
    assert [(event.event_id, event.event_hash) for event in history] == [(root.event_id, root.event_hash)]

    uow, publications, outbox = _publication_uow(sessions)
    assert publications.read_sealed("n1-snapshot", "n1-query", "signal-policy.v1") == {
        "publication_run": publications.get_by_identity("n1-snapshot", "n1-query", "signal-policy.v1"),
        "manifest": None,
        "signals": [],
    }
    signal = {
        "signal_id": "n1-upgrade-signal",
        "content_snapshot_id": "n1-upgrade-snapshot",
        "claim_id": "n1-claim",
        "signal_schema_version": "content-factor-signal.v4",
    }
    manifest = {"artifact_membership": {"claims": "n1-claim"}, "sealed_signals": [signal]}
    first = uow.publish(
        content_snapshot_id="n1-upgrade-snapshot",
        query_hash="n1-upgrade-query",
        signal_policy_version="signal-policy.v1",
        manifest=manifest,
        signals=[signal],
        outbox_events=[signal],
    )
    replay = uow.publish(
        content_snapshot_id="n1-upgrade-snapshot",
        query_hash="n1-upgrade-query",
        signal_policy_version="signal-policy.v1",
        manifest=manifest,
        signals=[signal],
        outbox_events=[signal],
    )
    assert replay == first
    sealed = publications.read_sealed("n1-upgrade-snapshot", "n1-upgrade-query", "signal-policy.v1")
    assert sealed is not None
    assert sealed["manifest"] == manifest
    assert outbox.get_by_signal_id("n1-upgrade-signal").payload == signal

    transient = ClaimStateEvent(
        claim_id="n1-claim",
        event_type="REPLAY_ROLLBACK",
        payload={"must_not_persist": True},
        known_from=datetime(2026, 9, 2, tzinfo=UTC),
        source_available_from=datetime(2026, 9, 2, tzinfo=UTC),
        previous_event_hash=root.event_hash,
    )
    with pytest.raises(RuntimeError, match="rollback fixture"):
        with sessions.begin() as session:
            ClaimStateEventRepository(sessions).append_in_session(session, transient)
            raise RuntimeError("rollback fixture")
    replayed_history = ClaimStateEventRepository(sessions).list_for_claim("n1-claim")
    assert [(event.event_id, event.event_hash) for event in replayed_history] == [
        (root.event_id, root.event_hash)
    ]

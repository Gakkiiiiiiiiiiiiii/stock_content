"""PostgreSQL coverage for the runtime schema guard using an isolated schema."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from stock_content.adapters.postgres.database import SchemaNotReadyError
from stock_content.adapters.postgres.migration_ledger import (
    MIGRATION_LEDGER_TABLE,
    apply_migrations,
    expected_migrations,
)


def test_postgres_schema_probe_is_read_only(postgres_database):
    statements: list[str] = []

    @event.listens_for(postgres_database.engine, "before_cursor_execute")
    def record_ddl(_connection, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith(("CREATE", "ALTER", "DROP", "UPDATE", "INSERT", "DELETE")):
            statements.append(statement)

    postgres_database.verify_schema()

    assert statements == []


def test_postgres_migration_is_idempotent_and_records_the_exact_release(postgres_database):
    assert apply_migrations(postgres_database.engine) == ()
    with postgres_database.engine.connect() as connection:
        rows = connection.execute(text(f"SELECT migration_id, checksum FROM {MIGRATION_LEDGER_TABLE}")).all()
    assert {(row.migration_id, row.checksum) for row in rows} == {
        (migration.migration_id, migration.checksum) for migration in expected_migrations()
    }


def test_concurrent_cold_start_has_one_migration_owner(postgres_empty_engine):
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _unused: apply_migrations(postgres_empty_engine), range(2)))

    assert sum(bool(result) for result in results) == 1
    assert set(results[0] + results[1]) == {migration.migration_id for migration in expected_migrations()}


def test_fresh_baseline_installs_sql_only_authority_guards(postgres_empty_engine):
    apply_migrations(postgres_empty_engine)

    with postgres_empty_engine.begin() as connection:
        trigger = connection.scalar(
            text(
                "SELECT 1 FROM pg_trigger "
                "WHERE tgrelid = to_regclass('claim_evidence') "
                "AND tgname = 'trg_reject_final_claim_evidence' AND NOT tgisinternal"
            )
        )
        index_definition = connection.scalar(
            text(
                "SELECT pg_get_indexdef(indexrelid) FROM pg_index "
                "WHERE indexrelid = to_regclass('uq_claim_state_event_predecessor') "
                "AND indrelid = to_regclass('claim_state_events') AND indisunique"
            )
        )
        connection.execute(
            text(
                "INSERT INTO financial_claim (claim_id, claim_type, fact_category, subject_type, subject_id, "
                "predicate, value, source_confidence, extractor_confidence, extraction_model_id, "
                "extraction_prompt_version, claim_schema_version, normalization_version, source_support_status, "
                "legacy_history_incomplete, payload, created_at) "
                "VALUES ('final-claim', 'FACT', 'FACT', 'issuer', 'issuer-1', 'revenue', "
                "CAST('{}' AS jsonb), 1, 1, 'test', 'test', 'claim.final.v1', 'test', 'UNSUPPORTED', "
                "FALSE, CAST('{}' AS jsonb), CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(DBAPIError, match="claim.final.v1 cannot own"):
            with connection.begin_nested():
                connection.execute(
                    text(
                        "INSERT INTO claim_evidence (member_id, claim_id, evidence_id, relation) "
                        "VALUES ('forbidden-evidence', 'final-claim', 'evidence-1', 'SUPPORTS')"
                    )
                )
        connection.execute(
            text(
                    "INSERT INTO claim_state_events (claim_state_event_id, claim_id, event_type, payload, "
                    "previous_event_hash, event_hash, legacy_history_incomplete, created_at) "
                    "VALUES ('event-root-1', 'final-claim', 'TEST', CAST('{}' AS jsonb), NULL, 'hash-root-1', FALSE, "
                "CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    text(
                        "INSERT INTO claim_state_events (claim_state_event_id, claim_id, event_type, payload, "
                        "previous_event_hash, event_hash, legacy_history_incomplete, created_at) "
                        "VALUES ('event-root-2', 'final-claim', 'TEST', CAST('{}' AS jsonb), NULL, "
                        "'hash-root-2', FALSE, "
                        "CURRENT_TIMESTAMP)"
                    )
                )

    assert trigger == 1
    assert "COALESCE(previous_event_hash" in index_definition


def test_postgres_runtime_rejects_wrong_migration_ledger(postgres_database):
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(f"UPDATE {MIGRATION_LEDGER_TABLE} SET checksum = '0' WHERE migration_id = :migration_id"),
            {"migration_id": expected_migrations()[-1].migration_id},
        )

    with pytest.raises(SchemaNotReadyError, match="ledger is incomplete or wrong"):
        postgres_database.verify_schema()


def test_postgres_runtime_rejects_an_incomplete_migration_ledger(postgres_database):
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(f"DELETE FROM {MIGRATION_LEDGER_TABLE} WHERE migration_id = :migration_id"),
            {"migration_id": expected_migrations()[-1].migration_id},
        )

    with pytest.raises(SchemaNotReadyError, match="missing=.*029_publication_sealed_projection"):
        postgres_database.verify_schema()


@pytest.mark.parametrize(
    ("statement", "guard"),
    (
        ("DROP TRIGGER trg_reject_final_claim_evidence ON claim_evidence", "trg_reject_final_claim_evidence"),
        ("DROP INDEX uq_claim_state_event_predecessor", "uq_claim_state_event_predecessor"),
    ),
)
def test_postgres_runtime_rejects_complete_forged_ledger_without_catalog_guard(postgres_database, statement, guard):
    with postgres_database.engine.begin() as connection:
        connection.execute(text(statement))

    with pytest.raises(SchemaNotReadyError, match=guard):
        postgres_database.verify_schema()


@pytest.mark.parametrize(
    ("statements", "guard"),
    (
        (
            (
                "CREATE OR REPLACE FUNCTION reject_final_claim_evidence() RETURNS trigger "
                "AS $$ BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql",
            ),
            "trg_reject_final_claim_evidence",
        ),
        (
            ("ALTER TABLE claim_evidence DISABLE TRIGGER trg_reject_final_claim_evidence",),
            "trg_reject_final_claim_evidence",
        ),
        (
            (
                "DROP TRIGGER trg_reject_final_claim_evidence ON claim_evidence",
                "CREATE TRIGGER trg_reject_final_claim_evidence "
                "BEFORE INSERT OR UPDATE ON claim_evidence FOR EACH ROW WHEN (false) "
                "EXECUTE FUNCTION reject_final_claim_evidence()",
            ),
            "trg_reject_final_claim_evidence",
        ),
        (
            (
                "DROP TRIGGER trg_reject_final_claim_evidence ON claim_evidence",
                "CREATE TRIGGER trg_reject_final_claim_evidence "
                "BEFORE INSERT OR UPDATE ON claim_evidence FOR EACH ROW "
                "EXECUTE FUNCTION reject_final_claim_evidence('unattested')",
            ),
            "trg_reject_final_claim_evidence",
        ),
        (
            (
                "DROP INDEX uq_claim_state_event_predecessor",
                "CREATE UNIQUE INDEX uq_claim_state_event_predecessor ON claim_state_events "
                "(claim_id, COALESCE(previous_event_hash, '__ROOT__'), event_hash)",
            ),
            "uq_claim_state_event_predecessor",
        ),
    ),
)
def test_postgres_runtime_rejects_complete_ledger_with_forged_guard_shape(postgres_database, statements, guard):
    with postgres_database.engine.begin() as connection:
        for catalog_change in statements:
            connection.execute(text(catalog_change))

    with pytest.raises(SchemaNotReadyError, match=guard):
        postgres_database.verify_schema()

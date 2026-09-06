"""Ordered PostgreSQL migration ownership and release-ledger verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from stock_content.adapters.postgres.models import Base

MIGRATION_LEDGER_TABLE = "content_schema_migrations"
_ADVISORY_LOCK_KEY = 7_320_448_109

# These migrations install PostgreSQL authority that is intentionally outside
# SQLAlchemy's table metadata.  A new database starts from the current mapped
# table shape, so it must execute these scripts before it can truthfully record
# the historical migration set.  Keep this an explicit audited list: adding a
# SQL-only guard requires both its migration id here and a catalog probe below.
_BASELINE_SQL_AUTHORITY_MIGRATIONS = frozenset(
    {
        "024_final_claim_evidence_ownership",
        "026_claim_state_events_publication",
    }
)

_FINAL_CLAIM_EVIDENCE_TRIGGER = "trg_reject_final_claim_evidence"
_FINAL_CLAIM_EVIDENCE_FUNCTION = "reject_final_claim_evidence"
_FINAL_CLAIM_EVIDENCE_TRIGGER_TYPE = 23  # ROW | BEFORE | INSERT | UPDATE
_CLAIM_EVENT_PREDECESSOR_INDEX = "uq_claim_state_event_predecessor"
_CLAIM_EVENT_PREDECESSOR_KEYS = (
    "claim_id",
    "coalesce(previous_event_hash,'__root__')",
)


class MigrationLedgerError(RuntimeError):
    """Raised when the database is not exactly at this release's migration set."""


@dataclass(frozen=True)
class Migration:
    migration_id: str
    checksum: str
    sql: str


def migration_directory() -> Path:
    return Path(__file__).resolve().parents[4] / "migrations"


def expected_migrations() -> tuple[Migration, ...]:
    migrations: list[Migration] = []
    for path in sorted(migration_directory().glob("[0-9][0-9][0-9]_*.sql")):
        payload = path.read_bytes()
        migrations.append(Migration(path.stem, sha256(payload).hexdigest(), payload.decode("utf-8")))
    if not migrations:
        raise MigrationLedgerError("no numbered PostgreSQL migrations are packaged with this release")
    return tuple(migrations)


def _create_ledger(connection: Connection) -> None:
    connection.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {MIGRATION_LEDGER_TABLE} ("
            "migration_id VARCHAR(128) PRIMARY KEY, "
            "checksum CHAR(64) NOT NULL, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
    )


def _preflight_unique_conflicts(connection: Connection, migration: Migration) -> None:
    """Report legacy duplicate keys before a migration's unique-index DDL fails.

    The migration SQL remains authoritative.  This narrow preflight covers the
    simple table/column unique indexes used by the numbered content migrations;
    expression indexes are left to PostgreSQL because their expression is the
    authoritative definition.
    """
    import re

    pattern = re.compile(
        r"CREATE\s+UNIQUE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?[\w_]+\s+"
        r"ON\s+([\w_]+)\s*\(([^()]+)\)",
        re.IGNORECASE,
    )
    tables = set(inspect(connection).get_table_names())
    for table, columns in pattern.findall(migration.sql):
        if table not in tables or any(not column.strip().replace("_", "").isalnum() for column in columns.split(",")):
            continue
        column_names = tuple(column.strip() for column in columns.split(","))
        available_columns = {column["name"] for column in inspect(connection).get_columns(table)}
        # A migration can add the indexed column before it creates the index;
        # only legacy table shapes can be checked before the script runs.
        if not set(column_names) <= available_columns:
            continue
        names = ", ".join(column_names)
        duplicate = connection.execute(
            text(
                f"SELECT {names}, COUNT(*) AS duplicate_count FROM {table} "
                f"GROUP BY {names} HAVING COUNT(*) > 1 LIMIT 5"
            )
        ).mappings().first()
        if duplicate:
            raise MigrationLedgerError(
                f"migration {migration.migration_id} cannot add a unique key to {table} "
                f"because duplicate legacy values exist: {dict(duplicate)}"
            )


def _verify_postgresql_authority_guards(connection: Connection) -> None:
    """Confirm the catalog guards that mapped metadata cannot represent.

    This is deliberately a read-only catalog probe.  The ledger alone is not
    evidence that a baseline has the trigger and expression index which carry
    the final-claim and append-only-chain authority rules.
    """
    expected_function_body = _expected_final_claim_evidence_function_body()
    trigger_rows = connection.execute(
        text(
            "SELECT trigger_row.tgenabled, trigger_row.tgtype, "
            "trigger_row.tgqual IS NULL AS has_no_when_predicate, "
            "trigger_row.tgattr = ''::int2vector AS has_no_update_columns, "
            "trigger_row.tgnargs, trigger_row.tgargs = ''::bytea AS has_no_trigger_arguments, "
            "function_schema.nspname AS function_schema, function_row.proname, "
            "pg_get_function_identity_arguments(function_row.oid) AS identity_arguments, "
            "pg_get_function_result(function_row.oid) AS function_result, "
            "function_language.lanname AS function_language, function_row.prosrc, "
            "function_row.prokind, function_row.prosecdef, function_row.proisstrict, "
            "function_row.provolatile, function_row.proparallel, function_row.proconfig "
            "FROM pg_trigger AS trigger_row "
            "JOIN pg_proc AS function_row ON function_row.oid = trigger_row.tgfoid "
            "JOIN pg_namespace AS function_schema ON function_schema.oid = function_row.pronamespace "
            "JOIN pg_language AS function_language ON function_language.oid = function_row.prolang "
            "WHERE trigger_row.tgrelid = to_regclass('claim_evidence') "
            "AND trigger_row.tgname = :trigger_name "
            "AND NOT trigger_row.tgisinternal"
        ),
        {"trigger_name": _FINAL_CLAIM_EVIDENCE_TRIGGER},
    ).mappings().all()

    missing: list[str] = []
    if len(trigger_rows) != 1:
        missing.append(_FINAL_CLAIM_EVIDENCE_TRIGGER)
    else:
        trigger = trigger_rows[0]
        expected_trigger = {
            "tgenabled": "O",
            "tgtype": _FINAL_CLAIM_EVIDENCE_TRIGGER_TYPE,
            "has_no_when_predicate": True,
            "has_no_update_columns": True,
            "tgnargs": 0,
            "has_no_trigger_arguments": True,
            "function_schema": connection.scalar(text("SELECT current_schema()")),
            "proname": _FINAL_CLAIM_EVIDENCE_FUNCTION,
            "identity_arguments": "",
            "function_result": "trigger",
            "function_language": "plpgsql",
            "prosrc": expected_function_body,
            "prokind": "f",
            "prosecdef": False,
            "proisstrict": False,
            "provolatile": "v",
            "proparallel": "u",
            "proconfig": None,
        }
        if any(trigger[key] != value for key, value in expected_trigger.items()):
            missing.append(f"{_FINAL_CLAIM_EVIDENCE_TRIGGER} (wrong definition)")

    index_rows = connection.execute(
        text(
            "SELECT index_row.indisunique, index_row.indisvalid, index_row.indisready, "
            "index_row.indpred IS NULL AS is_non_partial, index_row.indnkeyatts, index_row.indnatts, "
            "array_agg(pg_get_indexdef(index_row.indexrelid, key_position, TRUE) "
            "ORDER BY key_position) AS key_definitions "
            "FROM pg_index AS index_row "
            "CROSS JOIN LATERAL generate_series(1, index_row.indnatts) AS key_position "
            "WHERE index_row.indexrelid = to_regclass('uq_claim_state_event_predecessor') "
            "AND index_row.indrelid = to_regclass('claim_state_events') "
            "GROUP BY index_row.indexrelid, index_row.indisunique, index_row.indisvalid, "
            "index_row.indisready, index_row.indpred, index_row.indnkeyatts, index_row.indnatts"
        )
    ).mappings().all()
    if len(index_rows) != 1:
        missing.append(_CLAIM_EVENT_PREDECESSOR_INDEX)
    else:
        index = index_rows[0]
        key_definitions = tuple(
            _normalize_postgresql_index_key(definition) for definition in index["key_definitions"]
        )
        if (
            not index["indisunique"]
            or not index["indisvalid"]
            or not index["indisready"]
            or not index["is_non_partial"]
            or index["indnkeyatts"] != 2
            or index["indnatts"] != 2
            or key_definitions != _CLAIM_EVENT_PREDECESSOR_KEYS
        ):
            missing.append(f"{_CLAIM_EVENT_PREDECESSOR_INDEX} (wrong definition)")

    if missing:
        raise MigrationLedgerError(
            "required PostgreSQL catalog guards are missing or wrong: " + ", ".join(missing)
        )


def _expected_final_claim_evidence_function_body() -> str:
    """Extract the audited PL/pgSQL body rather than duplicating it in Python."""
    migration = migration_directory() / "024_final_claim_evidence_ownership.sql"
    match = re.search(
        r"CREATE OR REPLACE FUNCTION reject_final_claim_evidence\(\)\s+"
        r"RETURNS trigger AS \$\$(.*?)\$\$ LANGUAGE plpgsql;",
        migration.read_bytes().decode("utf-8"),
        re.DOTALL,
    )
    if match is None:
        raise MigrationLedgerError("cannot attest final-claim guard: migration 024 has no expected function body")
    return match.group(1)


def _normalize_postgresql_index_key(definition: str) -> str:
    """Normalize only PostgreSQL's display casts and whitespace for exact comparison."""
    return "".join(definition.lower().split()).replace("::charactervarying", "")


def _establish_current_baseline(connection: Connection, migrations: tuple[Migration, ...]) -> None:
    """Create the mapped release schema and execute its SQL-only authority.

    Historical migrations are not all replayable over the modern metadata
    shape.  The baseline therefore applies only the audited scripts whose
    effects do not exist in metadata, then probes their catalog effects before
    any historical migration is entered in the ledger.
    """
    Base.metadata.create_all(connection)
    by_id = {migration.migration_id: migration for migration in migrations}
    missing_scripts = _BASELINE_SQL_AUTHORITY_MIGRATIONS - set(by_id)
    if missing_scripts:
        raise MigrationLedgerError(
            "current baseline is missing packaged SQL authority migrations: " + ", ".join(sorted(missing_scripts))
        )
    for migration in migrations:
        if migration.migration_id in _BASELINE_SQL_AUTHORITY_MIGRATIONS:
            connection.exec_driver_sql(migration.sql)
    _verify_postgresql_authority_guards(connection)


def apply_migrations(engine: Engine) -> tuple[str, ...]:
    """Apply each packaged migration once under one PostgreSQL advisory lock."""
    if engine.dialect.name != "postgresql":
        raise MigrationLedgerError("content-migrate requires PostgreSQL; SQLite schema creation is development-only")
    applied: list[str] = []
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ADVISORY_LOCK_KEY})
        _create_ledger(connection)
        # The earliest migration files describe historical table shapes which
        # are intentionally not replayable from an empty modern database (for
        # example, a later migration expects a renamed frame primary key).
        # Establish the current expand-only baseline only for an empty schema;
        # upgrades of an existing schema still run its numbered history below.
        existing_tables = set(inspect(connection).get_table_names())
        baseline_created = not existing_tables & set(Base.metadata.tables)
        migrations = expected_migrations()
        if baseline_created:
            _establish_current_baseline(connection, migrations)
        recorded = {
            row.migration_id: row.checksum
            for row in connection.execute(text(f"SELECT migration_id, checksum FROM {MIGRATION_LEDGER_TABLE}"))
        }
        for migration in migrations:
            actual = recorded.get(migration.migration_id)
            if actual is not None:
                if actual != migration.checksum:
                    raise MigrationLedgerError(
                        f"migration ledger checksum mismatch for {migration.migration_id}; refusing schema mutation"
                    )
                continue
            if not baseline_created:
                _preflight_unique_conflicts(connection, migration)
                connection.exec_driver_sql(migration.sql)
            connection.execute(
                text(
                    f"INSERT INTO {MIGRATION_LEDGER_TABLE} (migration_id, checksum) "
                    "VALUES (:migration_id, :checksum)"
                ),
                {"migration_id": migration.migration_id, "checksum": migration.checksum},
            )
            applied.append(migration.migration_id)
    return tuple(applied)


def verify_migration_ledger(engine: Engine) -> None:
    """Require a complete, untampered ledger before a PostgreSQL runtime starts."""
    if engine.dialect.name != "postgresql":
        return
    inspector = inspect(engine)
    if MIGRATION_LEDGER_TABLE not in inspector.get_table_names():
        raise MigrationLedgerError("migration ledger is missing; run content-migrate before starting stock_content")
    with engine.connect() as connection:
        recorded = {
            row.migration_id: row.checksum
            for row in connection.execute(text(f"SELECT migration_id, checksum FROM {MIGRATION_LEDGER_TABLE}"))
        }
        _verify_postgresql_authority_guards(connection)
    expected = {migration.migration_id: migration.checksum for migration in expected_migrations()}
    missing = sorted(set(expected) - set(recorded))
    unexpected = sorted(set(recorded) - set(expected))
    wrong = sorted(key for key in set(expected) & set(recorded) if expected[key] != recorded[key])
    if missing or unexpected or wrong:
        details = "; ".join(
            item for item in (
                f"missing={','.join(missing)}" if missing else "",
                f"unexpected={','.join(unexpected)}" if unexpected else "",
                f"checksum_mismatch={','.join(wrong)}" if wrong else "",
            ) if item
        )
        raise MigrationLedgerError(f"migration ledger is incomplete or wrong for this release: {details}")

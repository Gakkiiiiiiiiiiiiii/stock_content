from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import event, inspect
from sqlalchemy.engine import Engine

from stock_content.adapters.postgres.database import Database, SchemaNotReadyError


def _ddl_statements(database: Database) -> list[str]:
    statements: list[str] = []

    @event.listens_for(database.engine, "before_cursor_execute")
    def record_ddl(_connection, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith(("CREATE", "ALTER", "DROP", "UPDATE", "INSERT", "DELETE")):
            statements.append(statement)

    return statements


def test_schema_verifier_rejects_missing_schema_without_creating_it(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'missing.db'}")
    ddl = _ddl_statements(database)

    with pytest.raises(SchemaNotReadyError, match="missing or outdated"):
        database.verify_schema()

    assert inspect(database.engine).get_table_names() == []
    assert ddl == []


def test_create_schema_refuses_postgresql_outside_the_migration_job():
    database = Database("postgresql+psycopg://postgres:postgres@127.0.0.1:55433/stock_content")

    with pytest.raises(RuntimeError, match="development-only for SQLite"):
        database.create_schema()


def test_schema_verifier_accepts_prepared_schema_without_ddl(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'prepared.db'}")
    database.create_schema()  # Explicit test-fixture setup, not production composition.
    ddl = _ddl_statements(database)

    database.verify_schema()

    assert ddl == []


def test_multiple_schema_probes_are_read_only_for_prepared_database(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'concurrent.db'}"
    setup = Database(database_url)
    setup.create_schema()  # Explicit test-fixture setup.
    statements: list[str] = []

    def record_ddl(_connection, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith(("CREATE", "ALTER", "DROP", "UPDATE", "INSERT", "DELETE")):
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record_ddl)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _unused: Database(database_url).verify_schema(), range(8)))
    finally:
        event.remove(Engine, "before_cursor_execute", record_ddl)

    assert statements == []


def test_api_composition_and_health_route_only_probe_prepared_schema(tmp_path, monkeypatch):
    from stock_content.api import dependencies
    from stock_content.api.main import create_app

    database = Database(f"sqlite:///{tmp_path / 'api.db'}")
    database.create_schema()  # Explicit test-fixture setup.
    ddl = _ddl_statements(database)
    monkeypatch.setattr(dependencies, "Database", lambda _url=None: database)

    application = dependencies.build_application(enable_qdrant=False)
    app = create_app(application)
    health_route = next(route for route in app.routes if getattr(route, "path", None) == "/healthz")

    assert health_route.endpoint()["status"] == "ok"
    assert ddl == []


def test_database_workers_only_probe_prepared_schema(tmp_path, monkeypatch):
    from stock_content.workers import signal_publisher_worker, verification_worker

    database = Database(f"sqlite:///{tmp_path / 'workers.db'}")
    database.create_schema()  # Explicit test-fixture setup.
    ddl = _ddl_statements(database)
    monkeypatch.setattr(verification_worker, "Database", lambda: database)
    monkeypatch.setattr(signal_publisher_worker, "Database", lambda: database)

    assert isinstance(verification_worker.run_db_once(), dict)
    assert isinstance(signal_publisher_worker.run_db_once(), dict)
    assert ddl == []


def test_production_composition_has_no_schema_bootstrap_calls():
    root = Path(__file__).resolve().parents[1] / "src" / "stock_content"
    production_sources = (
        root / "api" / "dependencies.py",
        root / "workers" / "content_worker.py",
        root / "workers" / "verification_worker.py",
        root / "workers" / "signal_publisher_worker.py",
        root / "cli" / "rebuild_index.py",
    )
    for source in production_sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_schema"
        ]
        assert not calls, source

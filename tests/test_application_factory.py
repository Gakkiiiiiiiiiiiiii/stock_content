from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stock_content.adapters.postgres.database import Database, SchemaNotReadyError
from stock_content.api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_import_main_has_no_external_side_effects():
    script = """
from unittest.mock import patch
with patch('sqlalchemy.create_engine', side_effect=AssertionError('database opened during import')):
    import stock_content.api.main
    import sys
    assert 'stock_content.api.dependencies' not in sys.modules
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_application_factory_composes_only_during_lifespan():
    calls: list[object] = []

    def factory() -> object:
        calls.append(object())
        return object()

    app = create_app(application_factory=factory)
    assert calls == []

    with TestClient(app) as client:
        assert calls
        assert client.get("/healthz").status_code == 200


def test_default_factory_fails_closed_with_schema_not_ready(tmp_path, monkeypatch):
    from stock_content.api import dependencies

    database = Database(f"sqlite:///{tmp_path / 'empty.db'}")
    monkeypatch.setattr(dependencies, "Database", lambda _url=None: database)
    monkeypatch.delenv("CONTENT_QDRANT_URL", raising=False)

    with pytest.raises(SchemaNotReadyError, match="SCHEMA_NOT_READY"):
        with TestClient(create_app()):
            pass


def test_default_factory_starts_after_explicit_test_schema_setup(tmp_path, monkeypatch):
    from stock_content.api import dependencies

    database = Database(f"sqlite:///{tmp_path / 'migrated.db'}")
    database.create_schema()
    monkeypatch.setattr(dependencies, "Database", lambda _url=None: database)
    monkeypatch.delenv("CONTENT_QDRANT_URL", raising=False)

    with TestClient(create_app()) as client:
        assert client.get("/healthz").json()["status"] == "ok"

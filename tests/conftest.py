"""Shared test-only database setup.

Production entrypoints verify a migrated schema and must not bootstrap one.
This fixture keeps legacy application tests explicit about their development
schema setup without restoring a production DDL path.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def prepared_application_database(monkeypatch):
    """Give application-factory tests an explicitly bootstrapped test database."""
    from stock_content.api import dependencies

    database_class = dependencies.Database

    def create_test_database(url=None):
        database = database_class(url)
        database.create_schema()
        return database

    monkeypatch.setattr(dependencies, "Database", create_test_database)

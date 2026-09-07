"""Explicit SQLite migration surrogate for application-factory tests.

Production uses ``stock-content-migrate`` against PostgreSQL.  SQLite is a
test-only fixture and is created only by callers that deliberately request it.
"""
from __future__ import annotations

from stock_content.adapters.postgres.database import Database


def create_migrated_test_database(url: str) -> Database:
    database = Database(url)
    database.create_schema()
    return database

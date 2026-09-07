"""Regression coverage for opt-in PostgreSQL fixture registration."""

from __future__ import annotations


def test_root_registered_postgres_database_skips_cleanly_without_opt_in(postgres_database):
    """The fixture is visible outside ``tests/postgres`` in any collection order."""
    assert postgres_database is not None

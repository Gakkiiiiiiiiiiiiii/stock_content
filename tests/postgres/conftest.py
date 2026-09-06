"""Opt-in PostgreSQL fixtures for publication transaction tests."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.migration_ledger import apply_migrations
from stock_content.adapters.postgres.models import Base


def _isolated_postgres_engine(url: str, schema: str):
    """Return an engine whose ORM and unqualified SQL can see only ``schema``."""
    base_url = make_url(url)
    query = dict(base_url.query)
    # The final -c setting overrides any caller-provided search_path, so
    # acceptance-test connections never fall through to ``public``.
    # schema_translate_map also makes metadata DDL deterministic when a
    # same-named table exists in another schema.
    existing_options = str(query.get("options") or "").strip()
    query["options"] = f"{existing_options} -csearch_path={schema}".strip()
    return create_engine(
        base_url.set(query=query),
        pool_pre_ping=True,
    ).execution_options(schema_translate_map={None: schema})


def _create_isolated_schema(url: str, prefix: str) -> str:
    schema = prefix + uuid4().hex[:16]
    admin = create_engine(url, pool_pre_ping=True)
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    finally:
        admin.dispose()
    return schema


def _drop_isolated_schema(url: str, schema: str) -> None:
    cleanup = create_engine(url, pool_pre_ping=True)
    try:
        with cleanup.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    finally:
        cleanup.dispose()


@pytest.fixture
def postgres_publication_store():
    url = os.getenv("CONTENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("CONTENT_TEST_POSTGRES_URL is not set")

    schema = _create_isolated_schema(url, "publication_test_")
    scoped_engine = _isolated_postgres_engine(url, schema)
    try:
        Base.metadata.create_all(scoped_engine)
        yield sessionmaker(scoped_engine, expire_on_commit=False), schema
    finally:
        scoped_engine.dispose()
        _drop_isolated_schema(url, schema)


@pytest.fixture
def postgres_database():
    """Fresh Database facade backed by one isolated PostgreSQL schema per test."""
    url = os.getenv("CONTENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("CONTENT_TEST_POSTGRES_URL is not set")

    schema = _create_isolated_schema(url, "claim_event_")
    database = Database(url)
    scoped_engine = _isolated_postgres_engine(url, schema)
    database.engine.dispose()
    database.engine = scoped_engine
    database.session_factory = sessionmaker(scoped_engine, expire_on_commit=False)
    try:
        apply_migrations(scoped_engine)
        yield database
    finally:
        scoped_engine.dispose()
        _drop_isolated_schema(url, schema)


@pytest.fixture
def postgres_empty_engine():
    """One empty, isolated schema for migration-owner concurrency tests."""
    url = os.getenv("CONTENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("CONTENT_TEST_POSTGRES_URL is not set")

    schema = _create_isolated_schema(url, "migration_test_")
    engine = _isolated_postgres_engine(url, schema)
    try:
        yield engine
    finally:
        engine.dispose()
        _drop_isolated_schema(url, schema)

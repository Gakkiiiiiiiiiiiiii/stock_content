"""Opt-in PostgreSQL fixtures for publication transaction tests."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import Base


@pytest.fixture
def postgres_publication_store():
    url = os.getenv("CONTENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("CONTENT_TEST_POSTGRES_URL is not set")

    schema = "publication_test_" + uuid4().hex[:16]
    admin = create_engine(url, pool_pre_ping=True)
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    finally:
        admin.dispose()

    base_url = make_url(url)
    query = dict(base_url.query)
    existing_options = str(query.get("options") or "").strip()
    query["options"] = f'{existing_options} -csearch_path="{schema}",public'.strip()
    scoped_engine = create_engine(base_url.set(query=query), pool_pre_ping=True)
    try:
        with scoped_engine.begin() as connection:
            Base.metadata.create_all(connection)
        yield sessionmaker(scoped_engine, expire_on_commit=False), schema
    finally:
        scoped_engine.dispose()
        cleanup = create_engine(url)
        try:
            with cleanup.begin() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            cleanup.dispose()

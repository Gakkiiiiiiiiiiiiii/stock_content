"""Real PostgreSQL concurrency and atomicity tests for claim state events.

These tests are intentionally opt-in.  Set ``CONTENT_TEST_POSTGRES_URL`` to a
PostgreSQL DSN to run them; normal local/SQLite test runs skip the module.
"""

from __future__ import annotations

import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.claim_event_repository import ClaimStateEventRepository
from stock_content.domain.claim_state_event import ClaimStateEvent, validate_event_chain

POSTGRES_URL = os.getenv("CONTENT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="CONTENT_TEST_POSTGRES_URL is required for real PostgreSQL tests",
)


@pytest.fixture(scope="module")
def postgres_database():
    if not POSTGRES_URL:  # pragma: no cover - module mark handles normal path
        pytest.skip("CONTENT_TEST_POSTGRES_URL is required")
    schema = f"claim_event_{secrets.token_hex(8)}"
    base_url = make_url(POSTGRES_URL)
    admin = create_engine(base_url, pool_pre_ping=True)
    schema_created = False
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        schema_created = True
        query = dict(base_url.query)
        existing_options = str(query.get("options") or "").strip()
        query["options"] = f"{existing_options} -csearch_path={schema},public".strip()
        scoped_url = base_url.set(query=query)
        database = Database(scoped_url.render_as_string(hide_password=False))
        database.create_schema()
        yield database
    finally:
        if schema_created:
            with admin.begin() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


def _event(
    suffix: str,
    *,
    claim_id: str = "claim-pg",
    previous_event_hash: str | None = None,
) -> ClaimStateEvent:
    known_from = datetime.now(UTC) + timedelta(seconds=1)
    return ClaimStateEvent(
        claim_id=claim_id,
        event_type="VERIFICATION_INITIAL",
        payload={"suffix": suffix},
        known_from=known_from,
        source_available_from=known_from,
        previous_event_hash=previous_event_hash,
    )


def _repository(database: Database) -> ClaimStateEventRepository:
    return ClaimStateEventRepository(database.session_factory)


def test_concurrent_root_appends_have_one_successor(postgres_database):
    repository = _repository(postgres_database)
    first, second = _event("root-a"), _event("root-b")

    def append(event):
        try:
            repository.append(event)
            return "success"
        except Exception as exc:  # database race may surface as IntegrityError or tail conflict
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, (first, second)))
    assert sum(result == "success" for result in results) == 1
    stored = repository.list_for_claim("claim-pg")
    assert len(stored) == 1
    assert validate_event_chain(stored)[0].previous_event_hash is None


def test_concurrent_same_predecessor_cannot_fork(postgres_database):
    repository = _repository(postgres_database)
    root = _event("chain-root", claim_id="claim-fork")
    repository.append(root)
    first = _event("successor-a", claim_id="claim-fork", previous_event_hash=root.event_hash)
    second = _event("successor-b", claim_id="claim-fork", previous_event_hash=root.event_hash)

    def append(event):
        try:
            repository.append(event)
            return "success"
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, (first, second)))
    assert sum(result == "success" for result in results) == 1
    stored = repository.list_for_claim("claim-fork")
    assert len(stored) == 2
    assert len(validate_event_chain(stored)) == 2


def test_duplicate_is_idempotent_and_conflicting_payload_is_rejected(postgres_database):
    repository = _repository(postgres_database)
    original = _event("idempotent", claim_id="claim-idempotent")
    repository.append(original)
    repository.append(original)
    assert len(repository.list_for_claim("claim-idempotent")) == 1

    conflicting = _event("different", claim_id="claim-idempotent")
    object.__setattr__(conflicting, "event_id", original.event_id)
    object.__setattr__(conflicting, "event_hash", original.event_hash)
    with pytest.raises(ValueError, match="event_id does not match canonical event identity"):
        repository.append(conflicting)
    assert len(repository.list_for_claim("claim-idempotent")) == 1


def test_transaction_rollback_leaves_no_event(postgres_database):
    repository = _repository(postgres_database)
    event = _event("rollback", claim_id="claim-rollback")
    with pytest.raises(RuntimeError, match="injected failure"):
        with postgres_database.session_factory.begin() as session:
            repository.append_in_session(session, event)
            raise RuntimeError("injected failure")
    assert repository.list_for_claim("claim-rollback") == []


def test_tampered_event_and_wrong_previous_hash_are_rejected(postgres_database):
    repository = _repository(postgres_database)
    root = _event("integrity-root", claim_id="claim-integrity")
    repository.append(root)

    wrong_previous = _event("wrong-previous", claim_id="claim-integrity", previous_event_hash="not-the-tail")
    with pytest.raises(ValueError, match="previous_event_hash"):
        repository.append(wrong_previous)

    tampered = _event("tampered", claim_id="claim-integrity", previous_event_hash=root.event_hash)
    object.__setattr__(tampered, "event_hash", "0" * 64)
    with pytest.raises(ValueError, match="event_hash"):
        repository.append(tampered)
    assert len(repository.list_for_claim("claim-integrity")) == 1

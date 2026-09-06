from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import update

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ClaimStateEventRow
from stock_content.adapters.postgres.repositories.claim_event_repository import ClaimStateEventRepository
from stock_content.domain.claim_state_event import ClaimStateEvent


def test_append_uses_hash_chain_tail_when_event_timestamps_tie(tmp_path):
    """A replay may project several claim events inside one SQLite clock tick."""
    known_from = datetime(2025, 1, 2, tzinfo=UTC)
    for salt in range(100):
        root = ClaimStateEvent(
            claim_id="claim-tied-tail",
            event_type="VERIFICATION_INITIAL",
            payload={"salt": salt, "position": "root"},
            known_from=known_from,
            source_available_from=known_from,
        )
        successor = ClaimStateEvent(
            claim_id="claim-tied-tail",
            event_type="LIFECYCLE",
            payload={"salt": salt, "position": "successor"},
            known_from=known_from,
            source_available_from=known_from,
            previous_event_hash=root.event_hash,
        )
        if root.event_id > successor.event_id:
            break
    else:  # pragma: no cover - the stable SHA-256 space always yields a pair.
        raise AssertionError("failed to construct a timestamp-tie regression fixture")

    database = Database(f"sqlite:///{tmp_path / 'events.db'}")
    database.create_schema()
    events = ClaimStateEventRepository(database.session_factory)
    events.append(root)
    events.append(successor)

    with database.session_factory.begin() as session:
        session.execute(
            update(ClaimStateEventRow)
            .where(ClaimStateEventRow.claim_id == root.claim_id)
            .values(created_at=known_from)
        )

    child = ClaimStateEvent(
        claim_id=root.claim_id,
        event_type="VERIFICATION_REFRESH",
        payload={"position": "child"},
        known_from=known_from,
        source_available_from=known_from,
        previous_event_hash=successor.event_hash,
    )
    events.append(child)

    assert [event.event_id for event in events.list_for_claim(root.claim_id)] == [
        root.event_id,
        successor.event_id,
        child.event_id,
    ]

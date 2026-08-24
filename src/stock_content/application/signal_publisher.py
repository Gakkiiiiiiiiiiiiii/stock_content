"""Durable signal outbox publisher application."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from stock_content.adapters.postgres.repositories.signal_outbox_repository import SignalOutboxRepository


class SignalPublisher(Protocol):
    def publish(self, payload: dict, idempotency_key: str) -> None: ...


class SignalPublisherApplication:
    def __init__(self, outbox: SignalOutboxRepository, publisher: SignalPublisher) -> None:
        self._outbox = outbox
        self._publisher = publisher

    def run_once(
        self, worker_id: str = "signal-publisher", limit: int = 10, now: datetime | None = None
    ) -> dict[str, int]:
        current = now or datetime.now(UTC)
        rows = self._outbox.claim_due(worker_id, limit=limit, now=current)
        published = 0
        retried = 0
        for row in rows:
            try:
                self._publisher.publish(dict(row.payload or {}), row.signal_id)
                self._outbox.mark_published(row.outbox_id, worker_id, current)
                published += 1
            except Exception as exc:  # noqa: BLE001 - durable retry boundary
                self._outbox.mark_retry(row.outbox_id, worker_id, str(exc), current)
                retried += 1
        return {"claimed": len(rows), "published": published, "retried": retried}


__all__ = ["SignalPublisher", "SignalPublisherApplication"]

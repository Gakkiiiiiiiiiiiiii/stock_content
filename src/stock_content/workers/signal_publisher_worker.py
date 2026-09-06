"""Production signal outbox publisher entrypoint."""
from __future__ import annotations

import logging
import os
import time

import httpx

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.signal_outbox_repository import SignalOutboxRepository
from stock_content.application.signal_publisher import SignalPublisherApplication
from stock_content.domain.lineage import default_code_sha
from stock_content.domain.worker_capability import TaskKind, WorkerProfile, require_capability

LOGGER = logging.getLogger(__name__)
WORKER_PROFILE = WorkerProfile.CORE
QUEUE = TaskKind.INDEX


class HttpSignalPublisher:
    def __init__(self, endpoint: str | None = None, *, dry_run: bool = False, timeout: float = 10.0) -> None:
        self.endpoint = endpoint or os.getenv("CONTENT_SIGNAL_PUBLISH_URL") or os.getenv("SIGNAL_PUBLISHER_URL")
        self.dry_run = dry_run or os.getenv("SIGNAL_PUBLISHER_DRY_RUN", "").lower() in {"1", "true", "yes"}
        self.timeout = timeout

    def publish(self, payload: dict, idempotency_key: str) -> None:
        if self.dry_run:
            return
        if not self.endpoint:
            raise RuntimeError("CONTENT_SIGNAL_PUBLISH_URL is required for signal publishing")
        producer = dict(payload.get("producer") or {})
        headers = {
            "Idempotency-Key": idempotency_key,
            "X-Trace-Id": str(producer.get("trace_id") or "unknown"),
            "X-Caller-Service": "stock_content",
        }
        response = httpx.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()


def run_db_once(worker_id: str = "signal-publisher", limit: int = 10) -> dict[str, int]:
    default_code_sha()
    database = Database()
    database.verify_schema()
    return SignalPublisherApplication(
        SignalOutboxRepository(database.session_factory), HttpSignalPublisher()
    ).run_once(worker_id, limit)


def main() -> None:  # pragma: no cover
    require_capability(WORKER_PROFILE, QUEUE)
    logging.basicConfig(level=logging.INFO)
    default_code_sha()
    interval = float(os.getenv("SIGNAL_PUBLISH_POLL_SECONDS", "30"))
    while True:
        try:
            LOGGER.info("signal publisher pass: %s", run_db_once())
        except Exception:
            LOGGER.exception("signal publisher pass failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()

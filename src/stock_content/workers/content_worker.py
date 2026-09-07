from __future__ import annotations

import logging
import os
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

from stock_content.api.dependencies import build_application
from stock_content.domain.worker_capability import TaskKind, WorkerProfile, require_capability

LOGGER = logging.getLogger("stock_content.worker")
WORKER_PROFILE = WorkerProfile(os.getenv("CONTENT_WORKER_PROFILE", WorkerProfile.CORE.value))
QUEUE = TaskKind(
    os.getenv(
        "CONTENT_WORKER_QUEUE",
        TaskKind.VIDEO_PIPELINE.value if WORKER_PROFILE is WorkerProfile.VIDEO else TaskKind.CORE.value,
    )
)


def _write_video_heartbeat() -> None:
    """Publish a non-secret liveness record for the API readiness probe."""
    if WORKER_PROFILE is not WorkerProfile.VIDEO:
        return
    configured = os.getenv("CONTENT_VIDEO_WORKER_HEARTBEAT_FILE", "")
    if not configured:
        return
    path = Path(configured)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            '{"profile":"video","observed_at":"' + datetime.now(UTC).isoformat().replace("+00:00", "Z") + '"}',
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        # The worker remains safe if its optional diagnostics volume is down;
        # the API will truthfully report it as unready.
        LOGGER.warning("video worker heartbeat could not be written")


def run_forever() -> None:
    require_capability(WORKER_PROFILE, QUEUE)
    logging.basicConfig(level=os.getenv("CONTENT_LOG_LEVEL", "INFO"))
    application = build_application()
    worker_id = os.getenv("CONTENT_WORKER_ID", f"{socket.gethostname()}:{os.getpid()}")
    poll_seconds = float(os.getenv("CONTENT_WORKER_POLL_SECONDS", "2"))
    lease_seconds = int(os.getenv("CONTENT_TASK_LEASE_SECONDS", "900"))
    retention_interval = max(1, int(os.getenv("CONTENT_RETENTION_SWEEP_INTERVAL_SECONDS", "3600")))
    projection_interval = max(1, int(os.getenv("CONTENT_QDRANT_PROJECTION_POLL_INTERVAL", "1")))
    next_retention_sweep = 0.0
    next_projection_sweep = 0.0
    LOGGER.info("content worker started", extra={"worker_id": worker_id})
    while True:
        _write_video_heartbeat()
        scheduler = getattr(application, "_retention_scheduler", None)
        if scheduler is not None and time.monotonic() >= next_retention_sweep:
            # The scheduler is deliberately hosted by an existing worker.  A
            # durable tombstone/effect state makes this at-least-once timer
            # safe across restarts; no extra Compose service is introduced.
            try:
                scheduler.sweep(dry_run=False)
            except Exception:
                LOGGER.exception("retention sweep failed")
            next_retention_sweep = time.monotonic() + retention_interval
        if time.monotonic() >= next_projection_sweep:
            # This is deliberately outside ``process_next``: an unavailable
            # Qdrant can leave projection intent pending, but cannot affect a
            # task's SQL publication or SUCCEEDED terminal state.
            try:
                application.dispatch_knowledge_projections(worker_id)
            except Exception:
                LOGGER.exception("knowledge projection dispatch failed")
            next_projection_sweep = time.monotonic() + projection_interval
        result = application.process_next(worker_id, QUEUE, lease_seconds)
        _write_video_heartbeat()
        if result is None:
            time.sleep(poll_seconds)
        else:
            LOGGER.info("content task processed", extra=result)


def main() -> None:
    run_forever()


if __name__ == "__main__":
    main()

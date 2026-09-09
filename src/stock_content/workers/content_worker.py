from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_content.adapters.media.ocr import PaddleOcrEngine
from stock_content.adapters.sources.xiaoe_page import page_resolver_from_environment
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

_VIDEO_HEARTBEAT_SCHEMA = "video-worker-readiness.v1"
_XIAOE_MATERIALIZER_IDENTITY = "XiaoeMaterializer.local.v1"
_TARGETED_FRAME_EXTRACTOR_IDENTITY = "FfmpegFrameExtractor.extract_targeted.v1"


def _ocr_attestation() -> dict[str, str]:
    """Copy only the isolated OCR process's non-secret device proof.

    The video worker does not infer GPU availability from its own environment.
    It republishes the observed proof produced by the separate Paddle runtime,
    so a configured ``gpu:0`` can never silently become a CPU capability.
    """
    path = os.getenv("CONTENT_OCR_HEARTBEAT_FILE", "")
    if not path:
        return {"health_code": "OCR_HEARTBEAT_MISSING"}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            "health_code": str(payload.get("health_code") or "OCR_HEARTBEAT_INVALID"),
            "requested_device": str(payload.get("requested_device") or ""),
            "actual_device": str(payload.get("actual_device") or ""),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"health_code": "OCR_HEARTBEAT_INVALID"}


def _video_readiness_payload() -> dict[str, Any]:
    """Build a durable, non-secret capability proof for the API process."""
    xiaoe_enabled = os.getenv("CONTENT_XIAOE_PAGE_RESOLVER_ENABLED", "").lower() == "true"
    credential_ref = os.getenv("CONTENT_XIAOE_CREDENTIAL_REF", "xiaoe-storage-state").strip()
    credential_provider = os.getenv("CONTENT_XIAOE_CREDENTIAL_PROVIDER", "file-secret").strip()
    template = os.getenv("CONTENT_XIAOE_PAGE_URL_TEMPLATE", "")
    state_file = os.getenv("CONTENT_XIAOE_STORAGE_STATE_FILE", "")
    resolver_ready = False
    if xiaoe_enabled and credential_ref and credential_provider and template and Path(state_file).is_file():
        # This only verifies worker-local configuration and the private mount's
        # existence through the resolver constructor.  The storage state is
        # never read or copied into the heartbeat.
        resolver_ready = page_resolver_from_environment() is not None
    extractor_ready = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    return {
        "schema": _VIDEO_HEARTBEAT_SCHEMA,
        "profile": "video",
        "health_code": "READY" if resolver_ready and extractor_ready else "CAPABILITY_NOT_READY",
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "xiaoe_page": {
            "enabled": xiaoe_enabled,
            "ready": resolver_ready,
            # These are opaque identifiers, not credential values.  They stay
            # in the worker-state volume and are never returned by the API.
            "credential_ref": credential_ref,
            "credential_provider": credential_provider,
            "materializer_identity": _XIAOE_MATERIALIZER_IDENTITY,
        },
        "frame_extractor": {
            "ready": extractor_ready,
            "identity": _TARGETED_FRAME_EXTRACTOR_IDENTITY,
        },
        "ocr": _ocr_attestation(),
    }


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
            json.dumps(_video_readiness_payload(), sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        # The worker remains safe if its optional diagnostics volume is down;
        # the API will truthfully report it as unready.
        LOGGER.warning("video worker heartbeat could not be written")


def _probe_video_ocr_runtime() -> None:
    """Prove the isolated OCR runtime, then release the startup probe.

    Pipeline stages create their own long-lived OCR adapter when work arrives;
    retaining this preliminary process would otherwise leak an idle GPU child.
    """

    engine = PaddleOcrEngine()
    try:
        engine.start_and_probe()
    finally:
        engine.close()


def run_forever() -> None:
    require_capability(WORKER_PROFILE, QUEUE)
    logging.basicConfig(level=os.getenv("CONTENT_LOG_LEVEL", "INFO"))
    # The video queue owns visual stages. It must not claim work until the
    # separate Paddle process has completed initialization and a real probe.
    if WORKER_PROFILE is WorkerProfile.VIDEO:
        _probe_video_ocr_runtime()
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

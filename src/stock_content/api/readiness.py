"""Framework adapter for the pure readiness application service."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import yaml
from fastapi import APIRouter, Response, status
from sqlalchemy import select, text

from stock_content.adapters.postgres.models import (
    ContentPublicationRunRow,
    ContentSnapshotRow,
    ContentTaskEffectRow,
    SignalOutboxRow,
)
from stock_content.adapters.qdrant import NullKnowledgeIndex
from stock_content.application.readiness_service import ReadinessDependencies, ReadinessService, SnapshotReadiness
from stock_content.application.source_resolution_service import credential_allowlist_from_environment

_VIDEO_HEARTBEAT_SCHEMA = "video-worker-readiness.v1"
_XIAOE_MATERIALIZER_IDENTITY = "XiaoeMaterializer.local.v1"
_TARGETED_FRAME_EXTRACTOR_IDENTITY = "FfmpegFrameExtractor.extract_targeted.v1"


def create_readiness_router(
    service: ReadinessService | None = None,
    dependencies: Callable[[], ReadinessDependencies] | None = None,
    operational_context: Callable[[], tuple[object, bool]] | None = None,
) -> APIRouter:
    readiness_service = service or ReadinessService()
    router = APIRouter(tags=["readiness"])

    @router.get(
        "/readiness",
        summary="Report authoritative and derived readiness separately",
        description=(
            "Reports SQL-backed read_only_facts and formal_publish separately "
            "from derived_search. An unavailable or unknown Qdrant index "
            "watermark degrades derived_search and never changes SQL fact "
            "authority."
        ),
    )
    def readiness() -> dict[str, object]:
        return readiness_service.evaluate((dependencies or (lambda: ReadinessDependencies()))()).to_dict()

    def evaluate():
        return readiness_service.evaluate((dependencies or (lambda: ReadinessDependencies()))())

    @router.get("/health/fact-ready")
    def fact_ready(response: Response) -> dict[str, object]:
        report = evaluate()
        response.status_code = _status_for(report.fact)
        return _component_payload(report, "fact")

    @router.get("/health/signal-ready")
    def signal_ready(response: Response) -> dict[str, object]:
        report = evaluate()
        response.status_code = _status_for(report.signal)
        return _component_payload(report, "signal")

    @router.get("/health/search-ready")
    def search_ready(response: Response) -> dict[str, object]:
        report = evaluate()
        response.status_code = _status_for(report.search)
        return _component_payload(report, "search")

    @router.get("/health/video-ingestion-ready")
    def video_ingestion_ready(response: Response) -> dict[str, object]:
        application, _auth_ready = (operational_context or (lambda: (None, False)))()
        components = _video_components((dependencies or (lambda: ReadinessDependencies()))(), application)
        response.status_code = (
            status.HTTP_200_OK
            if all(item["ready"] for item in components.values())
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return {"ready": response.status_code == status.HTTP_200_OK, "components": components}

    @router.get("/health/knowledge-bundle-ready")
    def knowledge_bundle_ready(response: Response) -> dict[str, object]:
        application, auth_ready = (operational_context or (lambda: (None, False)))()
        components = _bundle_components((dependencies or (lambda: ReadinessDependencies()))(), application, auth_ready)
        response.status_code = (
            status.HTTP_200_OK
            if all(item["ready"] for item in components.values())
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        # This is an inter-service readiness wire contract.  Keep the exact
        # schema checksum top-level so a consumer does not need to infer it
        # from a component's boolean state.  The nested components remain for
        # human diagnostics and backwards compatibility.
        return {
            "ready": response.status_code == status.HTTP_200_OK,
            "contract": "content-knowledge-bundle.v1",
            "contract_checksum": _bundle_contract_checksum(),
            "canonicalization_version": "content-bundle-c14n-v1",
            "components": components,
        }

    return router


def dependencies_from_application(application: object) -> ReadinessDependencies:
    """Build a conservative status view from the existing application ports.

    Adapter implementations can replace this provider later; unknown
    persistence state is represented as no READY snapshot, never as a false
    fact due to the search adapter.
    """
    task_repository = getattr(application, "_tasks", None)
    sessions = getattr(task_repository, "_sessions", None)
    postgres_ok = True
    if sessions is not None:
        try:
            with sessions() as session:
                session.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001 - readiness must report degraded state
            postgres_ok = False
    index = getattr(application, "_index", None)
    qdrant_ok = index is not None and not isinstance(index, NullKnowledgeIndex)
    client = getattr(index, "_client", None)
    if qdrant_ok and client is not None:
        try:
            client.get_collections()
        except Exception:  # noqa: BLE001 - search is independently degradable
            qdrant_ok = False
    snapshot_store = getattr(getattr(application, "_snapshots", None), "_store", None)
    publication_uow = getattr(application, "_publication_uow", None)
    publication_repository = getattr(publication_uow, "repository", None)
    sql_sessions = getattr(snapshot_store, "_sessions", None) or getattr(publication_repository, "_sessions", None)
    snapshot, outbox_lag = SnapshotReadiness(None), 0.0
    projection_pending = projection_dead_letters = 0
    if postgres_ok and sql_sessions:
        try:
            snapshot, outbox_lag, _pending_outbox_events = _sql_projection_state(sql_sessions)
            projection_pending, projection_dead_letters = _projection_effect_state(sql_sessions)
        except Exception:  # noqa: BLE001 - missing schema is not readiness
            postgres_ok = False
    inventory = _contract_inventory()
    return ReadinessDependencies(
        postgres_ok=postgres_ok,
        qdrant_ok=qdrant_ok,
        outbox_lag_seconds=outbox_lag,
        # SQL outbox state proves formal-signal delivery, not Qdrant
        # freshness.  There is no durable index watermark in this schema, so
        # report the derived-index SLO as unknown rather than relabeling an
        # outbox backlog as an index backlog.
        index_lag_events=None,
        index_state="UNKNOWN" if qdrant_ok else "DOWN",
        projection_pending_count=projection_pending,
        projection_dead_letter_count=projection_dead_letters,
        latest_snapshot=snapshot,
        contract_inventory=inventory,
        required_contracts=("content.v1", "content-factor-signal.v5.1"),
    )


def _component(ready: bool, code: str) -> dict[str, object]:
    return {"ready": ready, "code": code if not ready else "READY"}


def _video_components(dependencies: ReadinessDependencies, application: object | None) -> dict[str, dict[str, object]]:
    sessions = getattr(getattr(application, "_tasks", None), "_sessions", None)
    schema = bool(dependencies.postgres_ok and sessions is not None)
    xiaoe_enabled = os.getenv("CONTENT_XIAOE_PAGE_RESOLVER_ENABLED", "").lower() == "true"
    heartbeat_path = os.getenv("CONTENT_VIDEO_WORKER_HEARTBEAT_FILE", "")
    heartbeat_ok, heartbeat_payload = _video_worker_heartbeat(heartbeat_path)
    raw_dir = os.getenv("CONTENT_RAW_STORAGE_DIR", "")
    raw_ok = bool(raw_dir and Path(raw_dir).is_dir() and os.access(raw_dir, os.R_OK | os.W_OK))
    model_ok = bool(os.getenv("CONTENT_MODEL_URL", "") and os.getenv("CONTENT_MODEL_NAME", ""))
    vision_ok = bool(os.getenv("CONTENT_VISION_URL", "") and os.getenv("CONTENT_VISION_MODEL", ""))
    asr_ok = importlib.util.find_spec("faster_whisper") is not None
    # API processes do not import Paddle.  Only the isolated worker's signed
    # runtime observation can prove the requested GPU is available.
    ocr_ok, ocr_payload = _ocr_heartbeat(os.getenv("CONTENT_OCR_HEARTBEAT_FILE", ""))
    queue_ok = bool(
        sessions is not None and callable(getattr(getattr(application, "_tasks", None), "claim_pending", None))
    )
    # The API is deliberately not mounted with browser storage state.  It
    # trusts only the fresh, non-secret video-worker proof and verifies that
    # the worker's opaque credential reference/provider are allowlisted.
    browser_ok = (not xiaoe_enabled) or heartbeat_ok
    retention_status = str(getattr(application, "_retention_status", "RETENTION_DISABLED"))
    retention_ready = retention_status == "READY"
    return {
        "schema": _component(schema, "SCHEMA_NOT_READY"),
        "video_worker_heartbeat": {
            **_component(heartbeat_ok, "VIDEO_WORKER_HEARTBEAT_INVALID"),
            "proof": heartbeat_payload,
        },
        # Media binaries are worker capabilities, not API-container
        # capabilities.  Their tested identity is part of the signed
        # heartbeat above, preventing an API-side false positive.
        "ffmpeg": _component(heartbeat_ok, "FFMPEG_UNAVAILABLE"),
        "ffprobe": _component(heartbeat_ok, "FFPROBE_UNAVAILABLE"),
        "yt_dlp": _component(heartbeat_ok, "YTDLP_UNAVAILABLE"),
        "xiaoe_browser": _component(browser_ok, "XIAOE_BROWSER_NOT_READY"),
        "asr": _component(asr_ok, "ASR_NOT_READY"),
        "ocr_visual": {
            **_component(ocr_ok and vision_ok, "OCR_VISUAL_NOT_READY"),
            "ocr_runtime": ocr_payload,
        },
        "content_model": _component(model_ok, "CONTENT_MODEL_NOT_READY"),
        "raw_storage": _component(raw_ok, "RAW_STORAGE_NOT_READY"),
        "queue_claim": _component(queue_ok, "QUEUE_CLAIM_NOT_READY"),
        "retention": _component(retention_ready, retention_status),
    }


def _bundle_components(
    dependencies: ReadinessDependencies, application: object | None, auth_ready: bool
) -> dict[str, dict[str, object]]:
    service = getattr(application, "_knowledge_bundle_service", None)
    expected = _bundle_contract_checksum()
    contract = Path(__file__).parents[3] / "contracts" / "content-knowledge-bundle.v1.json"
    try:
        checksum_ok = "sha256:" + hashlib.sha256(contract.read_bytes()).hexdigest().upper() == expected
    except OSError:
        checksum_ok = False
    sessions = getattr(getattr(application, "_tasks", None), "_sessions", None)
    repository_ok = bool(
        service is not None and callable(getattr(service, "create", None)) and callable(getattr(service, "get", None))
    )
    # Qdrant is deliberately absent: formal bundles use the SQL authority.
    return {
        "schema": _component(bool(dependencies.postgres_ok and sessions is not None), "SCHEMA_NOT_READY"),
        "sql_snapshot_claim_evidence_lifecycle": _component(
            bool(sessions is not None), "SQL_BUNDLE_AUTHORITY_NOT_READY"
        ),
        "contract_checksum": _component(checksum_ok, "CONTRACT_CHECKSUM_MISMATCH"),
        "bundle_repository": _component(repository_ok, "KNOWLEDGE_BUNDLE_NOT_READY"),
        "service_auth": _component(auth_ready, "AUTH_NOT_READY"),
    }


def _bundle_contract_checksum() -> str:
    """Return the locked Bundle schema checksum, never a readiness-derived value."""
    return "sha256:EBFD13B78622C3846890438A4FB3CB858278F571FDAB247CDD72EF18CA211621"


def _video_worker_heartbeat(value: str) -> tuple[bool, dict[str, object]]:
    """Validate the worker-only, non-secret video capability attestation.

    A stale, malformed, CPU-attesting, or allowlist-mismatched proof fails
    closed.  Deliberately return only diagnostic statuses; the credential
    reference itself is never exposed by the public health endpoint.
    """
    if not value:
        return False, {"health_code": "VIDEO_WORKER_HEARTBEAT_MISSING"}
    try:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        observed = datetime.fromisoformat(str(payload["observed_at"]).replace("Z", "+00:00"))
        age = (datetime.now(UTC) - _as_utc(observed)).total_seconds()
        xiaoe = payload["xiaoe_page"]
        extractor = payload["frame_extractor"]
        ocr = payload["ocr"]
        allowed_refs, allowed_providers = credential_allowlist_from_environment()
        requested = str(ocr.get("requested_device") or "")
        actual = str(ocr.get("actual_device") or "")
        ready = (
            str(payload.get("schema")) == _VIDEO_HEARTBEAT_SCHEMA
            and str(payload.get("profile")) == "video"
            and str(payload.get("health_code")) == "READY"
            and 0 <= age <= float(os.getenv("CONTENT_VIDEO_HEARTBEAT_MAX_AGE_SECONDS", "120"))
            and isinstance(xiaoe, dict)
            and bool(xiaoe.get("enabled"))
            and bool(xiaoe.get("ready"))
            and str(xiaoe.get("credential_ref") or "") in allowed_refs
            and str(xiaoe.get("credential_provider") or "") in allowed_providers
            and str(xiaoe.get("materializer_identity")) == _XIAOE_MATERIALIZER_IDENTITY
            and isinstance(extractor, dict)
            and bool(extractor.get("ready"))
            and str(extractor.get("identity")) == _TARGETED_FRAME_EXTRACTOR_IDENTITY
            and isinstance(ocr, dict)
            and str(ocr.get("health_code")) == "READY"
            and requested == "gpu:0"
            and actual.lower().startswith("gpu:0")
        )
        return ready, {
            "health_code": "READY" if ready else "VIDEO_WORKER_CAPABILITY_MISMATCH",
            "schema": str(payload.get("schema") or ""),
            "ocr_actual_device": actual,
            "frame_extractor": str(extractor.get("identity") or ""),
            "xiaoe_page_resolver": "READY" if bool(xiaoe.get("ready")) else "NOT_READY",
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, {"health_code": "VIDEO_WORKER_HEARTBEAT_INVALID"}


def _ocr_heartbeat(value: str) -> tuple[bool, dict[str, object]]:
    """Read only the worker's non-secret GPU health proof; never import Paddle."""
    if not value:
        return False, {"health_code": "OCR_HEARTBEAT_MISSING"}
    try:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        observed = datetime.fromisoformat(str(payload["observed_at"]).replace("Z", "+00:00"))
        age = (datetime.now(UTC) - _as_utc(observed)).total_seconds()
        identity = payload["runtime_identity"]
        required = {
            "paddle_version",
            "paddleocr_version",
            "cuda_version",
            "cudnn_version",
            "device_count",
            "compiled_cuda",
            "requested_device",
            "actual_device",
        }
        requested = str(payload.get("requested_device") or "")
        actual = str(payload.get("actual_device") or "")
        ready = (
            str(payload.get("profile")) == "ocr"
            and str(payload.get("health_code")) == "READY"
            and 0 <= age <= float(os.getenv("CONTENT_OCR_HEARTBEAT_MAX_AGE_SECONDS", "120"))
            and requested == "gpu:0"
            and actual.lower().startswith("gpu:0")
            and isinstance(identity, dict)
            and required <= set(identity)
            and str(identity.get("requested_device")) == requested
            and str(identity.get("actual_device")) == actual
            and str(identity.get("compiled_cuda")).lower() == "true"
            and int(str(identity.get("device_count"))) >= 1
        )
        safe = {key: identity[key] for key in sorted(required) if key in identity}
        safe.update({"requested_device": requested, "actual_device": actual, "health_code": payload.get("health_code")})
        return ready, safe
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, {"health_code": "OCR_HEARTBEAT_INVALID"}


def _sql_projection_state(session_factory) -> tuple[SnapshotReadiness, float, int]:
    """Read authoritative publication/outbox state from SQL.

    Readiness must not inspect SnapshotService's in-memory implementation
    details. The final count is pending formal-signal outbox work; callers
    must not use it as a Qdrant rebuild backlog because the index adapter has
    no durable cursor to report.
    """
    now = datetime.now(UTC)
    with session_factory() as session:
        latest = session.execute(
            select(ContentSnapshotRow, ContentPublicationRunRow)
            .join(
                ContentPublicationRunRow,
                ContentPublicationRunRow.content_snapshot_id == ContentSnapshotRow.content_snapshot_id,
            )
            .where(ContentPublicationRunRow.state.in_(("READY", "PUBLISHING", "PUBLISHED")))
            .order_by(ContentSnapshotRow.created_at.desc(), ContentPublicationRunRow.updated_at.desc())
            .limit(1)
        ).first()
        pending = list(
            session.scalars(
                select(SignalOutboxRow)
                .where(SignalOutboxRow.status != "PUBLISHED")
                .order_by(SignalOutboxRow.created_at, SignalOutboxRow.outbox_id)
            ).all()
        )
    if latest is None:
        snapshot = SnapshotReadiness(None)
    else:
        snapshot_row, publication_row = latest
        snapshot = SnapshotReadiness(
            snapshot_row.content_snapshot_id,
            str(publication_row.state),
            _as_utc(publication_row.updated_at or snapshot_row.created_at),
        )
    oldest = next((item.created_at for item in pending if item.created_at is not None), None)
    outbox_lag = max(0.0, (now - _as_utc(oldest)).total_seconds()) if oldest else 0.0
    return snapshot, outbox_lag, len(pending)


def _projection_effect_state(session_factory) -> tuple[int, int]:
    """Return optional index-outbox backlog without treating it as SQL publish lag."""
    with session_factory() as session:
        rows = session.scalars(
            select(ContentTaskEffectRow.state).where(ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX")
        ).all()
    return (
        sum(state in {"PENDING", "DISPATCHING"} for state in rows),
        sum(state == "DEAD_LETTER" for state in rows),
    )


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _contract_inventory() -> tuple[str, ...]:
    """Load the local platform manifest for readiness diagnostics."""
    manifest = Path(__file__).resolve().parents[3] / "contracts" / "platform-manifest.yaml"
    try:
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        contracts = payload.get("contracts") or []
        return tuple(sorted(str(item["id"]) for item in contracts if isinstance(item, dict) and item.get("id")))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return ()


def _component_payload(report, name: str) -> dict[str, object]:
    component = getattr(report, name)
    return {
        "component": name,
        "ready": component.ready,
        "degraded": component.degraded,
        "blocking_reasons": list(component.blocking_reasons),
    }


def _status_for(component) -> int:
    return status.HTTP_200_OK if component.ready else status.HTTP_503_SERVICE_UNAVAILABLE


router = create_readiness_router()

__all__ = ["create_readiness_router", "dependencies_from_application", "router"]

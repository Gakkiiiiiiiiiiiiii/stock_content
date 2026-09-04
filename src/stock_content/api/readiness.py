"""Framework adapter for the pure readiness application service."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import yaml
from fastapi import APIRouter, Response, status
from sqlalchemy import select, text

from stock_content.adapters.postgres.models import (
    ContentPublicationRunRow,
    ContentSnapshotRow,
    SignalOutboxRow,
)
from stock_content.adapters.qdrant import NullKnowledgeIndex
from stock_content.application.readiness_service import ReadinessDependencies, ReadinessService, SnapshotReadiness


def create_readiness_router(
    service: ReadinessService | None = None,
    dependencies: Callable[[], ReadinessDependencies] | None = None,
) -> APIRouter:
    readiness_service = service or ReadinessService()
    router = APIRouter(tags=["readiness"])

    @router.get("/readiness")
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
    snapshot, outbox_lag, index_lag = SnapshotReadiness(None), 0.0, 0
    if postgres_ok and sql_sessions:
        try:
            snapshot, outbox_lag, index_lag = _sql_projection_state(sql_sessions)
        except Exception:  # noqa: BLE001 - missing schema is not readiness
            postgres_ok = False
    inventory = _contract_inventory()
    return ReadinessDependencies(
        postgres_ok=postgres_ok,
        qdrant_ok=qdrant_ok,
        outbox_lag_seconds=outbox_lag,
        index_lag_events=index_lag,
        latest_snapshot=snapshot,
        contract_inventory=inventory,
        required_contracts=("content.v1", "content-factor-signal.v5.1"),
    )


def _sql_projection_state(session_factory) -> tuple[SnapshotReadiness, float, int]:
    """Read authoritative publication/outbox state from SQL.

    Readiness must not inspect SnapshotService's in-memory implementation
    details. Unpublished outbox rows are also the conservative pending-index
    count because the index adapter has no durable cursor to report.
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

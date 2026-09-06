"""Readiness semantics for SQL authority and the derived Qdrant index."""
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stock_content.api.readiness import create_readiness_router
from stock_content.application.readiness_service import ReadinessDependencies, ReadinessService, SnapshotReadiness


def _healthy_dependencies(**changes: object) -> ReadinessDependencies:
    values: dict[str, object] = {
        "postgres_ok": True,
        "qdrant_ok": True,
        "index_state": "HEALTHY",
        "index_lag_events": 0,
        "latest_snapshot": SnapshotReadiness("snapshot-1", "READY"),
        "contract_inventory": ("content.v1", "content-factor-signal.v5.1"),
        "required_contracts": ("content.v1", "content-factor-signal.v5.1"),
    }
    values.update(changes)
    return ReadinessDependencies(**values)


def test_postgres_down_fails_closed_for_fact_and_formal_publish() -> None:
    report = ReadinessService().evaluate(_healthy_dependencies(postgres_ok=False))

    assert not report.fact.ready
    assert not report.signal.ready
    assert report.fact.blocking_reasons == ("postgres_unavailable",)
    assert "postgres_unavailable" in report.signal.blocking_reasons
    assert not report.to_dict()["capabilities"]["read_only_facts"]["ready"]
    assert not report.to_dict()["capabilities"]["formal_publish"]["ready"]


def test_qdrant_down_does_not_disguise_sql_fact_loss() -> None:
    report = ReadinessService().evaluate(_healthy_dependencies(qdrant_ok=False))

    assert report.fact.ready and report.signal.ready
    assert not report.search.ready
    assert report.index_state == "DOWN"
    assert report.search.blocking_reasons == ("qdrant_unavailable",)


def test_stale_index_exposes_backlog_slo_without_blocking_formal_publish() -> None:
    report = ReadinessService(max_index_lag_events=10).evaluate(
        _healthy_dependencies(index_state="STALE", index_lag_events=11)
    )

    assert report.fact.ready and report.signal.ready
    assert not report.search.ready
    assert report.search.blocking_reasons == ("index_stale", "index_backlog_exceeded")
    payload = report.to_dict()
    assert payload["index_backlog_slo_events"] == 10
    assert payload["index_lag_events"] == 11
    assert payload["capabilities"]["formal_publish"]["ready"]


def test_rebuilding_index_is_a_degraded_search_capability_with_reason_code() -> None:
    report = ReadinessService(max_index_lag_events=10).evaluate(
        _healthy_dependencies(index_state="REBUILDING", index_lag_events=2)
    )

    assert not report.search.ready and report.search.degraded
    assert report.search.blocking_reasons == ("index_rebuilding",)
    assert report.fact.ready and report.signal.ready


def test_unknown_rebuild_watermark_fails_search_freshness_claim_closed() -> None:
    report = ReadinessService().evaluate(
        _healthy_dependencies(index_state="UNKNOWN", index_lag_events=None)
    )

    assert not report.search.ready
    assert report.search.blocking_reasons == ("index_status_unknown", "index_backlog_unknown")
    assert report.fact.ready and report.signal.ready


def test_formal_publish_blocks_on_stale_outbox_but_read_only_facts_remain_available() -> None:
    report = ReadinessService(max_outbox_lag_seconds=10).evaluate(
        _healthy_dependencies(outbox_lag_seconds=11)
    )

    assert report.fact.ready
    assert not report.signal.ready
    payload = report.to_dict()["capabilities"]
    assert payload["read_only_facts"]["ready"]
    assert not payload["formal_publish"]["ready"]
    assert "outbox_lag_exceeded" in payload["formal_publish"]["blocking_reasons"]


def test_readiness_endpoint_exposes_capabilities_and_reason_codes() -> None:
    app = FastAPI()
    app.include_router(
        create_readiness_router(
            dependencies=lambda: _healthy_dependencies(index_state="REBUILDING", index_lag_events=4)
        )
    )

    response = TestClient(app).get("/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["formal_publish"]["ready"]
    assert not payload["capabilities"]["derived_search"]["ready"]
    assert payload["search"]["blocking_reasons"] == ["index_rebuilding"]


def test_compose_keeps_qdrant_a_started_dependency_while_postgres_is_authoritative() -> None:
    compose_path = Path(__file__).parents[1] / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    dependencies = compose["services"]["stock-content"]["depends_on"]

    assert dependencies["postgres"]["condition"] == "service_healthy"
    assert dependencies["qdrant"]["condition"] == "service_started"

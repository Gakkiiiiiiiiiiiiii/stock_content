from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fastapi.testclient import TestClient

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.knowledge_repository import PostgresKnowledgeRepository
from stock_content.adapters.qdrant.knowledge_index import _projection_payload
from stock_content.api.main import create_app
from stock_content.application.service import ContentApplication
from stock_content.domain.models import KnowledgeUnit
from stock_content.domain.signal_contract import upgrade_signal_v5, validate_signal_v5


def _projection(quality: str) -> dict:
    return {
        "knowledge_uid": "k1",
        "available_from": "2026-06-01T00:00:00+00:00",
        "lifecycle_status": "ACTIVE",
        "attributes": {
            "claim_id": "c1",
            "occurrence_id": "o1",
            "semantic_segment_id": "ss1",
            "asserted_at": "2026-05-01T00:00:00+00:00",
            "source_available_at": "2026-05-02T00:00:00+00:00",
            "source_availability_quality": quality,
            "temporal_bindings": [{
                "temporal_binding_id": "tb1",
                "role": "FORECAST_TARGET",
                "scope": "FORECAST",
                "period_label": "2026Q3",
                "start_date": "2026-07-01",
                "end_date": "2026-09-30",
                "assertion_status": "EXPECTED",
            }],
            "lifecycle_artifact_id": "la1",
            "content_snapshot_id": "s1",
        },
    }


def test_postgres_pit_modes_and_temporal_filters_are_authoritative():
    payload = _projection("PUBLISHED_TIME_PROXY")
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    assert not PostgresKnowledgeRepository._pit_matches(
        payload, {"availability_as_of": as_of, "pit_mode": "PUBLIC_STRICT"}
    )
    assert PostgresKnowledgeRepository._pit_matches(
        payload, {"availability_as_of": as_of, "pit_mode": "PUBLIC_ALLOW_PROXY"}
    )
    assert PostgresKnowledgeRepository._pit_matches(
        payload,
        {
            "availability_as_of": as_of,
            "pit_mode": "PUBLIC_ALLOW_PROXY",
            "temporal_role": "FORECAST_TARGET",
            "target_start": "2026-07-01",
            "target_end": "2026-09-30",
            "semantic_segment_id": "ss1",
        },
    )
    missing_proxy_time = _projection("PUBLISHED_TIME_PROXY")
    missing_proxy_time["attributes"]["source_available_at"] = None
    assert not PostgresKnowledgeRepository._pit_matches(
        missing_proxy_time, {"availability_as_of": as_of, "pit_mode": "PUBLIC_ALLOW_PROXY"}
    )
    assert not PostgresKnowledgeRepository._pit_matches(
        payload, {"availability_as_of": as_of, "pit_mode": "SYSTEM", "target_start": "2027-01-01"}
    )
    open_ended = _projection("EXACT")
    open_ended["attributes"]["temporal_bindings"][0] = {
        "temporal_binding_id": "tb-open", "role": "FORECAST_TARGET", "scope": "OPEN_ENDED",
        "value_type": "DATE", "start_date": "2026-07-01", "end_date": None,
    }
    assert PostgresKnowledgeRepository._pit_matches(
        open_ended, {"temporal_role": "FORECAST_TARGET", "target_start": "2030-01-01"}
    )
    assert not PostgresKnowledgeRepository._pit_matches(
        open_ended, {"temporal_role": "FORECAST_TARGET", "target_end": "2026-06-30"}
    )
    unresolved = _projection("EXACT")
    unresolved["attributes"]["temporal_bindings"][0].update({
        "value_type": "NONE", "start_date": None, "end_date": None,
    })
    assert not PostgresKnowledgeRepository._pit_matches(
        unresolved, {"temporal_role": "FORECAST_TARGET", "target_start": "2026-01-01"}
    )


def test_knowledge_payload_keeps_attributes_and_qdrant_projection_is_complete():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    unit = KnowledgeUnit(
        knowledge_uid="k1", video_id="v1", chapter_id=None, statement="statement",
        available_from=now, as_of=now, attributes=_projection("EXACT")["attributes"],
    )
    row = SimpleNamespace(**unit.__dict__)
    payload = PostgresKnowledgeRepository._payload(row)
    assert payload["attributes"]["semantic_segment_id"] == "ss1"
    projected = _projection_payload(unit)
    assert {"occurrence_id", "claim_id", "semantic_segment_id", "temporal_scope", "primary_temporal_role",
            "period_label", "forecast_start", "forecast_end", "granularity", "assertion_status",
            "available_from", "lifecycle_status", "content_snapshot_id"} <= set(projected)


def test_signal_v5_required_lineage_and_forbidden_trading_fields():
    signal = upgrade_signal_v5(_projection("EXACT"))
    assert signal["signal_schema_version"] == "content-factor-signal.v5"
    validate_signal_v5(signal)
    invalid = dict(signal, order_qty=1)
    try:
        validate_signal_v5(invalid)
    except ValueError as exc:
        assert "order_qty" in str(exc) or "trading" in str(exc)
    else:
        raise AssertionError("trading instruction must be rejected")


def test_signal_v5_projects_full_temporal_model_and_allows_null_timestamps():
    item = _projection("UNKNOWN")
    item["attributes"]["asserted_at"] = None
    item["attributes"]["source_available_at"] = None
    item["attributes"]["temporal_bindings"] = [{
        "temporal_binding_id": "tb-unresolved", "role": "FORECAST_TARGET", "scope": "UNRESOLVED",
        "value_type": "NONE", "precision": "UNKNOWN", "granularity": None,
        "period_label": None, "start_date": None, "end_date": None,
        "assertion_status": "EXPECTED", "reference_snapshot_id": "ref-1",
    }]
    signal = upgrade_signal_v5(item)
    assert signal["asserted_at"] is None
    assert signal["source_available_at"] is None
    assert signal["temporal_bindings"][0] == {
        "temporal_binding_id": "tb-unresolved", "role": "FORECAST_TARGET", "scope": "UNRESOLVED",
        "period_label": None, "start_date": None, "end_date": None, "assertion_status": "EXPECTED",
    }


def test_lifecycle_claim_contradiction_is_not_hidden_by_active_occurrence():
    class Result:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class Session:
        def __init__(self):
            self.calls = 0

        def scalars(self, _statement):
            self.calls += 1
            status = "CONTRADICTED" if self.calls == 1 else "ACTIVE"
            return Result([SimpleNamespace(
                effective_at=datetime(2026, 1, 1, tzinfo=UTC),
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
                lifecycle_event_id=f"event-{self.calls}",
                to_status=status,
            )])

    assert not PostgresKnowledgeRepository._lifecycle_matches(
        Session(), _projection("EXACT"), {}
    )


def test_search_filters_before_final_limit_and_fills_later_public_rows(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'knowledge.db'}")
    database.create_schema()
    repository = PostgresKnowledgeRepository(database.session_factory)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    units = []
    for index, quality in enumerate(("PUBLISHED_TIME_PROXY", "UNKNOWN", "PUBLISHED_TIME_PROXY", "EXACT", "EXACT")):
        units.append(KnowledgeUnit(
            knowledge_uid=f"k-{index}", video_id="v1", chapter_id=None,
            statement="eligible statement", confidence=1 - index / 10,
            as_of=now, available_from=now + timedelta(seconds=index),
            attributes={
                "source_available_at": now.isoformat(),
                "source_availability_quality": quality,
            },
        ))
    repository.replace_for_video("v1", units)
    result = repository.search("eligible", {"pit_mode": "PUBLIC_STRICT"}, 2)
    assert [item["knowledge_uid"] for item in result] == ["k-3", "k-4"]


def test_list_for_video_as_of_applies_public_filter_before_limit(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'video-pit.db'}")
    database.create_schema()
    repository = PostgresKnowledgeRepository(database.session_factory)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    repository.replace_for_video(
        "video-pit",
        [
            KnowledgeUnit(
                knowledge_uid="k-filtered",
                video_id="video-pit",
                chapter_id=None,
                statement="first candidate",
                as_of=now,
                available_from=now,
                attributes={
                    "source_available_at": now.isoformat(),
                    "source_availability_quality": "UNKNOWN",
                },
            ),
            KnowledgeUnit(
                knowledge_uid="k-eligible",
                video_id="video-pit",
                chapter_id=None,
                statement="later eligible candidate",
                as_of=now,
                available_from=now,
                attributes={
                    "source_available_at": now.isoformat(),
                    "source_availability_quality": "EXACT",
                },
            ),
        ],
    )

    result = repository.list_for_video_as_of(
        "video-pit", now, limit=1, availability_mode="PUBLIC_STRICT"
    )
    assert [item["knowledge_uid"] for item in result] == ["k-eligible"]


def test_semantic_candidate_shortfall_fills_from_authoritative_search():
    class CandidateIndex:
        def search(self, query, limit):
            return ["filtered-candidate"]

    class Knowledge:
        def hydrate(self, knowledge_uids, filters):
            return []

        def search(self, query, filters, limit):
            return [
                {"knowledge_uid": "authoritative-later"},
                {"knowledge_uid": "authoritative-later"},
            ]

    application = ContentApplication(
        None,
        None,
        Knowledge(),
        CandidateIndex(),
        None,
    )
    result = application.search_knowledge("revenue", {}, 1)
    assert [item["knowledge_uid"] for item in result] == ["authoritative-later"]


class _Application:
    def search_knowledge(self, query, filters, limit, **kwargs):
        if filters.get("pit_mode") not in {None, "SYSTEM", "PUBLIC_STRICT", "PUBLIC_ALLOW_PROXY"}:
            raise ValueError("unknown pit mode")
        return [{"query": query, "filters": kwargs, "limit": limit}]

    def factor_signals_v5(self, *args, **kwargs):
        return [upgrade_signal_v5(_projection("INGEST_TIME_UPPER_BOUND"))]


def test_retrieval_get_and_signal_v5_endpoint_contracts():
    client = TestClient(create_app(_Application()))
    response = client.get(
        "/api/v1/knowledge/search",
        params={
            "query": "中际旭创", "availability_as_of": "2026-06-30T15:00:00+08:00",
            "target_start": "2026-07-01", "target_end": "2026-09-30",
            "temporal_role": "FORECAST_TARGET", "pit_mode": "PUBLIC_ALLOW_PROXY",
        },
    )
    assert response.status_code == 200
    assert response.json()["filters"]["pit_mode"] == "PUBLIC_ALLOW_PROXY"
    response = client.post(
        "/api/v1/knowledge/search",
        json={
            "query": "中际旭创", "start": "2026-01-01T00:00:00Z", "end": "2026-12-31T00:00:00Z",
            "pit_mode": "PUBLIC_ALLOW_PROXY", "availability_as_of": "2026-06-30T00:00:00Z",
        },
    )
    assert response.status_code == 200
    assert response.json()["filters"]["pit_mode"] == "PUBLIC_ALLOW_PROXY"
    response = client.post(
        "/api/v1/knowledge/search",
        json={"query": "x", "pit_mode": "INVALID"},
    )
    assert response.status_code == 422
    response = client.post(
        "/internal/v1/factor-signals/v5",
        json={"symbols": [], "start": "2026-01-01T00:00:00Z", "end": "2026-12-31T00:00:00Z"},
    )
    assert response.status_code == 200
    assert response.json()["contract_version"] == "content-factor-signal.v5"
    assert response.json()["items"][0]["source_availability_quality"] == "INGEST_TIME_UPPER_BOUND"

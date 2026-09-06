from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from contracts.fixtures import accept_formal_signal
from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentPublicationRunRow, ContentSnapshotRow, SignalOutboxRow
from stock_content.api.readiness import _contract_inventory, _sql_projection_state
from stock_content.application.quality_report import QualityMetrics, evaluate_quality
from stock_content.application.readiness_service import ReadinessDependencies, ReadinessService, SnapshotReadiness
from stock_content.application.retention_service import RetentionService
from stock_content.application.task_lease_service import TaskLeaseService
from stock_content.domain.retention_policy import RetentionPolicy
from stock_content.domain.source_policy import AccessClassification, SourcePolicy, allow_source
from stock_content.domain.task_run import Checkpoint, LeaseError


def test_capability_independent_modules_are_covered_elsewhere():
    policy = SourcePolicy("feed", "public", frozenset({"index"}), "standard", AccessClassification.PUBLIC)
    assert allow_source(policy, "index")


def test_strict_consumer_fixture_rejects_compatibility_payload():
    payload = {
        "contract": "content-factor-signal.v3", "authority": "COMPATIBILITY_READ_ONLY",
        "formal_eligible": False,
    }
    assert not accept_formal_signal(payload)


def test_readiness_search_degrades_without_blocking_authority():
    dependencies = ReadinessDependencies(qdrant_ok=False, latest_snapshot=SnapshotReadiness("s1", "READY"))
    report = ReadinessService().evaluate(dependencies)
    assert report.fact.ready and report.signal.ready
    assert not report.search.ready and report.search.degraded


def test_readiness_signal_accepts_published_and_blocks_stale_outbox():
    dependencies = ReadinessDependencies(
        latest_snapshot=SnapshotReadiness("s1", "PUBLISHED"),
        outbox_lag_seconds=301,
        contract_inventory=("content.v1", "content-factor-signal.v5.1"),
        required_contracts=("content.v1", "content-factor-signal.v5.1"),
    )
    report = ReadinessService().evaluate(dependencies)
    assert not report.signal.ready
    assert report.signal.blocking_reasons == ("outbox_lag_exceeded",)


def test_readiness_contract_inventory_contains_formal_and_compatibility_ids():
    inventory = _contract_inventory()
    assert "content.v1" in inventory
    assert "content-factor-signal.v5.1" in inventory


def test_task_lease_fencing_and_checkpoint():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service = TaskLeaseService()
    task = service.create("rebuild", "run-1")
    leased = service.acquire(task.task_run_id, "worker-a", now=now, ttl=timedelta(seconds=10))
    service.checkpoint(task.task_run_id, "worker-a", leased.fencing_token, Checkpoint("step"), now=now)
    try:
        service.checkpoint(task.task_run_id, "worker-b", leased.fencing_token, Checkpoint("stale"), now=now)
    except LeaseError:
        pass
    else:
        raise AssertionError("stale worker accepted")


def test_retention_tombstone_is_idempotent():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    policy = RetentionPolicy("short", timedelta(days=1))
    service = RetentionService()
    first = service.tombstone("a1", policy, reason="expiry", actor="job", now=now)
    assert service.tombstone("a1", policy, reason="again", actor="job", now=now) == first


def test_quality_semantic_mismatch_blocks():
    report = evaluate_quality(QualityMetrics(1, 1, 1, 1), semantic_mismatch_count=1)
    assert report.gate_result == "BLOCKED"
    assert "semantic_mismatch" in report.blocking_reasons


def test_readiness_uses_sql_publication_and_outbox_state(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'readiness.db'}")
    database.create_schema()
    created = datetime.now(UTC) - timedelta(seconds=20)
    with database.session_factory.begin() as session:
        session.add(ContentSnapshotRow(
            content_snapshot_id="snap-ready",
            source_type="bilibili",
            source_ref="BV-ready",
            source_content_hash="a" * 64,
            created_at=created,
        ))
        session.add(ContentPublicationRunRow(
            publication_run_id="pub-ready",
            content_snapshot_id="snap-ready",
            query_hash="query",
            signal_policy_version="policy.v1",
            state="READY",
            manifest_hash="b" * 64,
            updated_at=created,
        ))
        session.add(SignalOutboxRow(
            outbox_id="outbox-ready",
            signal_id="signal-ready",
            content_snapshot_id="snap-ready",
            claim_id="claim-ready",
            status="PENDING",
            payload={"signal_id": "signal-ready"},
            created_at=created,
        ))
    snapshot, outbox_lag, pending_outbox_events = _sql_projection_state(database.session_factory)
    assert snapshot.snapshot_id == "snap-ready"
    assert snapshot.state == "READY"
    assert outbox_lag >= 10
    # Formal-signal outbox work is not evidence of Qdrant freshness.
    assert pending_outbox_events == 1


def test_compose_uses_internal_service_endpoints_and_heavy_profiles():
    compose = yaml.safe_load((Path(__file__).parents[1] / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert services["stock-content"]["environment"]["CONTENT_DATABASE_URL"].endswith("@postgres:5432/stock_content")
    assert services["stock-content"]["environment"]["CONTENT_QDRANT_URL"] == "http://qdrant:6333"
    assert services["media-worker"]["profiles"] == ["media"]
    assert services["media-worker"]["environment"]["CONTENT_WORKER_PROFILE"] == "media"
    assert services["multimodal-worker"]["profiles"] == ["multimodal"]
    assert services["multimodal-worker"]["environment"]["CONTENT_WORKER_PROFILE"] == "multimodal"

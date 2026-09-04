import pytest
from sqlalchemy import inspect

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.publication_repository import PublicationRepository
from stock_content.adapters.postgres.repositories.signal_outbox_repository import SignalOutboxRepository
from stock_content.application.publication_unit_of_work import (
    InMemoryPublicationRepository,
    PublicationConflictError,
    PublicationUnitOfWork,
)
from stock_content.cli.rebuild_index import rebuild_vector_index
from stock_content.domain.publication_run import PublicationState


def test_publication_is_idempotent_and_rollback_is_invisible():
    repository = InMemoryPublicationRepository()
    uow = PublicationUnitOfWork(repository)
    first = uow.publish(content_snapshot_id="snap", query_hash="q", signal_policy_version="p", manifest={"x": 1})
    assert first.state == PublicationState.READY
    assert (
        uow.publish(content_snapshot_id="snap", query_hash="q", signal_policy_version="p", manifest={"x": 1}) == first
    )
    with pytest.raises(RuntimeError):
        uow.publish(
            content_snapshot_id="snap-2",
            query_hash="q",
            signal_policy_version="p",
            manifest={},
            failure_hook=lambda point: (_ for _ in ()).throw(RuntimeError(point)) if point == "signals" else None,
        )
    assert repository.get_by_identity("snap-2", "q", "p") is None


def test_publication_identity_rejects_different_sealed_content():
    repository = InMemoryPublicationRepository()
    uow = PublicationUnitOfWork(repository)
    uow.publish(content_snapshot_id="snap", query_hash="q", signal_policy_version="p", manifest={"x": 1})
    with pytest.raises(PublicationConflictError, match="different sealed"):
        uow.publish(content_snapshot_id="snap", query_hash="q", signal_policy_version="p", manifest={"x": 2})
    with pytest.raises(PublicationConflictError, match="different sealed"):
        uow.publish(
            content_snapshot_id="snap", query_hash="q", signal_policy_version="p",
            manifest={"x": 1}, signals=[{"signal_id": "signal-1"}],
        )
    with pytest.raises(PublicationConflictError, match="different sealed"):
        uow.publish(
            content_snapshot_id="snap", query_hash="q", signal_policy_version="p",
            manifest={"x": 1, "artifact_membership": {"claims": "claims-v2"}},
        )


@pytest.mark.parametrize("failure_point", ["projecting", "snapshot", "sealing", "signals", "outbox", "ready"])
def test_sql_publication_rolls_back_all_rows_at_each_phase(tmp_path, failure_point):
    database = Database(f"sqlite:///{tmp_path / (failure_point + '.db')}")
    database.create_schema()
    repository = PublicationRepository(database.session_factory)
    outbox = SignalOutboxRepository(database.session_factory)
    payload = {
        "signal_id": "signal-" + failure_point,
        "content_snapshot_id": "snapshot-" + failure_point,
        "claim_id": "claim-1",
        "signal_schema_version": "content-factor-signal.v4",
    }
    uow = PublicationUnitOfWork(
        repository,
        snapshot_writer=lambda session, snapshot_id, manifest: None,
        signal_writer=lambda session, rows: None,
        outbox_writer=lambda session, rows: [outbox.enqueue_in_session(session, row) for row in rows],
    )

    def fail(point):
        if point == failure_point:
            raise RuntimeError("injected-" + point)

    with pytest.raises(RuntimeError, match="injected-" + failure_point):
        uow.publish(
            content_snapshot_id=payload["content_snapshot_id"],
            query_hash="ingest:q",
            signal_policy_version="signal-policy.v1",
            manifest={"artifact_membership": {"claims": "claims-v1"}, "signals": [payload]},
            signals=[payload], outbox_events=[payload], failure_hook=fail,
        )

    assert repository.get_by_identity(payload["content_snapshot_id"], "ingest:q", "signal-policy.v1") is None
    assert outbox.get_by_signal_id(payload["signal_id"]) is None


def test_sql_publication_reloads_sealed_manifest_and_signals(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'sealed.db'}")
    database.create_schema()
    repository = PublicationRepository(database.session_factory)
    outbox = SignalOutboxRepository(database.session_factory)
    payload_z = {
        "signal_id": "z-signal",
        "content_snapshot_id": "snapshot-sealed",
        "claim_id": "a-claim",
        "signal_schema_version": "content-factor-signal.v4",
        "decision_id": "decision-z",
    }
    payload_a = {
        "signal_id": "a-signal",
        "content_snapshot_id": "snapshot-sealed",
        "claim_id": "z-claim",
        "signal_schema_version": "content-factor-signal.v4",
        "decision_id": "decision-a",
    }
    signals = [payload_z, payload_a]
    uow = PublicationUnitOfWork(
        repository,
        snapshot_writer=lambda session, snapshot_id, manifest: None,
        signal_writer=lambda session, rows, *, publication_run_id: repository.save_sealed_signals_in_session(
            session, publication_run_id, tuple(rows)
        ),
        outbox_writer=lambda session, rows: [outbox.enqueue_in_session(session, row) for row in rows],
    )
    manifest = {"artifact_membership": {"claims": "claims-v1"}, "sealed_signals": signals}
    uow.publish(
        content_snapshot_id="snapshot-sealed",
        query_hash="ingest:sealed",
        signal_policy_version="signal-policy.v1",
        manifest=manifest,
        signals=signals, outbox_events=signals,
    )
    # The database reader orders by signal_id (a-signal, z-signal), while the
    # retry caller retains canonical payload order (z-signal, a-signal).
    retry = uow.publish(
        content_snapshot_id="snapshot-sealed",
        query_hash="ingest:sealed",
        signal_policy_version="signal-policy.v1",
        manifest=manifest,
        signals=signals, outbox_events=signals,
    )

    sealed = repository.read_sealed("snapshot-sealed", "ingest:sealed", "signal-policy.v1")
    assert sealed is not None
    assert sealed["manifest"] == manifest
    assert [item["signal_id"] for item in sealed["signals"]] == ["a-signal", "z-signal"]
    assert retry == sealed["publication_run"]
    assert sealed["publication_run"].state == PublicationState.READY


def test_sqlite_legacy_schema_upgrade_adds_sealed_projection_tables(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'legacy-upgrade.db'}")
    from stock_content.adapters.postgres.models import ContentPublicationRunRow

    ContentPublicationRunRow.__table__.create(database.engine)
    database.create_schema()
    with database.engine.connect() as connection:
        table_names = set(inspect(connection).get_table_names())
    assert "content_publication_manifests" in table_names
    assert "content_sealed_signals" in table_names


def test_rebuild_report_filters_inactive_items():
    class Index:
        def __init__(self):
            self.items = None

        def index(self, items):
            self.items = items

    index = Index()
    report = rebuild_vector_index(
        [
            {"knowledge_uid": "a", "statement": "ok", "lifecycle_status": "ACTIVE"},
            {"knowledge_uid": "b", "statement": "old", "lifecycle_status": "WITHDRAWN"},
        ],
        index,
    )
    assert report["indexed_count"] == 1 and report["skipped_inactive"] == 1

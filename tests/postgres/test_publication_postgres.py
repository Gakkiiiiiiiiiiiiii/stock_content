from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func, select

from stock_content.adapters.postgres.models import (
    ContentPublicationManifestRow,
    ContentPublicationRunRow,
    ContentSealedSignalRow,
    SignalOutboxRow,
)
from stock_content.adapters.postgres.repositories.publication_repository import PublicationRepository
from stock_content.adapters.postgres.repositories.signal_outbox_repository import SignalOutboxRepository
from stock_content.application.publication_unit_of_work import PublicationConflictError, PublicationUnitOfWork
from stock_content.domain.publication_run import PublicationState


def _payloads(snapshot_id: str) -> list[dict[str, str]]:
    return [
        {
            "signal_id": "pg-signal-a",
            "content_snapshot_id": snapshot_id,
            "claim_id": "claim-a",
            "signal_schema_version": "content-factor-signal.v4",
        },
        {
            "signal_id": "pg-signal-b",
            "content_snapshot_id": snapshot_id,
            "claim_id": "claim-b",
            "signal_schema_version": "content-factor-signal.v4",
        },
    ]


def _uow(session_factory):
    repository = PublicationRepository(session_factory)
    outbox = SignalOutboxRepository(session_factory)

    def signal_writer(session, rows, *, publication_run_id):
        repository.save_sealed_signals_in_session(session, publication_run_id, tuple(rows))

    return PublicationUnitOfWork(
        repository,
        snapshot_writer=lambda session, snapshot_id, manifest: None,
        signal_writer=signal_writer,
        outbox_writer=lambda session, rows: [outbox.enqueue_in_session(session, row) for row in rows],
    ), repository


def _assert_no_publication_rows(session_factory, snapshot_id: str, query_hash: str):
    with session_factory() as session:
        run = session.scalar(select(ContentPublicationRunRow).where(
            ContentPublicationRunRow.content_snapshot_id == snapshot_id,
            ContentPublicationRunRow.query_hash == query_hash,
        ))
        assert run is None
        assert session.scalar(select(func.count()).select_from(ContentPublicationManifestRow)) == 0
        assert session.scalar(select(func.count()).select_from(ContentSealedSignalRow)) == 0
        assert session.scalar(select(func.count()).select_from(SignalOutboxRow)) == 0


@pytest.mark.parametrize("failure_point", ["projecting", "snapshot", "sealing", "signals", "outbox", "ready"])
def test_postgres_publication_failure_rolls_back_every_phase(postgres_publication_store, failure_point):
    session_factory, _schema = postgres_publication_store
    uow, _repository = _uow(session_factory)
    snapshot_id = "pg-failure-" + failure_point
    query_hash = "pg-query-" + failure_point
    signals = _payloads(snapshot_id)

    def fail(point):
        if point == failure_point:
            raise RuntimeError("injected-" + point)

    with pytest.raises(RuntimeError, match="injected-" + failure_point):
        uow.publish(
            content_snapshot_id=snapshot_id,
            query_hash=query_hash,
            signal_policy_version="signal-policy.v1",
            manifest={"artifact_membership": {"claims": "claims-v1"}, "sealed_signals": signals},
            signals=signals,
            outbox_events=signals,
            failure_hook=fail,
        )

    _assert_no_publication_rows(session_factory, snapshot_id, query_hash)


def test_postgres_publication_concurrent_same_payload_is_one_ready_run(postgres_publication_store):
    session_factory, _schema = postgres_publication_store
    snapshot_id = "pg-concurrent"
    query_hash = "pg-concurrent-query"
    signals = _payloads(snapshot_id)
    manifest = {"artifact_membership": {"claims": "claims-v1"}, "sealed_signals": signals}

    def publish_once(_index):
        uow, _repository = _uow(session_factory)
        return uow.publish(
            content_snapshot_id=snapshot_id,
            query_hash=query_hash,
            signal_policy_version="signal-policy.v1",
            manifest=manifest,
            signals=signals,
            outbox_events=signals,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish_once, range(2)))

    assert all(result.state == PublicationState.READY for result in results)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ContentPublicationRunRow)) == 1
        assert session.scalar(select(func.count()).select_from(ContentPublicationManifestRow)) == 1
        assert session.scalar(select(func.count()).select_from(ContentSealedSignalRow)) == 2
        assert session.scalar(select(func.count()).select_from(SignalOutboxRow)) == 2
    repository = PublicationRepository(session_factory)
    sealed = repository.read_sealed(snapshot_id, query_hash, "signal-policy.v1")
    assert sealed is not None
    assert sealed["manifest"] == manifest
    assert {item["signal_id"] for item in sealed["signals"]} == {"pg-signal-a", "pg-signal-b"}


def test_postgres_publication_identity_conflict_preserves_first_sealed_payload(postgres_publication_store):
    session_factory, _schema = postgres_publication_store
    uow, repository = _uow(session_factory)
    snapshot_id = "pg-conflict"
    query_hash = "pg-conflict-query"
    signals = _payloads(snapshot_id)
    first_manifest = {"artifact_membership": {"claims": "claims-v1"}, "sealed_signals": signals}
    uow.publish(
        content_snapshot_id=snapshot_id,
        query_hash=query_hash,
        signal_policy_version="signal-policy.v1",
        manifest=first_manifest,
        signals=signals,
        outbox_events=signals,
    )

    with pytest.raises(PublicationConflictError):
        uow.publish(
            content_snapshot_id=snapshot_id,
            query_hash=query_hash,
            signal_policy_version="signal-policy.v1",
            manifest={"artifact_membership": {"claims": "claims-v2"}, "sealed_signals": signals},
            signals=signals,
            outbox_events=signals,
        )
    sealed = repository.read_sealed(snapshot_id, query_hash, "signal-policy.v1")
    assert sealed is not None
    assert sealed["manifest"] == first_manifest

"""Regression coverage for the optional, durable Qdrant projection outbox."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from stock_content.adapters.postgres.models import ContentTaskEffectRow
from stock_content.api.dependencies import build_application
from stock_content.api.readiness import dependencies_from_application
from stock_content.application.knowledge_projection_dispatcher import KnowledgeProjectionDispatcher


class _UnavailableIndex:
    def index(self, _units, *, idempotency_key=None):
        raise ConnectionError(f"qdrant unavailable: {idempotency_key}")


class _RecordingIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def index(self, units, *, idempotency_key=None):
        self.calls.append((tuple(unit.knowledge_uid for unit in units), idempotency_key))


class _RemoteSuccessThenAmbiguousFailureIndex(_RecordingIndex):
    """The remote accepts a batch but its acknowledgement is lost."""

    def __init__(self) -> None:
        super().__init__()
        self.point_ids: set[str] = set()
        self._crash_once = True

    def index(self, units, *, idempotency_key=None):
        super().index(units, idempotency_key=idempotency_key)
        self.point_ids.update(unit.knowledge_uid for unit in units)
        if self._crash_once:
            self._crash_once = False
            raise ConnectionError("remote accepted batch but acknowledgement was lost")


def _fixture_options() -> dict:
    as_of = datetime.now(UTC).replace(microsecond=0)
    return {
        "metadata": {"title": "optional qdrant", "duration_seconds": 10},
        "transcript": "宁德时代300750业绩增长。",
        "segments": [{"start_seconds": 0, "end_seconds": 10, "text": "宁德时代300750业绩增长。", "confidence": 1.0}],
        "as_of": as_of.isoformat(),
        "offline_fixture": True,
    }


def test_qdrant_outage_leaves_succeeded_sql_task_and_durable_projection_then_recovers(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    dispatcher = application._knowledge_projection_dispatcher  # noqa: SLF001 - production composition seam
    dispatcher._index = _UnavailableIndex()  # noqa: SLF001 - deterministic outage
    queued = application.enqueue("bilibili", "BV1qdrantoptional", _fixture_options())

    result = application.process_next("ingest-worker")

    assert result and result["status"] == "SUCCEEDED"
    assert application.get_task(queued["task_id"])["status"] == "SUCCEEDED"
    assert application.get_video(result["video_id"])["video_id"] == result["video_id"]
    failed_dispatch = application.dispatch_knowledge_projections("projection-worker")
    assert failed_dispatch == {"dispatched": 0, "retried": 1, "dead_lettered": 0, "pending": 1}

    sessions = application._tasks._sessions  # noqa: SLF001 - inspect durable outbox state
    with sessions.begin() as session:
        effect = session.scalar(
            select(ContentTaskEffectRow).where(
                ContentTaskEffectRow.task_id == queued["task_id"],
                ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX",
            )
        )
        assert effect and effect.state == "PENDING"
        assert effect.projection_payload["knowledge_ids"]
        assert effect.last_error_code == "ConnectionError"
        # Make the deterministic retry due without waiting in a test.
        effect.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)

    recovered = _RecordingIndex()
    dispatcher._index = recovered  # noqa: SLF001 - recovery simulation
    # Simulate a process crash after it obtained a projection lease.  A later
    # worker can take that expired lease, while the old owner cannot receipt.
    assert dispatcher._claim_due("crashed-worker", limit=1, lease_seconds=60)  # noqa: SLF001
    with sessions.begin() as session:
        effect = session.scalar(
            select(ContentTaskEffectRow).where(ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX")
        )
        effect.dispatch_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    claim = dispatcher._claim_due("recovery-worker", limit=1, lease_seconds=60)[0]  # noqa: SLF001
    effect_id = claim.effect_id
    with pytest.raises(RuntimeError, match="LEASE_LOST"):
        dispatcher._complete(effect_id, "crashed-worker")  # noqa: SLF001
    dispatcher._index.index(dispatcher._load_units(effect_id), idempotency_key=effect_id)  # noqa: SLF001
    dispatcher._complete(claim, "recovery-worker")  # noqa: SLF001
    assert len(recovered.calls) == 1
    # A second worker sees the completion receipt and cannot duplicate it.
    assert application.dispatch_knowledge_projections("takeover-worker")["dispatched"] == 0
    assert len(recovered.calls) == 1
    with sessions() as session:
        effect = session.scalar(
            select(ContentTaskEffectRow).where(
                ContentTaskEffectRow.task_id == queued["task_id"],
                ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX",
            )
        )
        assert effect and effect.state == "COMPLETED"


def test_qdrant_degradation_is_separate_from_authoritative_readiness(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'readiness.db'}", enable_qdrant=False)

    report = dependencies_from_application(application)

    assert report.qdrant_ok is False
    assert report.projection_pending_count == 0


def test_dispatch_takeover_after_old_load_makes_old_worker_a_noop_before_index(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'takeover.db'}", enable_qdrant=False)
    queued = application.enqueue("bilibili", "BV1qdranttakeover", _fixture_options())
    assert application.process_next("ingest-worker")
    sessions = application._tasks._sessions  # noqa: SLF001 - durable outbox inspection
    index = _RecordingIndex()
    old = KnowledgeProjectionDispatcher(sessions, index)
    successor = KnowledgeProjectionDispatcher(sessions, index)
    original_load = old._load_units  # noqa: SLF001 - pause exactly after harmless SQL load

    def pause_after_load(effect_id):
        units = original_load(effect_id)
        with sessions.begin() as session:
            effect = session.scalar(
                select(ContentTaskEffectRow).where(
                    ContentTaskEffectRow.task_id == queued["task_id"],
                    ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX",
                )
            )
            assert effect and effect.dispatch_owner == "old-worker"
            effect.dispatch_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        # The successor completes before the old worker reaches the final
        # row-lock/fence at the actual external-effect seam.
        assert successor.dispatch_due("new-worker") == {
            "dispatched": 1, "retried": 0, "dead_lettered": 0, "pending": 0,
        }
        assert len(index.calls) == 1, index.calls
        return units

    old._load_units = pause_after_load  # type: ignore[method-assign]  # noqa: SLF001
    old_result = old.dispatch_due("old-worker")
    assert old_result == {
        "dispatched": 0, "retried": 0, "dead_lettered": 0, "pending": 0,
    }, old_result
    assert len(index.calls) == 1
    with sessions() as session:
        effect = session.scalar(
            select(ContentTaskEffectRow).where(
                ContentTaskEffectRow.task_id == queued["task_id"],
                ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX",
            )
        )
        assert effect and effect.state == "COMPLETED", effect.last_error_code if effect else None
        assert effect.dispatch_owner is None and effect.last_error_code is None


def test_ambiguous_remote_success_retries_the_stable_effect_without_duplicate_points(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'crash-retry.db'}", enable_qdrant=False)
    queued = application.enqueue("bilibili", "BV1qdrantcrash", _fixture_options())
    assert application.process_next("ingest-worker")
    sessions = application._tasks._sessions  # noqa: SLF001 - durable outbox inspection
    index = _RemoteSuccessThenAmbiguousFailureIndex()
    crashed = KnowledgeProjectionDispatcher(sessions, index)

    # A remote success followed by a lost acknowledgement has the same
    # at-least-once transport property as a process death before DB receipt.
    assert crashed.dispatch_due("crashed-worker") == {
        "dispatched": 0, "retried": 1, "dead_lettered": 0, "pending": 1,
    }
    assert len(index.calls) == 1
    first_key = index.calls[0][1]
    with sessions.begin() as session:
        effect = session.scalar(
            select(ContentTaskEffectRow).where(
                ContentTaskEffectRow.task_id == queued["task_id"],
                ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX",
            )
        )
        assert effect and effect.state == "PENDING", effect.last_error_code if effect else None
        effect.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)

    recovered = KnowledgeProjectionDispatcher(sessions, index)
    assert recovered.dispatch_due("recovery-worker") == {
        "dispatched": 1, "retried": 0, "dead_lettered": 0, "pending": 0,
    }
    assert [call[1] for call in index.calls] == [first_key, first_key]
    # Qdrant's deterministic knowledge-unit IDs collapse the at-least-once
    # transport calls into one business projection.
    assert len(index.point_ids) == len(index.calls[0][0])
    with sessions() as session:
        effect = session.scalar(
            select(ContentTaskEffectRow).where(
                ContentTaskEffectRow.task_id == queued["task_id"],
                ContentTaskEffectRow.effect_kind == "KNOWLEDGE_INDEX",
            )
        )
        assert effect and effect.state == "COMPLETED" and effect.completed_at is not None

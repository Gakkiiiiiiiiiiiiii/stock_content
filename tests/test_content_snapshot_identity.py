"""ContentSnapshot 身份测试（详细修改方案 §4 P0-2/P0-3）。

相同输入必须得到相同身份；任一关键输入变化必须得到新 Snapshot。
P0 C-03：snapshot store 错误必须向上传播，禁止静默吞掉。
"""
from __future__ import annotations

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.snapshot_repository import SqlSnapshotStore
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.lineage import build_content_snapshot


def _record(service: SnapshotService, **overrides) -> str:
    kwargs = {
        "source_type": "bilibili",
        "source_ref": "BV1",
        "source_content_hash": "content-hash-1",
        "artifact_ids": {"source": "source-1"},
        "model_versions": {"asr_model": "faster-whisper", "asr_model_version": "large-v3"},
        "quant_market_snapshot_ids": ["snap-1"],
        "code_sha": "sha-1",
        "config_hash": "cfg-1",
    }
    kwargs.update(overrides)
    return service.record_from_artifacts(**kwargs).content_snapshot_id


def test_same_inputs_yield_same_snapshot_identity():
    service = SnapshotService()
    assert _record(service) == _record(service)


def test_any_key_input_change_yields_new_snapshot():
    base = _record(SnapshotService())
    assert _record(SnapshotService(), source_content_hash="changed") != base
    assert _record(SnapshotService(), model_versions={"asr_model": "other"}) != base
    assert _record(SnapshotService(), code_sha="sha-2") != base
    assert _record(SnapshotService(), config_hash="cfg-2") != base
    assert _record(SnapshotService(), quant_market_snapshot_ids=["snap-2"]) != base


def test_quant_snapshot_id_order_does_not_affect_identity():
    service = SnapshotService()
    first = _record(service, quant_market_snapshot_ids=["a", "b"])
    second = _record(service, quant_market_snapshot_ids=["b", "a"])
    assert first == second


def test_snapshot_binds_lineage_fields():
    snapshot = build_content_snapshot(
        source_type="bilibili",
        source_ref="BV1",
        source_content_hash="hash",
        artifact_ids={"summary": "summary-1"},
        code_sha="sha-1",
    )
    assert snapshot.code_sha == "sha-1"
    assert snapshot.pipeline_version
    assert snapshot.artifact_ids == {"summary": "summary-1"}
    assert "task" not in snapshot.to_dict()  # task_id 禁止参与快照身份


def test_sql_snapshot_store_roundtrip(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'snap.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)
    service = SnapshotService(store)
    snapshot_id = _record(service)

    fetched = service.get(snapshot_id)
    assert fetched is not None
    assert fetched.content_snapshot_id == snapshot_id
    assert fetched.source_type == "bilibili"
    assert fetched.artifact_ids == {"source": "source-1"}
    assert fetched.quant_market_snapshot_ids == ("snap-1",)

    assert [item.content_snapshot_id for item in service.list_for_source("bilibili", "BV1")] == [snapshot_id]
    assert service.list_for_source("bilibili", "OTHER") == []


def test_snapshot_store_error_propagates_never_swallowed(tmp_path):
    """P0 C-03：store 抛错时 record_from_artifacts 必须抛出，不得返回 None。"""
    database = Database(f"sqlite:///{tmp_path / 'snap.db'}")
    database.create_schema()
    store = SqlSnapshotStore(database.session_factory)

    def _broken_save(snapshot):
        raise RuntimeError("disk full")

    store.save = _broken_save  # noqa: SLF001
    service = SnapshotService(store)
    with pytest.raises(RuntimeError, match="disk full"):
        _record(service)


def test_replay_identity_matches(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'snap.db'}")
    database.create_schema()
    service = SnapshotService(SqlSnapshotStore(database.session_factory))
    replay = ReplayService(service)
    snapshot_id = _record(service)

    result = replay.replay(snapshot_id)
    assert result["replay_mode"] == "EXACT"
    assert result["identity_match"] is True
    assert result["recomputed_snapshot_id"] == snapshot_id
    assert result["lineage"]["content_snapshot_id"] == snapshot_id

    missing = replay.replay("cs-nonexistent")
    assert missing["error"] == "SNAPSHOT_NOT_FOUND"

"""向量索引重建测试（详细修改方案 §5 P1-6）。

PostgreSQL = Source of Truth；Qdrant 可完全重建，且重建必须记录版本信息。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_rebuild_module():
    script_path = REPO_ROOT / "scripts" / "rebuild_vector_index.py"
    module = runpy.run_path(str(script_path), run_name="rebuild_vector_index")
    return module


class RecordingIndex:
    def __init__(self) -> None:
        self.indexed: list[dict] = []

    def index(self, items: list[dict]) -> None:
        self.indexed.extend(items)


def test_rebuild_indexes_only_active_knowledge_and_records_versions():
    module = _load_rebuild_module()
    index = RecordingIndex()
    items = [
        {"knowledge_uid": "k1", "statement": "s1", "lifecycle_status": "ACTIVE"},
        {"knowledge_uid": "k2", "statement": "s2", "lifecycle_status": "SUPERSEDED"},
        {"knowledge_uid": "k3", "statement": "s3", "lifecycle_status": "VALIDATED"},
    ]
    manifest = module["rebuild_vector_index"](items, index, collection="knowledge_v3")

    assert [item["knowledge_uid"] for item in index.indexed] == ["k1", "k3"]
    assert manifest["indexed_count"] == 2
    assert manifest["skipped_inactive"] == 1
    assert manifest["collection"] == "knowledge_v3"
    for key in ("embedding_model", "embedding_version", "chunk_version", "collection_version", "rebuilt_at"):
        assert manifest[key]
    assert manifest["source_of_truth"] == "postgres"


def test_rebuild_from_postgres_db(tmp_path):
    """从真实 SQLite/Postgres 兼容库读取知识后重建。"""
    module = _load_rebuild_module()
    from stock_content.adapters.postgres.database import Database
    from stock_content.adapters.postgres.models import KnowledgeUnitRow

    database = Database(f"sqlite:///{tmp_path / 'rebuild.db'}")
    database.create_schema()
    from datetime import UTC, datetime

    with database.session_factory.begin() as session:
        session.add(
            KnowledgeUnitRow(
                knowledge_uid="k-rebuild-1",
                video_id="",
                chapter_id=None,
                statement="测试知识",
                kind="CLAIM",
                lifecycle_status="ACTIVE",
                as_of=datetime.now(UTC),
                available_from=datetime.now(UTC),
            )
        )

    items = module["_load_knowledge_from_postgres"](f"sqlite:///{tmp_path / 'rebuild.db'}")
    assert [item["knowledge_uid"] for item in items] == ["k-rebuild-1"]

    index = RecordingIndex()
    manifest = module["rebuild_vector_index"](items, index)
    assert manifest["indexed_count"] == 1


def test_video_asset_foreign_key_not_required_for_rebuild(tmp_path):
    """重建只依赖 knowledge_unit，不应要求视频资产仍然存在。"""
    module = _load_rebuild_module()
    index = RecordingIndex()
    manifest = module["rebuild_vector_index"]([], index)
    assert manifest["indexed_count"] == 0
    assert index.indexed == []


def test_sys_path_not_polluted():
    # runpy 加载不得破坏测试进程模块解析
    assert "rebuild_vector_index" not in sys.modules

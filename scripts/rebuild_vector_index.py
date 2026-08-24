"""Qdrant 向量索引全量重建（详细修改方案 §5 P1-6）。

原则：PostgreSQL = Source of Truth；Qdrant = Search Projection，可完全重建。

用法：
    python scripts/rebuild_vector_index.py --from-postgres --collection knowledge_v3

重建记录 embedding_model / embedding_version / chunk_version / collection_version，
保证索引产物可追溯、可对比。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

DEFAULT_EMBEDDING_MODEL = os.getenv("CONTENT_EMBEDDING_MODEL", "content-embedding.v1")
DEFAULT_EMBEDDING_VERSION = os.getenv("CONTENT_EMBEDDING_VERSION", "1")
DEFAULT_CHUNK_VERSION = os.getenv("CONTENT_CHUNK_VERSION", "chunk.v1")
DEFAULT_COLLECTION_VERSION = "collection.v3"


def rebuild_vector_index(
    knowledge_items: list[dict[str, Any]],
    index: Any,
    *,
    collection: str = "knowledge_v3",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_version: str = DEFAULT_EMBEDDING_VERSION,
    chunk_version: str = DEFAULT_CHUNK_VERSION,
    collection_version: str = DEFAULT_COLLECTION_VERSION,
) -> dict[str, Any]:
    """从 PostgreSQL 知识条目全量重建向量索引，返回重建清单。"""
    active_items = [
        item for item in knowledge_items if str(item.get("lifecycle_status") or "ACTIVE") in {"ACTIVE", "VALIDATED"}
    ]
    index.index(active_items)
    return {
        "collection": collection,
        "collection_version": collection_version,
        "embedding_model": embedding_model,
        "embedding_version": embedding_version,
        "chunk_version": chunk_version,
        "indexed_count": len(active_items),
        "skipped_inactive": len(knowledge_items) - len(active_items),
        "rebuilt_at": datetime.now(UTC).isoformat(),
        "source_of_truth": "postgres",
    }


def _load_knowledge_from_postgres(database_url: str | None) -> list[dict[str, Any]]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from sqlalchemy import select

    from stock_content.adapters.postgres.database import Database
    from stock_content.adapters.postgres.models import KnowledgeUnitRow

    database = Database(database_url)
    database.create_schema()
    with database.session() as session:
        rows = session.scalars(select(KnowledgeUnitRow)).all()
        return [
            {
                "knowledge_uid": row.knowledge_uid,
                "video_id": row.video_id,
                "chapter_id": row.chapter_id,
                "statement": row.statement,
                "kind": row.kind,
                "knowledge_kind": row.knowledge_kind,
                "knowledge_version": row.knowledge_version,
                "subject": row.subject,
                "subject_key": row.subject_key,
                "predicate_key": row.predicate_key,
                "ticker": row.ticker,
                "support_status": row.support_status,
                "truth_status": row.truth_status,
                "review_status": row.review_status,
                "lifecycle_status": row.lifecycle_status,
                "confidence": row.confidence,
                "as_of": row.as_of,
                "available_from": row.available_from,
                "valid_from": row.valid_from,
                "valid_to": row.valid_to,
                "source_statement_hash": row.source_statement_hash,
                "content_hash": row.content_hash,
                "attributes": dict(row.attributes or {}),
                "provenance": dict(row.provenance or {}),
            }
            for row in rows
        ]


class _RebuildTarget:
    """CLI 目标：真实 Qdrant 或 stdout dry-run。"""

    def __init__(self, collection: str, dry_run: bool) -> None:
        self._dry_run = dry_run
        self._collection = collection
        self._inner = None

    def index(self, items: list[dict[str, Any]]) -> None:
        if self._dry_run:
            return
        if self._inner is None:
            from stock_content.adapters.qdrant import QdrantKnowledgeIndex

            self._inner = QdrantKnowledgeIndex(collection=self._collection)
        from stock_content.domain.models import KnowledgeUnit

        units = [
            KnowledgeUnit(
                knowledge_uid=item["knowledge_uid"],
                video_id=str(item.get("video_id") or ""),
                chapter_id=item.get("chapter_id"),
                statement=item["statement"],
                kind=item.get("kind") or "CLAIM",
                knowledge_kind=item.get("knowledge_kind") or "STATE",
                knowledge_version=int(item.get("knowledge_version") or 1),
                subject=item.get("subject"),
                subject_key=item.get("subject_key"),
                predicate_key=item.get("predicate_key"),
                ticker=item.get("ticker"),
                support_status=item.get("support_status") or "SOURCE_SUPPORTED",
                truth_status=item.get("truth_status") or "NOT_CHECKED",
                review_status=item.get("review_status") or "UNREVIEWED",
                confidence=float(item.get("confidence") or 0.6),
                as_of=item.get("as_of") or datetime.now(UTC),
                available_from=item.get("available_from") or datetime.now(UTC),
                valid_from=item.get("valid_from"),
                valid_to=item.get("valid_to"),
                source_statement_hash=item.get("source_statement_hash"),
                content_hash=item.get("content_hash"),
                attributes=dict(item.get("attributes") or {}),
                provenance=dict(item.get("provenance") or {}),
            )
            for item in items
        ]
        self._inner.index(units)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Qdrant knowledge index from PostgreSQL")
    parser.add_argument("--from-postgres", action="store_true", required=True)
    parser.add_argument("--collection", default=os.getenv("CONTENT_QDRANT_COLLECTION", "knowledge_v3"))
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = _load_knowledge_from_postgres(args.database_url)
    manifest = rebuild_vector_index(items, _RebuildTarget(args.collection, args.dry_run), collection=args.collection)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

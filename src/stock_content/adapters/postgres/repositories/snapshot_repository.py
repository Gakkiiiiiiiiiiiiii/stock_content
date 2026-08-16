"""ContentSnapshot 持久化（详细修改方案 §4 P0-2）。

PostgreSQL 保持权威状态；identity 列保存完整身份 payload 以便原样还原。
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import ContentSnapshotRow
from stock_content.domain.lineage import ContentSnapshot


def _row_to_snapshot(row: ContentSnapshotRow) -> ContentSnapshot:
    identity = dict(row.identity or {})
    created_at = identity.get("created_at")
    if isinstance(created_at, str):
        parsed = datetime.fromisoformat(created_at)
        created_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    if not isinstance(created_at, datetime):
        created_at = row.created_at
    return ContentSnapshot(
        content_snapshot_id=row.content_snapshot_id,
        source_type=row.source_type,
        source_ref=row.source_ref,
        source_content_hash=row.source_content_hash,
        parser_version=identity.get("parser_version"),
        asr_model=identity.get("asr_model"),
        asr_model_version=identity.get("asr_model_version"),
        vision_model=identity.get("vision_model"),
        llm_model=identity.get("llm_model"),
        prompt_bundle_version=str(identity.get("prompt_bundle_version") or "prompt_bundle.v1"),
        entity_alias_version=str(identity.get("entity_alias_version") or "entity_alias.v1"),
        verification_policy_version=str(identity.get("verification_policy_version") or "verification_policy.v1"),
        quant_market_snapshot_ids=tuple(identity.get("quant_market_snapshot_ids") or []),
        code_sha=identity.get("code_sha") or "",
        config_hash=identity.get("config_hash") or "",
        pipeline_version=identity.get("pipeline_version") or row.pipeline_version,
        schema_version=identity.get("schema_version") or row.schema_version,
        artifact_ids=dict(identity.get("artifact_ids") or {}),
        created_at=created_at or row.created_at,
    )


class SqlSnapshotStore:
    """SnapshotStore 的 SQL 实现（SQLite / PostgreSQL 通用）。"""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def save(self, snapshot: ContentSnapshot) -> ContentSnapshot:
        payload = snapshot.to_dict()
        # identity 存入 JSON 列：时间统一 ISO8601，保证可序列化且 canonical。
        if isinstance(payload.get("created_at"), datetime):
            payload["created_at"] = payload["created_at"].isoformat()
        with self._sessions.begin() as session:
            row = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
            if row is None:
                session.add(
                    ContentSnapshotRow(
                        content_snapshot_id=snapshot.content_snapshot_id,
                        source_type=snapshot.source_type,
                        source_ref=snapshot.source_ref,
                        source_content_hash=snapshot.source_content_hash,
                        identity=payload,
                        artifact_ids=dict(snapshot.artifact_ids),
                        quant_market_snapshot_ids=list(snapshot.quant_market_snapshot_ids),
                        pipeline_version=snapshot.pipeline_version,
                        schema_version=snapshot.schema_version,
                        code_sha=snapshot.code_sha,
                        config_hash=snapshot.config_hash,
                        created_at=snapshot.created_at,
                    )
                )
            else:
                row.identity = payload
                row.artifact_ids = dict(snapshot.artifact_ids)
        return snapshot

    def get(self, content_snapshot_id: str) -> ContentSnapshot | None:
        with self._sessions() as session:
            row = session.get(ContentSnapshotRow, content_snapshot_id)
            return _row_to_snapshot(row) if row else None

    def list_for_source(self, source_type: str, source_ref: str) -> list[ContentSnapshot]:
        with self._sessions() as session:
            rows = session.scalars(
                select(ContentSnapshotRow)
                .where(
                    ContentSnapshotRow.source_type == source_type,
                    ContentSnapshotRow.source_ref == source_ref,
                )
                .order_by(ContentSnapshotRow.created_at)
            ).all()
            return [_row_to_snapshot(row) for row in rows]


__all__ = ["SqlSnapshotStore"]

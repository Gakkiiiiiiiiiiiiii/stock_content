"""ContentSnapshot 服务（详细修改方案 §4 P0-2）。

负责快照的构建、持久化与查询。PostgreSQL 保持权威状态；
同一组身份输入必须得到同一 content_snapshot_id（幂等）。
"""
from __future__ import annotations

from typing import Any, Protocol

from stock_content.domain.lineage import (
    ContentSnapshot,
    build_content_snapshot,
    compute_config_hash,
    default_code_sha,
    lineage_of,
)


class SnapshotIntegrityError(ValueError):
    """A snapshot id was reused for a different immutable payload."""


class SnapshotStore(Protocol):
    def save(self, snapshot: ContentSnapshot) -> ContentSnapshot: ...

    def get(self, content_snapshot_id: str) -> ContentSnapshot | None: ...

    def list_for_source(self, source_type: str, source_ref: str) -> list[ContentSnapshot]: ...


class InMemorySnapshotStore:
    """测试与单机部署使用的内存存储。"""

    def __init__(self) -> None:
        self._snapshots: dict[str, ContentSnapshot] = {}

    def save(self, snapshot: ContentSnapshot) -> ContentSnapshot:
        existing = self._snapshots.get(snapshot.content_snapshot_id)
        if existing is not None:
            left = existing.to_dict()
            right = snapshot.to_dict()
            left.pop("created_at", None)
            right.pop("created_at", None)
            if left != right:
                raise SnapshotIntegrityError(
                    f"snapshot id {snapshot.content_snapshot_id} already stores a different payload"
                )
            return existing
        self._snapshots[snapshot.content_snapshot_id] = snapshot
        return snapshot

    def get(self, content_snapshot_id: str) -> ContentSnapshot | None:
        return self._snapshots.get(content_snapshot_id)

    def list_for_source(self, source_type: str, source_ref: str) -> list[ContentSnapshot]:
        items = [
            snapshot
            for snapshot in self._snapshots.values()
            if snapshot.source_type == source_type and snapshot.source_ref == source_ref
        ]
        return sorted(items, key=lambda item: item.created_at)


class SnapshotService:
    def __init__(self, store: SnapshotStore | None = None) -> None:
        self._store = store or InMemorySnapshotStore()

    def record_from_artifacts(
        self,
        *,
        source_type: str,
        source_ref: str,
        source_content_hash: str,
        artifact_ids: dict[str, str] | None = None,
        model_versions: dict[str, Any] | None = None,
        producer_manifest: dict[str, Any] | None = None,
        prompt_versions: dict[str, str] | None = None,
        configuration: dict[str, Any] | None = None,
        external_snapshots: list[str] | tuple[str, ...] = (),
        policy_versions: dict[str, str] | None = None,
        source_artifact_id: str | None = None,
        quant_market_snapshot_ids: list[str] | tuple[str, ...] = (),
        code_sha: str | None = None,
        config_hash: str = "",
        snapshot_kind: str = "INITIAL",
        parent_snapshot_id: str | None = None,
        supersedes_snapshot_id: str | None = None,
        pipeline_version: str = "pipeline.v3",
    ) -> ContentSnapshot:
        models = model_versions or {}
        # Keep the redundant top-level fields and nested producer manifest
        # aligned for callers that use SnapshotService directly (the pipeline
        # stage computes these same effective values before this boundary).
        normalized_manifest = dict(producer_manifest or {})
        effective_code_sha = str(code_sha or normalized_manifest.get("code_sha") or default_code_sha())
        normalized_manifest["code_sha"] = effective_code_sha
        configs = dict(normalized_manifest.get("configs") or {})
        effective_config_hash = str(
            config_hash or configs.get("config_hash") or compute_config_hash(configuration)
        )
        configs["config_hash"] = effective_config_hash
        normalized_manifest["configs"] = configs
        snapshot = build_content_snapshot(
            source_type=source_type,
            source_ref=source_ref,
            source_content_hash=source_content_hash,
            artifact_ids=artifact_ids,
            source_artifact_id=source_artifact_id or (artifact_ids or {}).get("source", ""),
            parser_version=models.get("parser_version"),
            asr_model=models.get("asr_model"),
            asr_model_version=models.get("asr_model_version"),
            vision_model=models.get("vision_model"),
            llm_model=models.get("llm_model"),
            prompt_bundle_version=str(models.get("prompt_bundle_version") or "prompt_bundle.v1"),
            entity_alias_version=str(models.get("entity_alias_version") or "entity_alias.v1"),
            verification_policy_version=str(models.get("verification_policy_version") or "verification_policy.v1"),
            quant_market_snapshot_ids=quant_market_snapshot_ids,
            code_sha=effective_code_sha,
            config_hash=effective_config_hash,
            producer_manifest=normalized_manifest,
            prompt_versions=prompt_versions,
            configuration=configuration,
            external_snapshots=external_snapshots,
            policy_versions=policy_versions,
            model_versions=models,
            snapshot_kind=snapshot_kind,
            parent_snapshot_id=parent_snapshot_id,
            supersedes_snapshot_id=supersedes_snapshot_id,
            pipeline_version=pipeline_version,
        )
        return self._store.save(snapshot)

    def get(self, content_snapshot_id: str) -> ContentSnapshot | None:
        return self._store.get(content_snapshot_id)

    def list_for_source(self, source_type: str, source_ref: str) -> list[ContentSnapshot]:
        return self._store.list_for_source(source_type, source_ref)

    def lineage(self, content_snapshot_id: str) -> dict[str, Any] | None:
        snapshot = self._store.get(content_snapshot_id)
        return lineage_of(snapshot).to_dict() if snapshot else None


__all__ = ["InMemorySnapshotStore", "SnapshotIntegrityError", "SnapshotService", "SnapshotStore"]

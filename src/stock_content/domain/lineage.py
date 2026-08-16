"""ContentSnapshot / ContentLineage（详细修改方案 §4 P0-2）。

ContentSnapshot 身份 = 源内容 + pipeline 版本 + 模型 + Prompt + code SHA + config + Quant 快照。
task_id 禁止参与快照哈希（同一内容不同任务必须得到同一快照身份）。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from stock_content.domain.artifacts import canonical_json

PIPELINE_VERSION = "pipeline.v2"
CONTENT_SNAPSHOT_SCHEMA_VERSION = "content.snapshot.v1"


@dataclass(frozen=True)
class ContentSnapshot:
    content_snapshot_id: str
    source_type: str
    source_ref: str
    source_content_hash: str

    parser_version: str | None = None
    asr_model: str | None = None
    asr_model_version: str | None = None
    vision_model: str | None = None
    llm_model: str | None = None

    prompt_bundle_version: str = "prompt_bundle.v1"
    entity_alias_version: str = "entity_alias.v1"
    verification_policy_version: str = "verification_policy.v1"

    quant_market_snapshot_ids: tuple[str, ...] = ()

    code_sha: str = ""
    config_hash: str = ""
    pipeline_version: str = PIPELINE_VERSION
    schema_version: str = CONTENT_SNAPSHOT_SCHEMA_VERSION

    artifact_ids: dict[str, str] = field(default_factory=dict)

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quant_market_snapshot_ids"] = list(self.quant_market_snapshot_ids)
        return payload


def default_code_sha() -> str:
    return os.getenv("CONTENT_GIT_COMMIT", "unknown")


def compute_config_hash(config: dict[str, Any] | None) -> str:
    if not config:
        return ""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def snapshot_identity_payload(
    *,
    source_content_hash: str,
    pipeline_version: str = PIPELINE_VERSION,
    parser_version: str | None = None,
    asr_model: str | None = None,
    asr_model_version: str | None = None,
    vision_model: str | None = None,
    llm_model: str | None = None,
    prompt_bundle_version: str = "prompt_bundle.v1",
    entity_alias_version: str = "entity_alias.v1",
    verification_policy_version: str = "verification_policy.v1",
    quant_market_snapshot_ids: list[str] | tuple[str, ...] = (),
    code_sha: str = "",
    config_hash: str = "",
) -> dict[str, Any]:
    """参与 content_snapshot_id 的全部身份字段（canonical、无随机字段）。"""
    return {
        "source_content_hash": source_content_hash,
        "pipeline_version": pipeline_version,
        "parser_version": parser_version,
        "asr_model": asr_model,
        "asr_model_version": asr_model_version,
        "vision_model": vision_model,
        "llm_model": llm_model,
        "prompt_bundle_version": prompt_bundle_version,
        "entity_alias_version": entity_alias_version,
        "verification_policy_version": verification_policy_version,
        "quant_market_snapshot_ids": sorted(set(quant_market_snapshot_ids)),
        "code_sha": code_sha,
        "config_hash": config_hash,
    }


def compute_content_snapshot_id(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def build_content_snapshot(
    *,
    source_type: str,
    source_ref: str,
    source_content_hash: str,
    artifact_ids: dict[str, str] | None = None,
    parser_version: str | None = None,
    asr_model: str | None = None,
    asr_model_version: str | None = None,
    vision_model: str | None = None,
    llm_model: str | None = None,
    prompt_bundle_version: str = "prompt_bundle.v1",
    entity_alias_version: str = "entity_alias.v1",
    verification_policy_version: str = "verification_policy.v1",
    quant_market_snapshot_ids: list[str] | tuple[str, ...] = (),
    code_sha: str | None = None,
    config_hash: str = "",
    pipeline_version: str = PIPELINE_VERSION,
) -> ContentSnapshot:
    identity = snapshot_identity_payload(
        source_content_hash=source_content_hash,
        pipeline_version=pipeline_version,
        parser_version=parser_version,
        asr_model=asr_model,
        asr_model_version=asr_model_version,
        vision_model=vision_model,
        llm_model=llm_model,
        prompt_bundle_version=prompt_bundle_version,
        entity_alias_version=entity_alias_version,
        verification_policy_version=verification_policy_version,
        quant_market_snapshot_ids=quant_market_snapshot_ids,
        code_sha=code_sha if code_sha is not None else default_code_sha(),
        config_hash=config_hash,
    )
    snapshot_id = f"cs-{compute_content_snapshot_id(identity)[:32]}"
    return ContentSnapshot(
        content_snapshot_id=snapshot_id,
        source_type=source_type,
        source_ref=source_ref,
        source_content_hash=source_content_hash,
        parser_version=parser_version,
        asr_model=asr_model,
        asr_model_version=asr_model_version,
        vision_model=vision_model,
        llm_model=llm_model,
        prompt_bundle_version=prompt_bundle_version,
        entity_alias_version=entity_alias_version,
        verification_policy_version=verification_policy_version,
        quant_market_snapshot_ids=tuple(sorted(set(quant_market_snapshot_ids))),
        code_sha=identity["code_sha"],
        config_hash=config_hash,
        pipeline_version=pipeline_version,
        artifact_ids=dict(artifact_ids or {}),
    )


@dataclass(frozen=True)
class ContentLineage:
    """ContentSnapshot 的血缘视图：输入 -> 产物 -> 外部引用。"""

    content_snapshot_id: str
    source_type: str
    source_ref: str
    artifact_ids: dict[str, str]
    quant_market_snapshot_ids: tuple[str, ...] = ()
    code_sha: str = ""
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quant_market_snapshot_ids"] = list(self.quant_market_snapshot_ids)
        return payload


def lineage_of(snapshot: ContentSnapshot) -> ContentLineage:
    return ContentLineage(
        content_snapshot_id=snapshot.content_snapshot_id,
        source_type=snapshot.source_type,
        source_ref=snapshot.source_ref,
        artifact_ids=dict(snapshot.artifact_ids),
        quant_market_snapshot_ids=snapshot.quant_market_snapshot_ids,
        code_sha=snapshot.code_sha,
        pipeline_version=snapshot.pipeline_version,
    )


__all__ = [
    "CONTENT_SNAPSHOT_SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "ContentLineage",
    "ContentSnapshot",
    "build_content_snapshot",
    "compute_config_hash",
    "compute_content_snapshot_id",
    "default_code_sha",
    "lineage_of",
    "snapshot_identity_payload",
]

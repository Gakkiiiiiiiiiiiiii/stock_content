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

PIPELINE_VERSION = "pipeline.v3"
CONTENT_SNAPSHOT_SCHEMA_VERSION = "content.snapshot.v2"


@dataclass(frozen=True)
class ContentSnapshot:
    content_snapshot_id: str
    source_type: str
    source_ref: str
    source_content_hash: str
    source_artifact_id: str = ""
    artifact_root_hash: str = ""

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
    snapshot_kind: str = "INITIAL"
    parent_snapshot_id: str | None = None
    supersedes_snapshot_id: str | None = None
    producer_manifest: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    external_snapshots: tuple[str, ...] = ()
    policy_versions: dict[str, str] = field(default_factory=dict)

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quant_market_snapshot_ids"] = list(self.quant_market_snapshot_ids)
        return payload

    @property
    def kind(self) -> str:
        """Legacy read alias; ``snapshot_kind`` is the sole stored field."""
        return self.snapshot_kind


def default_code_sha() -> str:
    value = os.getenv("CONTENT_GIT_COMMIT", "unknown")
    # CONTENT_ENV is the preferred deployment switch, but an orchestrator
    # may expose only the conventional ENVIRONMENT variable.  Treat an empty
    # CONTENT_ENV as unset so production cannot accidentally use ``unknown``.
    environment = (os.getenv("CONTENT_ENV") or os.getenv("ENVIRONMENT") or "development").lower()
    if environment in {"production", "prod"} and value in {"", "unknown", "none", "null"}:
        raise ValueError("CONTENT_GIT_COMMIT must be set in production")
    return value


def compute_artifact_root_hash(artifact_ids: dict[str, str] | None) -> str:
    """Stable Merkle-root seed for the complete artifact registry.

    Artifact IDs are already content addressed; sorting the slot mapping makes
    the root independent of stage execution order while remaining sensitive to
    any artifact replacement/addition.
    """
    return hashlib.sha256(canonical_json(dict(sorted((artifact_ids or {}).items()))).encode("utf-8")).hexdigest()


def compute_config_hash(config: dict[str, Any] | None) -> str:
    if not config:
        return ""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _is_present(value: Any) -> bool:
    """Return whether a provenance value is explicitly usable.

    Empty strings are the legacy/default representation for an omitted value.
    Other scalar values are converted to strings at this boundary so the
    persisted manifest and redundant snapshot columns cannot diverge merely
    because a caller supplied a non-string scalar.
    """

    return value is not None and value != ""


def normalize_producer_manifest(
    producer_manifest: dict[str, Any] | None,
    *,
    code_sha: str | None = None,
    config_hash: str | None = None,
    configuration: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Normalize redundant release/config provenance to one authoritative set.

    An explicit top-level value wins over a conflicting nested manifest value.
    When the top-level value is omitted, the nested value is used, followed by
    the normal deployment/configuration defaults.  The returned manifest is a
    deep-enough copy for the mutable nested ``configs`` object and always
    carries the effective values used by the snapshot identity.
    """

    if producer_manifest is None:
        manifest: dict[str, Any] = {}
    elif isinstance(producer_manifest, dict):
        manifest = dict(producer_manifest)
    else:
        raise ValueError("producer_manifest must be an object")

    manifest_code_sha = manifest.get("code_sha")
    effective_code_sha = (
        code_sha
        if _is_present(code_sha)
        else manifest_code_sha
        if _is_present(manifest_code_sha)
        else default_code_sha()
    )
    effective_code_sha = str(effective_code_sha)

    raw_configs = manifest.get("configs")
    if raw_configs is None:
        configs: dict[str, Any] = {}
    elif isinstance(raw_configs, dict):
        configs = dict(raw_configs)
    else:
        raise ValueError("producer_manifest.configs must be an object")
    manifest_config_hash = configs.get("config_hash")
    effective_config_hash = (
        config_hash
        if _is_present(config_hash)
        else manifest_config_hash
        if _is_present(manifest_config_hash)
        else compute_config_hash(configuration)
    )
    effective_config_hash = str(effective_config_hash)

    manifest["code_sha"] = effective_code_sha
    configs["config_hash"] = effective_config_hash
    manifest["configs"] = configs
    return effective_code_sha, effective_config_hash, manifest


def snapshot_identity_payload(
    *,
    source_content_hash: str,
    source_artifact_id: str = "",
    artifact_root_hash: str = "",
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
    producer_manifest: dict[str, Any] | None = None,
    policy_versions: dict[str, str] | None = None,
    model_versions: dict[str, str] | None = None,
    prompt_versions: dict[str, str] | None = None,
    configuration: dict[str, Any] | None = None,
    external_snapshots: list[str] | tuple[str, ...] = (),
    snapshot_kind: str = "INITIAL",
    parent_snapshot_id: str | None = None,
    supersedes_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """参与 content_snapshot_id 的全部身份字段（canonical、无随机字段）。"""
    return {
        "source_content_hash": source_content_hash,
        "source_artifact_id": source_artifact_id,
        "artifact_root_hash": artifact_root_hash,
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
        "producer_manifest": dict(producer_manifest or {}),
        "policy_versions": dict(policy_versions or {}),
        "model_versions": dict(model_versions or {}),
        "prompt_versions": dict(prompt_versions or {}),
        "configuration": dict(configuration or {}),
        "external_snapshots": sorted(set(external_snapshots)),
        "snapshot_kind": snapshot_kind,
        "parent_snapshot_id": parent_snapshot_id,
        "supersedes_snapshot_id": supersedes_snapshot_id,
    }


def compute_content_snapshot_id(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def build_content_snapshot(
    *,
    source_type: str,
    source_ref: str,
    source_content_hash: str,
    source_artifact_id: str = "",
    artifact_root_hash: str | None = None,
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
    kind: str = "INITIAL",
    snapshot_kind: str | None = None,
    parent_snapshot_id: str | None = None,
    supersedes_snapshot_id: str | None = None,
    producer_manifest: dict[str, Any] | None = None,
    policy_versions: dict[str, str] | None = None,
    model_versions: dict[str, str] | None = None,
    prompt_versions: dict[str, str] | None = None,
    configuration: dict[str, Any] | None = None,
    external_snapshots: list[str] | tuple[str, ...] = (),
) -> ContentSnapshot:
    normalized_artifact_ids = dict(artifact_ids or {})
    computed_root_hash = compute_artifact_root_hash(normalized_artifact_ids)
    if artifact_root_hash and normalized_artifact_ids and artifact_root_hash != computed_root_hash:
        raise ValueError("artifact_root_hash does not match artifact_ids")
    root_hash = artifact_root_hash or computed_root_hash
    effective_code_sha, effective_config_hash, normalized_manifest = normalize_producer_manifest(
        producer_manifest,
        code_sha=code_sha,
        config_hash=config_hash,
        configuration=configuration,
    )
    identity = snapshot_identity_payload(
        source_content_hash=source_content_hash,
        source_artifact_id=source_artifact_id or normalized_artifact_ids.get("source", ""),
        artifact_root_hash=root_hash,
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
        code_sha=effective_code_sha,
        config_hash=effective_config_hash,
        producer_manifest=normalized_manifest,
        policy_versions=policy_versions,
        model_versions=model_versions,
        prompt_versions=prompt_versions,
        configuration=configuration,
        external_snapshots=external_snapshots,
        snapshot_kind=snapshot_kind or kind,
        parent_snapshot_id=parent_snapshot_id,
        supersedes_snapshot_id=supersedes_snapshot_id,
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
        config_hash=identity["config_hash"],
        pipeline_version=pipeline_version,
        artifact_ids=dict(artifact_ids or {}),
        source_artifact_id=source_artifact_id or normalized_artifact_ids.get("source", ""),
        artifact_root_hash=root_hash,
        snapshot_kind=snapshot_kind or kind,
        parent_snapshot_id=parent_snapshot_id,
        supersedes_snapshot_id=supersedes_snapshot_id,
        producer_manifest=normalized_manifest,
        model_versions=dict(model_versions or {}),
        prompt_versions=dict(prompt_versions or {}),
        configuration=dict(configuration or {}),
        external_snapshots=tuple(sorted(set(external_snapshots))),
        policy_versions=dict(policy_versions or {}),
    )


@dataclass(frozen=True)
class ContentLineage:
    """ContentSnapshot 的血缘视图：输入 -> 产物 -> 外部引用。"""

    content_snapshot_id: str
    source_type: str
    source_ref: str
    artifact_ids: dict[str, str]
    source_artifact_id: str = ""
    artifact_root_hash: str = ""
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
        source_artifact_id=snapshot.source_artifact_id,
        artifact_root_hash=snapshot.artifact_root_hash,
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
    "compute_artifact_root_hash",
    "compute_content_snapshot_id",
    "default_code_sha",
    "lineage_of",
    "normalize_producer_manifest",
    "snapshot_identity_payload",
]

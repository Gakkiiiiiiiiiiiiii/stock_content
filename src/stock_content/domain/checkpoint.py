"""Checkpoint v2：Artifact Checkpoint（详细修改方案 §4 P0-4）。

Checkpoint 从“进度记录”升级为“Artifact Checkpoint”：记录每个 Stage 的
输入/输出 artifact 与哈希，支持断点恢复前的完整性校验。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

CHECKPOINT_SCHEMA_VERSION = "checkpoint.v2"

CHECKPOINT_STATUSES = ("SUCCEEDED", "FAILED", "SKIPPED")


@dataclass(frozen=True)
class CheckpointRecord:
    stage: str
    stage_version: str = "1.0.0"
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    output_hashes: tuple[str, ...] = ()
    # Checkpoints retain only public, content-addressed recovery facts.
    # Runtime locators and credentials must never become recovery state.
    input_hashes: tuple[str, ...] = ()
    public_materialization_hash: str = ""
    model_identity: dict[str, str] | None = None
    prompt_identity: dict[str, str] | None = None
    worker_id: str | None = None
    fencing_token: int | None = None
    state_checksum: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: str = "SUCCEEDED"
    retry_count: int = 0
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_artifact_ids"] = list(self.input_artifact_ids)
        payload["output_artifact_ids"] = list(self.output_artifact_ids)
        payload["output_hashes"] = list(self.output_hashes)
        payload["input_hashes"] = list(self.input_hashes)
        # canonical：时间统一 ISO8601，保证可直接 JSON 持久化。
        for key in ("started_at", "finished_at"):
            if isinstance(payload.get(key), datetime):
                payload[key] = payload[key].isoformat()
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "CheckpointRecord":
        def _parse_time(value: Any) -> datetime | None:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                parsed = datetime.fromisoformat(value)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            return None

        return CheckpointRecord(
            stage=str(payload.get("stage") or ""),
            stage_version=str(payload.get("stage_version") or "1.0.0"),
            input_artifact_ids=tuple(payload.get("input_artifact_ids") or ()),
            output_artifact_ids=tuple(payload.get("output_artifact_ids") or ()),
            output_hashes=tuple(payload.get("output_hashes") or ()),
            input_hashes=tuple(payload.get("input_hashes") or ()),
            public_materialization_hash=str(payload.get("public_materialization_hash") or ""),
            model_identity=_identity_mapping(payload.get("model_identity")),
            prompt_identity=_identity_mapping(payload.get("prompt_identity")),
            worker_id=(str(payload["worker_id"]) if payload.get("worker_id") is not None else None),
            fencing_token=(int(payload["fencing_token"]) if payload.get("fencing_token") is not None else None),
            state_checksum=str(payload.get("state_checksum") or ""),
            started_at=_parse_time(payload.get("started_at")),
            finished_at=_parse_time(payload.get("finished_at")),
            status=str(payload.get("status") or "SUCCEEDED"),
            retry_count=int(payload.get("retry_count") or 0),
            schema_version=str(payload.get("schema_version") or CHECKPOINT_SCHEMA_VERSION),
            error=payload.get("error"),
        )


def build_checkpoint(
    *,
    stage: str,
    stage_version: str = "1.0.0",
    input_artifact_ids: list[str] | tuple[str, ...] = (),
    input_hashes: list[str] | tuple[str, ...] = (),
    output_artifacts: list[Any] | tuple[Any, ...] = (),
    public_materialization_hash: str = "",
    model_identity: dict[str, str] | None = None,
    prompt_identity: dict[str, str] | None = None,
    worker_id: str | None = None,
    fencing_token: int | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    status: str = "SUCCEEDED",
    retry_count: int = 0,
    error: str | None = None,
) -> CheckpointRecord:
    """从 Artifact 对象构建 CheckpointRecord（哈希来自 artifact.content_hash）。"""
    if status not in CHECKPOINT_STATUSES:
        raise ValueError(f"invalid checkpoint status: {status}")
    output_ids: list[str] = []
    output_hashes: list[str] = []
    for artifact in output_artifacts:
        output_ids.append(str(getattr(artifact, "artifact_id", "") or ""))
        output_hashes.append(str(getattr(artifact, "content_hash", "") or ""))
    record = CheckpointRecord(
        stage=stage,
        stage_version=stage_version,
        input_artifact_ids=tuple(input_artifact_ids),
        output_artifact_ids=tuple(output_ids),
        output_hashes=tuple(output_hashes),
        input_hashes=tuple(input_hashes),
        public_materialization_hash=public_materialization_hash,
        model_identity=_identity_mapping(model_identity),
        prompt_identity=_identity_mapping(prompt_identity),
        worker_id=worker_id,
        fencing_token=fencing_token,
        started_at=started_at,
        finished_at=finished_at or datetime.now(UTC),
        status=status,
        retry_count=retry_count,
        error=error,
    )
    return CheckpointRecord(**{**record.__dict__, "state_checksum": checkpoint_state_checksum(record)})


def _identity_mapping(value: Any) -> dict[str, str] | None:
    if not value:
        return None
    if not isinstance(value, dict):
        raise ValueError("checkpoint identity must be a mapping")
    return {str(key): str(item) for key, item in sorted(value.items())}


def checkpoint_state_checksum(record: CheckpointRecord) -> str:
    """Hash deterministic recovery facts, excluding clock and lease ownership."""
    payload = {
        "stage": record.stage,
        "stage_version": record.stage_version,
        "input_artifact_ids": list(record.input_artifact_ids),
        "input_hashes": list(record.input_hashes),
        "output_artifact_ids": list(record.output_artifact_ids),
        "output_hashes": list(record.output_hashes),
        "public_materialization_hash": record.public_materialization_hash,
        "model_identity": record.model_identity or {},
        "prompt_identity": record.prompt_identity or {},
        "status": record.status,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CheckpointValidationError(Exception):
    """断点恢复前 artifact 完整性校验失败。"""


def validate_resume(
    checkpoints: list[CheckpointRecord],
    artifacts_by_id: dict[str, Any],
    *,
    stage_versions: dict[str, str] | None = None,
) -> list[str]:
    """校验断点可恢复：artifact 哈希一致且 stage 版本兼容。

    返回可恢复的已完成 stage 名称列表；任何校验失败抛出 CheckpointValidationError。
    """
    completed: list[str] = []
    for record in checkpoints:
        if record.status != "SUCCEEDED":
            break
        for artifact_id, expected_hash in zip(record.output_artifact_ids, record.output_hashes):
            artifact = artifacts_by_id.get(artifact_id)
            if artifact is None:
                raise CheckpointValidationError(f"missing artifact for stage {record.stage}: {artifact_id}")
            actual = str(getattr(artifact, "content_hash", "") or "")
            if expected_hash != actual:
                raise CheckpointValidationError(
                    f"artifact hash mismatch for stage {record.stage}: {artifact_id}"
                )
        if record.input_hashes:
            if len(record.input_artifact_ids) != len(record.input_hashes):
                raise CheckpointValidationError(f"input hash count mismatch for stage {record.stage}")
            for artifact_id, expected_hash in zip(record.input_artifact_ids, record.input_hashes):
                artifact = artifacts_by_id.get(artifact_id)
                if artifact is None:
                    raise CheckpointValidationError(f"missing input artifact for stage {record.stage}: {artifact_id}")
                if str(getattr(artifact, "content_hash", "") or "") != expected_hash:
                    raise CheckpointValidationError(
                        f"input artifact hash mismatch for stage {record.stage}: {artifact_id}"
                    )
        if record.state_checksum and record.state_checksum != checkpoint_state_checksum(record):
            raise CheckpointValidationError(f"checkpoint state checksum mismatch for stage {record.stage}")
        expected_version = (stage_versions or {}).get(record.stage)
        if expected_version and record.stage_version and expected_version != record.stage_version:
            raise CheckpointValidationError(
                f"stage version incompatible for {record.stage}: "
                f"checkpoint={record.stage_version} current={expected_version}"
            )
        completed.append(record.stage)
    return completed


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STATUSES",
    "CheckpointRecord",
    "CheckpointValidationError",
    "build_checkpoint",
    "checkpoint_state_checksum",
    "validate_resume",
]

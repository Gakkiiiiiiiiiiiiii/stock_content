"""Checkpoint v2：Artifact Checkpoint（详细修改方案 §4 P0-4）。

Checkpoint 从“进度记录”升级为“Artifact Checkpoint”：记录每个 Stage 的
输入/输出 artifact 与哈希，支持断点恢复前的完整性校验。
"""
from __future__ import annotations

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
    output_artifacts: list[Any] | tuple[Any, ...] = (),
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
    return CheckpointRecord(
        stage=stage,
        stage_version=stage_version,
        input_artifact_ids=tuple(input_artifact_ids),
        output_artifact_ids=tuple(output_ids),
        output_hashes=tuple(output_hashes),
        started_at=started_at,
        finished_at=finished_at or datetime.now(UTC),
        status=status,
        retry_count=retry_count,
        error=error,
    )


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
            if expected_hash and actual and actual != expected_hash:
                raise CheckpointValidationError(
                    f"artifact hash mismatch for stage {record.stage}: {artifact_id}"
                )
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
    "validate_resume",
]

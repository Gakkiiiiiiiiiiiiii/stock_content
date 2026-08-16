"""StageRunner：为既有 Stage 包装 Artifact Checkpoint v2（详细修改方案 §4 P0-4）。

不改变 Stage 执行语义；仅在 Stage 执行前后记录输入/输出 Artifact 与哈希，
并在失败时写入 FAILED checkpoint，供断点恢复判定。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from stock_content.application.pipeline import PipelineContext
from stock_content.domain.checkpoint import CheckpointRecord, build_checkpoint


class StageRunner:
    def __init__(self, stage: Any, stage_version: str = "1.0.0") -> None:
        self._stage = stage
        self._stage_version = stage_version

    @property
    def name(self) -> str:
        return str(self._stage.name)

    @property
    def stage_version(self) -> str:
        return self._stage_version

    def execute(self, context: PipelineContext) -> PipelineContext:
        registry = context.artifacts
        before = {artifact.artifact_id for artifact in registry.artifacts()}
        input_artifact_ids = [artifact.artifact_id for artifact in registry.artifacts()]
        started_at = datetime.now(UTC)
        try:
            context = self._stage.execute(context)
        except Exception as exc:  # noqa: BLE001 - checkpoint 需要记录失败后继续抛出
            context.checkpoints.append(
                build_checkpoint(
                    stage=self.name,
                    stage_version=self._stage_version,
                    input_artifact_ids=input_artifact_ids,
                    started_at=started_at,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        outputs = [
            artifact for artifact in context.artifacts.artifacts() if artifact.artifact_id not in before
        ]
        context.checkpoints.append(
            build_checkpoint(
                stage=self.name,
                stage_version=self._stage_version,
                input_artifact_ids=input_artifact_ids,
                output_artifacts=outputs,
                started_at=started_at,
                status="SUCCEEDED",
            )
        )
        return context


def wrap_all(stages: list[Any], stage_versions: dict[str, str] | None = None) -> list[StageRunner]:
    versions = stage_versions or {}
    return [StageRunner(stage, versions.get(str(stage.name), "1.0.0")) for stage in stages]


def checkpoint_payload(context: PipelineContext, legacy_output_keys: list[str]) -> dict[str, Any]:
    """on_checkpoint 回调 payload：兼容旧 output_keys + 附加 checkpoint v2。"""
    payload: dict[str, Any] = {"completed": True, "output_keys": legacy_output_keys}
    if context.checkpoints:
        payload["checkpoint"] = context.checkpoints[-1].to_dict()
    return payload


def records_from_checkpoint_state(state: dict[str, Any] | list[Any] | None) -> list[CheckpointRecord]:
    """从任务 checkpoint 存储还原 CheckpointRecord 列表。"""
    if not state:
        return []
    items = state.get("records") if isinstance(state, dict) else state
    records: list[CheckpointRecord] = []
    for item in items or []:
        if isinstance(item, dict) and item.get("stage"):
            records.append(CheckpointRecord.from_dict(item))
    return records


__all__ = ["StageRunner", "checkpoint_payload", "records_from_checkpoint_state", "wrap_all"]

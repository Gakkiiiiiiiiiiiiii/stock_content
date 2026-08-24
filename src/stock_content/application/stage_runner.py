"""StageRunner：为既有 Stage 包装 Artifact Checkpoint v2（详细修改方案 §4 P0-4）。

不改变 Stage 执行语义；仅在 Stage 执行前后记录输入/输出 Artifact 与哈希，
并在失败时写入 FAILED checkpoint，供断点恢复判定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from stock_content.application.pipeline import PipelineContext
from stock_content.domain.checkpoint import CheckpointRecord, build_checkpoint


@dataclass(frozen=True)
class StageContract:
    name: str
    version: str = "1.0.0"
    required_inputs: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    optional_output_types: tuple[str, ...] = ()


@dataclass
class StageResult:
    """Explicit stage boundary result for production stages."""

    produced_artifacts: tuple[Any, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    context: PipelineContext | None = None


class StageRunner:
    def __init__(
        self,
        stage: Any,
        stage_version: str = "1.0.0",
        contract: StageContract | None = None,
        artifact_repository: Any | None = None,
        legacy_fallback: bool = True,
    ) -> None:
        self._stage = stage
        self._stage_version = stage_version
        self.contract = contract or StageContract(
            name=str(stage.name),
            version=stage_version,
            required_inputs=tuple(getattr(stage, "required_inputs", ())),
            output_types=tuple(getattr(stage, "output_types", ())),
            optional_output_types=tuple(getattr(stage, "optional_output_types", ())),
        )
        self._artifact_repository = artifact_repository
        self._legacy_fallback = legacy_fallback

    @property
    def name(self) -> str:
        return str(self._stage.name)

    @property
    def stage_version(self) -> str:
        return self._stage_version

    def execute(self, context: PipelineContext) -> PipelineContext:
        self._validate_inputs(context)
        registry = context.artifacts
        before = (
            {artifact.artifact_id for artifact in registry.artifacts()}
            if self._legacy_fallback
            else set()
        )
        input_artifact_ids = [artifact.artifact_id for artifact in registry.artifacts()]
        started_at = datetime.now(UTC)
        try:
            result = self._stage.execute(context)
            explicit_result = isinstance(result, StageResult)
            if explicit_result:
                if result.context is None:
                    result.context = context
                context = result.context
                outputs = list(result.produced_artifacts)
                self._validate_outputs(outputs)
            else:
                if not self._legacy_fallback:
                    raise TypeError(
                        f"production stage {self.name} must return StageResult"
                    )
                context = result
                outputs = []
        except Exception as exc:  # noqa: BLE001 - checkpoint 需要记录失败后继续抛出
            failed_checkpoint = build_checkpoint(
                    stage=self.name,
                    stage_version=self._stage_version,
                    input_artifact_ids=input_artifact_ids,
                    started_at=started_at,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                )
            context.checkpoints.append(failed_checkpoint)
            if self._artifact_repository and hasattr(self._artifact_repository, "put_with_checkpoint"):
                self._artifact_repository.put_with_checkpoint((), context.task_id, failed_checkpoint)
            raise
        if not explicit_result and not outputs:
            # Legacy stage adapter only. Production StageResult must declare
            # produced_artifacts explicitly.
            outputs = [
                artifact for artifact in context.artifacts.artifacts() if artifact.artifact_id not in before
            ]
        checkpoint = build_checkpoint(
                stage=self.name,
                stage_version=self._stage_version,
                input_artifact_ids=input_artifact_ids,
                output_artifacts=outputs,
                started_at=started_at,
                status="SUCCEEDED",
            )
        context.checkpoints.append(checkpoint)
        if self._artifact_repository and hasattr(self._artifact_repository, "put_with_checkpoint"):
            self._artifact_repository.put_with_checkpoint(outputs, context.task_id, checkpoint)
        return context

    def _validate_inputs(self, context: PipelineContext) -> None:
        missing = [slot for slot in self.contract.required_inputs if context.artifacts.get(slot) is None]
        if missing:
            raise ValueError(f"stage {self.name} missing required artifacts: {missing}")

    def _validate_outputs(self, outputs: list[Any]) -> None:
        actual = {str(getattr(item, "artifact_type", "")) for item in outputs}
        declared = set(self.contract.output_types)
        undeclared = sorted(actual - declared)
        if undeclared:
            raise ValueError(f"stage {self.name} produced undeclared outputs: {undeclared}")
        missing = [
            expected
            for expected in self.contract.output_types
            if expected not in actual and expected not in self.contract.optional_output_types
        ]
        if missing:
            raise ValueError(f"stage {self.name} missing declared outputs: {missing}")


def wrap_all(
    stages: list[Any],
    stage_versions: dict[str, str] | None = None,
    artifact_repository: Any | None = None,
    legacy_fallback: bool = True,
) -> list[StageRunner]:
    versions = stage_versions or {}
    return [
        StageRunner(
            stage,
            versions.get(str(stage.name), "1.0.0"),
            StageContract(
                name=str(stage.name),
                version=versions.get(str(stage.name), "1.0.0"),
                required_inputs=tuple(getattr(stage, "required_inputs", ())),
                output_types=tuple(getattr(stage, "output_types", ())),
                optional_output_types=tuple(getattr(stage, "optional_output_types", ())),
            ),
            artifact_repository=artifact_repository,
            legacy_fallback=legacy_fallback,
        )
        for stage in stages
    ]


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


__all__ = [
    "StageContract",
    "StageResult",
    "StageRunner",
    "checkpoint_payload",
    "records_from_checkpoint_state",
    "wrap_all",
]

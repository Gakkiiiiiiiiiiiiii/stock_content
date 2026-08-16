"""The composable ingestion pipeline boundary for migrated content stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from stock_content.domain.artifacts import ArtifactRegistry
from stock_content.domain.checkpoint import CheckpointRecord, validate_resume


@dataclass
class PipelineContext:
    task_id: str
    source: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    # deprecated：正式业务对象请经 artifacts 传递；Stage 迁移完成后删除。
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: ArtifactRegistry = field(default_factory=ArtifactRegistry)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    current_stage: str = "queued"


class PipelineStage(Protocol):
    name: str

    def execute(self, context: PipelineContext) -> PipelineContext: ...


class ContentPipeline:
    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def process(
        self,
        context: PipelineContext,
        on_progress: Callable[[str, int], None] | None = None,
        on_checkpoint: Callable[[str, dict[str, Any], int], None] | None = None,
        resume: bool = False,
    ) -> PipelineContext:
        total = max(len(self._stages), 1)
        completed: set[str] = set()
        if resume and context.checkpoints:
            # 断点恢复前提：artifact 哈希校验通过且 stage 版本兼容。
            artifacts_by_id = {
                artifact.artifact_id: artifact for artifact in context.artifacts.artifacts()
            }
            completed = set(validate_resume(context.checkpoints, artifacts_by_id))
        for index, stage in enumerate(self._stages):
            context.current_stage = stage.name
            if stage.name in completed:
                continue
            if on_progress:
                on_progress(stage.name, int(index * 100 / total))
            context = stage.execute(context)
            if on_checkpoint:
                payload: dict[str, Any] = {
                    "completed": True,
                    "output_keys": sorted(context.data.keys()),
                }
                if context.checkpoints:
                    payload["checkpoint"] = context.checkpoints[-1].to_dict()
                on_checkpoint(stage.name, payload, int((index + 1) * 100 / total))
        context.current_stage = "completed"
        if on_progress:
            on_progress("completed", 100)
        return context

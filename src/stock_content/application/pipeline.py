"""The composable ingestion pipeline boundary for migrated content stages."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PipelineContext:
    task_id: str
    source: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
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
    ) -> PipelineContext:
        total = max(len(self._stages), 1)
        for index, stage in enumerate(self._stages):
            context.current_stage = stage.name
            if on_progress:
                on_progress(stage.name, int(index * 100 / total))
            context = stage.execute(context)
        context.current_stage = "completed"
        if on_progress:
            on_progress("completed", 100)
        return context

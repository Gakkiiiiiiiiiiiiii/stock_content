"""The composable ingestion pipeline boundary for migrated content stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PipelineContext:
    task_id: str
    source: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    def execute(self, context: PipelineContext) -> PipelineContext: ...


class ContentPipeline:
    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def process(self, context: PipelineContext) -> PipelineContext:
        for stage in self._stages:
            context = stage.execute(context)
        return context

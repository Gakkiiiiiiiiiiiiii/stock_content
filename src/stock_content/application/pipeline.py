"""The composable ingestion pipeline boundary for migrated content stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from stock_content.domain.artifacts import ArtifactRegistry
from stock_content.domain.checkpoint import CheckpointRecord, validate_resume

CORE_METRIC_KEYS = (
    "semantic_segments_per_video",
    "semantic_segment_duration_p50",
    "semantic_segment_duration_p95",
    "segmentation_repair_rate",
    "segmentation_failure_rate",
    "claims_per_semantic_segment",
    "zero_claim_segment_ratio",
    "claim_grounding_reject_rate",
    "temporal_expression_grounding_reject_rate",
    "temporal_binding_count",
    "temporal_normalization_success_rate",
    "temporal_normalization_partial_rate",
    "temporal_normalization_unresolved_rate",
    "temporal_partial_rate",
    "temporal_unresolved_rate",
    "temporal_role_distribution",
    "unresolved_expression_collision_rate",
    "forecast_target_missing_rate",
    "fiscal_period_unresolved_rate",
    "market_session_unresolved_rate",
    "metric_temporal_nature_unknown_rate",
    "planned_vs_actual_ratio",
    "occurrences_per_claim",
    "lifecycle_transition_count",
    "correction_count",
    "lifecycle_correction_count",
    "dependency_availability_delay_ms",
    "cross_source_claim_ratio",
    "pit_query_mismatch_count",
)


def _default_metrics() -> dict[str, float]:
    return {key: 0.0 for key in CORE_METRIC_KEYS}


@dataclass
class RuntimeWorkspace:
    """Ephemeral handles/paths only; never a business-data transport."""

    work_dir: Path | None = None
    video_path: Path | None = None
    audio_path: Path | None = None
    metrics: dict[str, float] = field(default_factory=_default_metrics)


@dataclass
class PipelineState:
    """Explicit typed state used by production stages."""

    metadata: dict[str, Any] = field(default_factory=dict)
    segments: list[Any] = field(default_factory=list)
    transcript: str = ""
    frames: list[Any] = field(default_factory=list)
    frame_insights: list[dict[str, Any]] = field(default_factory=list)
    ocr_evidence: list[dict[str, Any]] = field(default_factory=list)
    multimodal_context: dict[str, Any] = field(default_factory=dict)
    temporal_windows: list[dict[str, Any]] = field(default_factory=list)
    semantic_segments: list[Any] = field(default_factory=list)
    semantic_contexts: list[Any] = field(default_factory=list)
    claim_drafts: list[Any] = field(default_factory=list)
    grounded_occurrences: list[Any] = field(default_factory=list)
    temporal_bindings: list[Any] = field(default_factory=list)
    temporal_bindings_by_draft: dict[int, list[Any]] = field(default_factory=dict)
    occurrences: list[Any] = field(default_factory=list)
    lifecycle_events: list[Any] = field(default_factory=list)
    chapters: list[Any] = field(default_factory=list)
    video: Any = None
    knowledge: list[Any] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    claims: list[Any] = field(default_factory=list)
    financial_numeric_facts: list[dict[str, Any]] = field(default_factory=list)
    financial_events: list[dict[str, Any]] = field(default_factory=list)
    summary: Any = None
    content_snapshot_id: str | None = None
    diarization_status: str = "UNKNOWN"
    quality_warnings: list[str] = field(default_factory=list)
    claims_persisted: bool = False

    # Compatibility methods make legacy custom stages explicit adapters while
    # production code uses named fields above.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        value = getattr(self, key)
        if value is None:
            setattr(self, key, default)
            return default
        return value

    def keys(self) -> list[str]:
        return list(self.__dataclass_fields__)


@dataclass(init=False)
class PipelineContext:
    task_id: str
    source: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    state: PipelineState = field(default_factory=PipelineState)
    artifacts: ArtifactRegistry = field(default_factory=ArtifactRegistry)
    # Immutable artifacts restored from durable checkpoints, including
    # superseded versions that no longer occupy the registry's active slot.
    # Keeping this separate lets resume validation check every checkpoint
    # output without making a stage consume stale slot values.
    restored_artifacts: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    runtime: RuntimeWorkspace = field(default_factory=RuntimeWorkspace)
    trace: dict[str, str] = field(default_factory=dict)
    current_stage: str = "queued"

    def __init__(
        self,
        task_id: str,
        source: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        *,
        source_request: dict[str, Any] | None = None,
        state: PipelineState | None = None,
        data: dict[str, Any] | PipelineState | None = None,
        artifacts: ArtifactRegistry | None = None,
        restored_artifacts: dict[str, Any] | None = None,
        checkpoints: list[CheckpointRecord] | None = None,
        runtime: RuntimeWorkspace | None = None,
        trace: dict[str, str] | None = None,
        current_stage: str = "queued",
    ) -> None:
        self.task_id = task_id
        self.source = dict(source_request or source or {})
        self.options = dict(options or {})
        self.state = state or PipelineState()
        self._legacy_data: dict[str, Any] = {}
        if data is not None:
            if isinstance(data, PipelineState):
                self.state = data
            else:
                self._legacy_data.update(data)
                for key, value in data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
        self.artifacts = artifacts or ArtifactRegistry()
        self.restored_artifacts = dict(restored_artifacts or {})
        self.checkpoints = list(checkpoints or [])
        self.runtime = runtime or RuntimeWorkspace()
        self.trace = dict(trace or {})
        self.current_stage = current_stage

    @property
    def source_request(self) -> dict[str, Any]:
        return self.source

    @property
    def data(self) -> dict[str, Any]:
        """Deprecated test adapter; production stages must use ``state``."""
        for key in self.state.keys():
            value = getattr(self.state, key)
            if value not in (None, "", [], {}):
                self._legacy_data[key] = value
        return self._legacy_data


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
            artifacts_by_id.update(context.restored_artifacts)
            validated = validate_resume(context.checkpoints, artifacts_by_id)
            # A checkpoint from an older semantic-major pipeline may contain
            # a later legacy stage (for example ``knowledge``) but no record
            # for the newly inserted semantic stages.  Only the faithful
            # contiguous prefix of the current stage graph is skippable;
            # otherwise a later checkpoint could bypass required new work.
            completed = set()
            for stage in self._stages:
                if stage.name in validated:
                    completed.add(stage.name)
                else:
                    break
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
                    "output_keys": sorted(context.state.keys()),
                }
                if context.checkpoints:
                    payload["checkpoint"] = context.checkpoints[-1].to_dict()
                on_checkpoint(stage.name, payload, int((index + 1) * 100 / total))
        context.current_stage = "completed"
        if on_progress:
            on_progress("completed", 100)
        return context

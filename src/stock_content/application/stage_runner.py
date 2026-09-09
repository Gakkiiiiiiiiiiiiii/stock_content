"""StageRunner：为既有 Stage 包装 Artifact Checkpoint v2（详细修改方案 §4 P0-4）。

不改变 Stage 执行语义；仅在 Stage 执行前后记录输入/输出 Artifact 与哈希，
并在失败时写入 FAILED checkpoint，供断点恢复判定。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from stock_content.application.pipeline import PipelineContext
from stock_content.domain.artifacts import canonical_json
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
        before = {artifact.artifact_id for artifact in registry.artifacts()} if self._legacy_fallback else set()
        input_artifact_ids = [artifact.artifact_id for artifact in registry.artifacts()]
        input_hashes = [str(artifact.content_hash) for artifact in registry.artifacts()]
        checkpoint_identity = _checkpoint_identity(context)
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
                    raise TypeError(f"production stage {self.name} must return StageResult")
                context = result
                outputs = []
        except Exception as exc:  # noqa: BLE001 - checkpoint 需要记录失败后继续抛出
            failed_checkpoint = build_checkpoint(
                stage=self.name,
                stage_version=self._stage_version,
                input_artifact_ids=input_artifact_ids,
                input_hashes=input_hashes,
                **checkpoint_identity,
                started_at=started_at,
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            context.checkpoints.append(failed_checkpoint)
            self._persist_checkpoint((), context, failed_checkpoint)
            raise
        if not explicit_result and not outputs:
            # Legacy stage adapter only. Production StageResult must declare
            # produced_artifacts explicitly.
            outputs = [artifact for artifact in context.artifacts.artifacts() if artifact.artifact_id not in before]
        # Some stages discover their immutable execution identity while doing
        # the work (the isolated OCR worker is the important example).  A
        # successful checkpoint must seal that observed identity, rather than
        # the pre-execution view used for a FAILED attempt.  This leaves
        # ordinary stages unchanged while making a completed OCR result
        # inseparable from the runtime that produced it.
        checkpoint_identity = _checkpoint_identity(context)
        checkpoint = build_checkpoint(
            stage=self.name,
            stage_version=self._stage_version,
            input_artifact_ids=input_artifact_ids,
            input_hashes=input_hashes,
            **checkpoint_identity,
            output_artifacts=outputs,
            started_at=started_at,
            status="SUCCEEDED",
        )
        context.checkpoints.append(checkpoint)
        self._persist_checkpoint(outputs, context, checkpoint)
        return context

    def _persist_checkpoint(self, outputs: list[Any] | tuple[Any, ...], context: PipelineContext, checkpoint) -> None:
        if self._artifact_repository is None:
            return
        if (
            context.worker_id
            and context.fencing_token is not None
            and hasattr(self._artifact_repository, "put_with_fenced_checkpoint")
        ):
            self._artifact_repository.put_with_fenced_checkpoint(
                outputs, context.task_id, checkpoint, context.worker_id, context.fencing_token
            )
        elif hasattr(self._artifact_repository, "put_with_checkpoint"):
            self._artifact_repository.put_with_checkpoint(outputs, context.task_id, checkpoint)

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


def _checkpoint_identity(context: PipelineContext) -> dict[str, Any]:
    """Extract the safe, deterministic checkpoint manifest from pipeline state."""
    source = context.artifacts.source
    public_materialization = {
        "source_type": str(getattr(source, "source_type", "") or context.source.get("type") or ""),
        "source_ref": str(getattr(source, "source_ref", "") or context.source.get("ref") or ""),
        "source_identity_hash": str(getattr(source, "source_identity_hash", "") or ""),
        "source_version_id": str(getattr(source, "source_version_id", "") or ""),
    }
    public_hash = hashlib.sha256(canonical_json(public_materialization).encode("utf-8")).hexdigest()
    config = dict(context.options.get("pipeline_config") or {})
    # Semantic-context planning is deliberately text-first.  Its checkpoint
    # is made before targeted frame extraction, so it must not become coupled
    # to a future OCR/Vision runtime merely because the application happens to
    # have those adapters configured.  A post-visual semantic context (if a
    # graph explicitly introduces one) has visual artifacts in its inputs and
    # retains the full visual identity below.
    text_only_semantic_context = (
        context.current_stage == "semantic_context"
        and not (context.artifacts.frames or context.artifacts.ocr or context.artifacts.vision)
    )
    visual_model_keys = {
        "knowledge_evidence_window_planner",
        "knowledge_frame_planner",
        "transcript_visual_crosscheck",
        "ocr_engine",
        "ocr_engine_version",
        "ocr_requested_device",
        "vision",
        "vision_version",
    }
    model_identity = {
        key: str(value)
        for key, value in {
            "asr": context.options.get("asr_model") or "faster-whisper",
            "asr_version": context.options.get("asr_model_version") or "1.0",
            "segmentation": context.options.get("segmentation_model") or config.get("segmentation_model") or "",
            "extraction": context.options.get("extraction_model") or config.get("extraction_model") or "",
            "knowledge_evidence_window_planner": config.get("knowledge_evidence_window_planner_version") or "",
            "knowledge_frame_planner": config.get("knowledge_frame_planner_version") or "",
            "transcript_visual_crosscheck": config.get("transcript_visual_crosscheck_version") or "",
            "ocr_engine": config.get("ocr_engine") or context.options.get("ocr_model") or "",
            "ocr_engine_version": config.get("ocr_engine_version") or context.options.get("ocr_model_version") or "",
            "ocr_requested_device": config.get("ocr_device") or "",
            "vision": config.get("vision_model") or context.options.get("vision_model") or "",
            "vision_version": config.get("vision_model_version") or context.options.get("vision_model_version") or "",
        }.items()
        if value and (not text_only_semantic_context or key not in visual_model_keys)
    }
    # Only an OCR stage that actually observed a worker runtime may add this
    # provenance.  Empty metadata must not mutate unrelated stage checkpoints.
    runtime_identity = context.options.get("ocr_runtime_identity")
    if not text_only_semantic_context and isinstance(runtime_identity, dict) and runtime_identity:
        model_identity["ocr_actual_device"] = str(runtime_identity.get("actual_device") or "")
        model_identity["ocr_runtime_identity"] = canonical_json(runtime_identity)
    visual_prompt_keys = {"vision", "vision_adapter"}
    prompt_identity = {
        key: str(value)
        for key, value in {
            "segmentation": (
                context.options.get("segmentation_prompt_version") or config.get("segmentation_prompt_version")
            ),
            "extraction": (
                context.options.get("atomic_claim_prompt_version") or config.get("extraction_prompt_version")
            ),
            "vision": context.options.get("vision_prompt_version") or config.get("vision_prompt_version"),
            "vision_adapter": config.get("vision_adapter_version"),
        }.items()
        if value and (not text_only_semantic_context or key not in visual_prompt_keys)
    }
    return {
        "public_materialization_hash": public_hash,
        "model_identity": model_identity,
        "prompt_identity": prompt_identity,
        "worker_id": context.worker_id,
        "fencing_token": context.fencing_token,
    }


__all__ = [
    "StageContract",
    "StageResult",
    "StageRunner",
    "checkpoint_payload",
    "records_from_checkpoint_state",
    "wrap_all",
]

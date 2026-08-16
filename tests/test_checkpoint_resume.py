"""Checkpoint v2 断点恢复测试（详细修改方案 §4 P0-4）。"""
from __future__ import annotations

import pytest

from stock_content.application.pipeline import ContentPipeline, PipelineContext
from stock_content.application.stage_runner import StageRunner
from stock_content.domain.artifacts import SourceArtifact, make_artifact_id
from stock_content.domain.checkpoint import (
    CheckpointValidationError,
    build_checkpoint,
    validate_resume,
)


class SourceStage:
    name = "source"

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.data["source_runs"] = context.data.get("source_runs", 0) + 1
        payload = {"ref": context.source["ref"]}
        context.artifacts.source = SourceArtifact(
            artifact_id=make_artifact_id("source", payload),
            artifact_type="source",
            source_type=context.source["type"],
            source_ref=context.source["ref"],
            source_content_hash="hash-1",
            producer_stage="source",
        )
        return context


class KnowledgeStage:
    name = "knowledge"

    def __init__(self, fail_once: bool = False) -> None:
        self._fail_once = fail_once
        self.runs = 0

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.runs += 1
        if self._fail_once and self.runs == 1:
            raise RuntimeError("knowledge stage failed")
        context.data["knowledge_runs"] = context.data.get("knowledge_runs", 0) + 1
        return context


def test_checkpoint_v2_records_artifacts_and_hashes():
    source = SourceStage()
    knowledge = KnowledgeStage(fail_once=True)
    pipeline = ContentPipeline([StageRunner(source), StageRunner(knowledge)])
    context = PipelineContext(task_id="t1", source={"type": "bilibili", "ref": "BV1"})

    with pytest.raises(RuntimeError):
        pipeline.process(context)

    assert [record.stage for record in context.checkpoints] == ["source", "knowledge"]
    succeeded, failed = context.checkpoints
    assert succeeded.status == "SUCCEEDED"
    assert succeeded.output_artifact_ids == (context.artifacts.source.artifact_id,)
    assert succeeded.output_hashes == (context.artifacts.source.content_hash,)
    assert succeeded.schema_version == "checkpoint.v2"
    assert failed.status == "FAILED"
    assert "knowledge stage failed" in str(failed.error)


def test_resume_skips_completed_stages_after_hash_validation():
    source = SourceStage()
    knowledge = KnowledgeStage(fail_once=True)
    pipeline = ContentPipeline([StageRunner(source), StageRunner(knowledge)])
    context = PipelineContext(task_id="t1", source={"type": "bilibili", "ref": "BV1"})

    with pytest.raises(RuntimeError):
        pipeline.process(context)
    assert context.data["source_runs"] == 1

    # 断点恢复：source 已完成且哈希一致，只从 knowledge 继续。
    context.checkpoints = [record for record in context.checkpoints if record.status == "SUCCEEDED"]
    pipeline.process(context, resume=True)
    assert context.data["source_runs"] == 1  # source 不重做
    assert context.data["knowledge_runs"] == 1
    assert knowledge.runs == 2  # 首次失败一次 + 恢复后成功一次


def test_resume_rejects_tampered_artifact():
    source = SourceStage()
    knowledge = KnowledgeStage()
    pipeline = ContentPipeline([StageRunner(source), StageRunner(knowledge)])
    context = PipelineContext(task_id="t1", source={"type": "bilibili", "ref": "BV1"})
    pipeline.process(context)

    # 篡改 artifact 后哈希不一致，断点恢复必须拒绝。
    tampered = SourceArtifact(
        artifact_id=context.artifacts.source.artifact_id,
        artifact_type="source",
        source_type="bilibili",
        source_ref="BV1",
        source_content_hash="tampered",
    )
    context.artifacts.source = tampered
    with pytest.raises(CheckpointValidationError):
        pipeline.process(context, resume=True)


def test_resume_rejects_incompatible_stage_version():
    artifact = SourceArtifact(artifact_id="source-1", artifact_type="source", source_content_hash="h")
    record = build_checkpoint(stage="knowledge", stage_version="1.0.0", output_artifacts=[artifact])
    with pytest.raises(CheckpointValidationError):
        validate_resume([record], {artifact.artifact_id: artifact}, stage_versions={"knowledge": "2.0.0"})


def test_validate_resume_stops_at_failed_checkpoint():
    artifact = SourceArtifact(artifact_id="source-1", artifact_type="source", source_content_hash="h")
    ok = build_checkpoint(stage="source", output_artifacts=[artifact])
    failed = build_checkpoint(stage="knowledge", status="FAILED", error="boom")
    completed = validate_resume([ok, failed], {artifact.artifact_id: artifact})
    assert completed == ["source"]


def test_production_pipeline_stages_are_stage_runners(tmp_path):
    """P0 C-01：production build_application 的每个 Stage 均为 StageRunner，
    且 stage_version 来自稳定常量（非随机）。"""
    from stock_content.api.dependencies import STAGE_VERSIONS, build_application
    from stock_content.application.stage_runner import StageRunner

    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    stages = application._pipeline._stages  # noqa: SLF001
    assert stages, "生产 pipeline 不得为空"
    for stage in stages:
        assert isinstance(stage, StageRunner), f"{getattr(stage, 'name', stage)} 未经 StageRunner 包装"
        assert stage.stage_version == STAGE_VERSIONS[stage.name]
    # 稳定常量：两次构建版本一致。
    second = build_application(f"sqlite:///{tmp_path / 'content2.db'}", enable_qdrant=False)
    assert [s.stage_version for s in second._pipeline._stages] == [s.stage_version for s in stages]  # noqa: SLF001


def test_production_run_writes_checkpoint_v2_payload(tmp_path):
    """P0 C-01：生产链路成功 Stage 产生 SUCCEEDED checkpoint 并持久化到 task。"""
    from fastapi.testclient import TestClient

    from stock_content.api.dependencies import build_application
    from stock_content.api.main import create_app

    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    enqueue = client.post(
        "/api/v1/videos/bilibili/ingest",
        json={
            "bv_id": "BV1ckpt",
            "options": {"metadata": {"title": "ckpt"}, "transcript": "股票600000基本面良好。", "offline_fixture": True},
        },
    )
    application.process_next("ckpt-test")
    task = client.get(f"/api/v1/tasks/{enqueue.json()['task_id']}").json()
    assert task["status"] == "SUCCEEDED"
    checkpoint = task["checkpoint"]
    for stage_name in ("resolve", "asr", "knowledge", "summary", "content_snapshot", "persist"):
        record = checkpoint[stage_name]["checkpoint"]
        assert record["schema_version"] == "checkpoint.v2"
        assert record["status"] == "SUCCEEDED"
        assert record["stage_version"] == "1.0.0"

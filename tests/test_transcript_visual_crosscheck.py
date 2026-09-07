from __future__ import annotations

import pytest

from stock_content.application.pipeline import PipelineContext
from stock_content.application.service import ContentApplication, _validate_visual_checkpoint_identity
from stock_content.application.stage_runner import _checkpoint_identity
from stock_content.application.stages import (
    MultimodalContextStage,
    PersistStage,
    SemanticContextStage,
    TranscriptVisualCrosscheckStage,
)
from stock_content.domain.artifacts import (
    FrameArtifact,
    OCRArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    VisionArtifact,
    artifact_identity_payload,
    content_hash_of,
    deserialize_artifact,
    serialize_artifact,
)
from stock_content.domain.checkpoint import CheckpointValidationError, build_checkpoint
from stock_content.domain.multimodal_context_builder import MultimodalContextBuilder
from stock_content.domain.semantic_segment import build_semantic_segment_artifact, materialize_semantic_segments
from stock_content.domain.transcript_visual_crosscheck import TranscriptVisualCrossChecker


def _check(transcript: str, ocr: str = "", vision: str = "", *, confidence: float = 0.99, owned=True):
    return TranscriptVisualCrossChecker().check(
        frame={
            "frame_id": "f1",
            "timestamp_ms": 1_000,
            "semantic_segment_ids": ["s1"] if owned else [],
            "evidence_window_ids": ["w1"] if owned else [],
        },
        ocr_items=[{"text": ocr, "confidence_score": confidence, "ocr_engine": "paddle", "ocr_engine_version": "3"}]
        if ocr
        else [],
        vision_item={
            "visual_summary": vision,
            "symbols": [],
            "labels": [],
            "confidence_score": 0.8,
            "narration_aligned": True,
        }
        if vision
        else None,
        transcript_segments=[type("Segment", (), {"text": transcript, "segment_id": "t1", "confidence": 0.9})()],
    )


@pytest.mark.parametrize(
    ("transcript", "ocr", "vision", "relation", "reason"),
    [
        (
            "贵州茅台 600519 收入增长12%，2026年Q1",
            "贵州茅台 600519 12% 2026Q1",
            "上涨图",
            "SUPPORTS",
            "EXACT_TICKER_MATCH",
        ),
        ("政策文件提出银行增资", "政策 文件 银行 增资", "", "SUPPORTS", "EXACT_TERM_MATCH"),
        ("贵州茅台 600519 收入增长12%", "宁德时代 300750 收入下降8%", "", "CONTRADICTS", "TICKER_MISMATCH"),
        ("政策文件支持银行业", "直播间二维码", "", "UNRELATED", "NO_SHARED_HARD_FACT"),
        ("政策文件支持银行业", "600519", "", "UNKNOWN", "MISSING_OWNING_TRANSCRIPT_PROVENANCE"),
    ],
)
def test_crosscheck_classifies_hard_facts_and_records_audit_reason(transcript, ocr, vision, relation, reason):
    result = _check(transcript, ocr, vision, owned=relation != "UNKNOWN")
    assert result["relation"] == relation
    assert reason in result["reason_codes"]
    assert result["frame_id"] == "f1" and result["transcript_segment_ids"] == ["t1"]
    assert result["evidence_window_ids"] == (["w1"] if relation != "UNKNOWN" else [])


def test_high_confidence_ocr_candidate_is_trace_only_and_never_mutates_transcript():
    transcript = "贵州茅台 600519 收入增长12%"
    result = _check(transcript, "贵州茅台 600519 收入增长13%")
    assert result["relation"] == "SUPPORTS"
    candidate = next(item for item in result["correction_candidates"] if item["kind"] == "NUMBER")
    assert candidate["original"] == ["12%"] and candidate["replacement"] == ["13%"]
    assert candidate["status"] == "TRACE_ONLY_REQUIRES_INDEPENDENT_TRANSCRIPT_GROUNDING"
    assert transcript == "贵州茅台 600519 收入增长12%"


def test_malformed_or_unconfident_visual_inputs_stay_unknown_or_do_not_create_candidate():
    unknown = _check("600519收入12%")
    assert unknown["relation"] == "UNKNOWN"
    low = _check("600519收入12%", "600519收入13%", confidence=0.5)
    assert low["correction_candidates"] == []


def test_model_narration_alignment_alone_never_becomes_financial_fact_support():
    result = _check("600519收入增长12%", vision="600519收入增长12%")
    assert result["relation"] == "UNKNOWN"
    assert "MODEL_ONLY_FACT_NOT_INDEPENDENT_SUPPORT" in result["reason_codes"]


def _stage_context() -> PipelineContext:
    context = PipelineContext(task_id="crosscheck", source={})
    transcript = TranscriptArtifact(
        artifact_id="t",
        artifact_type="transcript",
        media_artifact_id="m",
        asr_model="fixture",
        asr_model_version="1",
        segments=[TranscriptSegmentItem(segment_index=0, start_seconds=0, end_seconds=2, text="600519收入增长12%")],
    )
    semantic = materialize_semantic_segments(transcript, [])[0]
    context.artifacts.transcript = transcript
    context.state.segments = list(transcript.segments)
    context.state.semantic_segments = [semantic]
    context.artifacts.semantic_segments = build_semantic_segment_artifact(transcript, [])
    for frame_id, text in (("support", "600519 12%"), ("unrelated", "直播二维码")):
        frame = FrameArtifact(
            artifact_id=f"a-{frame_id}",
            artifact_type="frame",
            frame_id=frame_id,
            timestamp_ms=1000,
            image_hash=frame_id,
            semantic_segment_ids=(semantic.semantic_segment_id,),
            evidence_window_ids=("w1",),
        )
        context.artifacts.frames.append(frame)
        context.state.frames.append(
            {
                "frame_id": frame_id,
                "timestamp_ms": 1000,
                "semantic_segment_ids": [semantic.semantic_segment_id],
                "evidence_window_ids": ["w1"],
            }
        )
        context.artifacts.ocr.append(
            OCRArtifact(
                artifact_id=f"o-{frame_id}",
                artifact_type="ocr",
                frame_artifact_id=frame.artifact_id,
                frame_id=frame_id,
                timestamp_ms=1000,
                image_hash=frame_id,
                semantic_segment_ids=(semantic.semantic_segment_id,),
                evidence_window_ids=("w1",),
                text=text,
                confidence_score=0.99,
                engine="paddle",
                engine_version="3",
            )
        )
        context.state.ocr_evidence.append(
            {
                "frame_id": frame_id,
                "timestamp_ms": 1000,
                "evidence_text": text,
                "text": text,
                "confidence_score": 0.99,
            }
        )
        context.artifacts.vision.append(
            VisionArtifact(
                artifact_id=f"v-{frame_id}",
                artifact_type="vision",
                frame_artifact_id=frame.artifact_id,
                frame_id=frame_id,
                timestamp_ms=1000,
                image_hash=frame_id,
                semantic_segment_ids=(semantic.semantic_segment_id,),
                evidence_window_ids=("w1",),
                payload={
                    "frame_id": frame_id,
                    "frame_artifact_id": frame.artifact_id,
                    "timestamp_ms": 1000,
                    "image_hash": frame_id,
                    "semantic_segment_ids": [semantic.semantic_segment_id],
                    "evidence_window_ids": ["w1"],
                    "ocr_text": text,
                    "visual_summary": text,
                    "label": text,
                    "symbols": [],
                    "labels": [],
                    "themes": [],
                    "confidence_score": 0.9,
                    "narration_aligned": True,
                    "model": "fixture",
                    "model_version": "fixture.v1",
                },
            )
        )
        context.state.frame_insights.append(
            {
                "frame_id": frame_id,
                "frame_artifact_id": f"a-{frame_id}",
                "timestamp_ms": 1000,
                "image_hash": frame_id,
                "semantic_segment_ids": [semantic.semantic_segment_id],
                "evidence_window_ids": ["w1"],
                "ocr_text": text,
                "visual_summary": text,
                "label": text,
                "labels": [],
                "themes": [],
                "symbols": [],
                "confidence_score": 0.9,
                "narration_aligned": True,
                "model": "fixture",
                "model_version": "fixture.v1",
            }
        )
    return context


def test_only_support_or_contradiction_enter_multimodal_and_semantic_context():
    context = _stage_context()
    TranscriptVisualCrosscheckStage().execute(context)
    assert [item["relation"] for item in context.state.transcript_visual_crosschecks] == ["SUPPORTS", "UNRELATED"]
    assert [item["frame_id"] for item in context.state.eligible_frame_insights] == ["support"]
    MultimodalContextStage(MultimodalContextBuilder()).execute(context)
    assert len(context.state.multimodal_context["items"]) == 1
    SemanticContextStage().execute(context)
    assert context.state.semantic_contexts[0].frame_refs == ["a-support"]


def test_crosscheck_artifact_roundtrips_is_content_addressed_and_rehydrates_eligible_state():
    context = _stage_context()
    TranscriptVisualCrosscheckStage().execute(context)
    artifact = context.artifacts.transcript_visual_crosscheck
    assert artifact is not None
    assert artifact.eligible_frame_ids == ("support",)
    assert [item.relation for item in artifact.relations] == ["SUPPORTS", "UNRELATED"]
    restored_artifact = deserialize_artifact(serialize_artifact(artifact))
    assert restored_artifact.content_hash == artifact.content_hash
    assert restored_artifact.to_dict() == artifact.to_dict()

    tampered = serialize_artifact(artifact)
    tampered["relations"][0]["reason_codes"] = [*tampered["relations"][0]["reason_codes"], "TAMPERED"]
    tampered_artifact = deserialize_artifact(tampered)
    assert content_hash_of(artifact_identity_payload(tampered_artifact)) != tampered_artifact.content_hash

    restored = PipelineContext(task_id="crosscheck", source={"type": "bilibili", "ref": "BV1fixture"})
    restored.artifacts = context.artifacts
    app = object.__new__(ContentApplication)
    app._occurrence_repository = None  # noqa: SLF001 - isolated resume seam
    app._lifecycle_repository = None  # noqa: SLF001 - isolated resume seam
    app._claim_repository = None  # noqa: SLF001 - isolated resume seam
    app._restore_typed_prefix(restored, {})  # noqa: SLF001 - isolated resume seam
    assert restored.state.transcript_visual_crosschecks == list(artifact.to_dict()["relations"])
    assert [item["frame_id"] for item in restored.state.eligible_frame_insights] == ["support"]
    builder = MultimodalContextBuilder()
    MultimodalContextStage(builder).execute(context)
    MultimodalContextStage(builder).execute(restored)
    assert content_hash_of(restored.state.multimodal_context) == content_hash_of(context.state.multimodal_context)


def test_visual_checkpoint_identity_rejects_changed_model_or_planner_version():
    context = _stage_context()
    context.options["pipeline_config"] = {
        "knowledge_evidence_window_planner_version": "windows.v1",
        "knowledge_frame_planner_version": "frames.v1",
        "transcript_visual_crosscheck_version": "crosscheck.v1",
        "ocr_engine": "paddleocr",
        "ocr_engine_version": "3",
        "vision_model": "gpt-terra",
        "vision_model_version": "2026-09-08",
        "vision_prompt_version": "vision.v1",
        "vision_adapter_version": "adapter.v1",
    }
    identity = _checkpoint_identity(context)
    record = build_checkpoint(
        stage="transcript_visual_crosscheck",
        model_identity=identity["model_identity"],
        prompt_identity=identity["prompt_identity"],
    )
    _validate_visual_checkpoint_identity([record], context)
    context.options["pipeline_config"]["vision_model_version"] = "2026-09-09"
    with pytest.raises(CheckpointValidationError, match="visual model identity incompatible"):
        _validate_visual_checkpoint_identity([record], context)


def test_crosscheck_without_visual_inputs_keeps_an_empty_internal_audit_artifact():
    context = _stage_context()
    context.artifacts.frames = []
    context.artifacts.ocr = []
    context.artifacts.vision = []
    context.state.frames = []
    context.state.ocr_evidence = []
    context.state.frame_insights = []
    TranscriptVisualCrosscheckStage().execute(context)
    artifact = context.artifacts.transcript_visual_crosscheck
    assert artifact is not None
    assert artifact.relations == () and artifact.eligible_frame_ids == ()
    assert context.state.eligible_frame_insights == []


class _Repository:
    def __init__(self) -> None:
        self.calls = []

    def upsert(self, *args) -> None:
        self.calls.append(args)

    def replace_for_video(self, *args) -> None:
        self.calls.append(args)

    def replace(self, *args) -> None:
        self.calls.append(args)


class _SummaryRepository:
    def upsert(self, *args) -> None:
        return None


def test_persist_projects_only_crosscheck_admitted_visual_rows():
    from types import SimpleNamespace

    context = _stage_context()
    TranscriptVisualCrosscheckStage().execute(context)
    context.state.video = SimpleNamespace(video_id="video-1")
    context.state.chapters = []
    context.state.knowledge = []
    context.state.summary = SimpleNamespace()
    videos, chapters, knowledge, multimodal = _Repository(), _Repository(), _Repository(), _Repository()
    PersistStage(videos, chapters, knowledge, _SummaryRepository(), multimodal=multimodal).execute(context)
    _video_id, frames, ocr, vision, _windows = multimodal.calls[0]
    assert [item["frame_id"] for item in frames] == ["support"]
    assert [item["frame_id"] for item in ocr] == ["support"]
    assert [item["frame_id"] for item in vision] == ["support"]

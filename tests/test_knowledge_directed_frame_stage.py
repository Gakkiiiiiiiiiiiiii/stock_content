from pathlib import Path

from stock_content.api.dependencies import build_application
from stock_content.application.pipeline import PipelineContext, RuntimeWorkspace
from stock_content.application.stages import KnowledgeDirectedFrameExtractionStage
from stock_content.domain.artifacts import MediaArtifact, TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.semantic_segment import materialize_semantic_segments


class _TargetedExtractor:
    def extract_targeted(self, _video_path, output_dir, requests, *, existing_image_hashes):
        assert not existing_image_hashes
        output: list[dict] = []
        for index, request in enumerate(requests):
            path = output_dir / f"target-{index}.jpg"
            path.write_bytes(f"image-{request.timestamp_ms}".encode())
            output.append(
                {
                    "timestamp_ms": request.timestamp_ms,
                    "image_path": str(path),
                    "image_hash": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                    "extraction_reason": request.extraction_reason,
                    "semantic_segment_ids": list(request.semantic_segment_ids),
                    "evidence_window_ids": list(request.evidence_window_ids),
                    "planner_version": request.planner_version,
                }
            )
        return output


def _context(tmp_path: Path) -> PipelineContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    transcript = TranscriptArtifact(
        artifact_id="transcript-targeted",
        artifact_type="transcript",
        media_artifact_id="media-targeted",
        asr_model="fixture",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(segment_index=0, start_seconds=0, end_seconds=2, text="开场。"),
            TranscriptSegmentItem(
                segment_index=1, start_seconds=2, end_seconds=6, text="600519收入增长12%，政策文件。"
            ),
        ],
    )
    context = PipelineContext(
        task_id="targeted-stage",
        source={"type": "fixture", "ref": "targeted"},
        options={"duration_ms": 8_000, "raw_storage_dir": str(tmp_path / "durable")},
        runtime=RuntimeWorkspace(work_dir=tmp_path, video_path=tmp_path / "video.mp4"),
    )
    context.runtime.video_path.write_bytes(b"video")
    context.artifacts.media = MediaArtifact(
        artifact_id="media-targeted", artifact_type="media", duration_ms=8_000, video_hash="immutable-video"
    )
    context.artifacts.transcript = transcript
    context.state.semantic_segments = materialize_semantic_segments(transcript, [])
    return context


def test_targeted_frame_stage_materializes_traceable_frames_after_semantic_segmentation(tmp_path):
    context = _context(tmp_path)
    result = KnowledgeDirectedFrameExtractionStage(_TargetedExtractor()).execute(context)

    assert result.produced_artifacts
    assert len(context.state.knowledge_evidence_windows) == 1
    frames = context.artifacts.frames
    assert all(item.producer_stage == "knowledge_frame" for item in frames)
    assert all(item.semantic_segment_ids for item in frames)
    assert all(item.evidence_window_ids for item in frames)
    assert all(item.planner_version == "knowledge-frame-plan.v1" for item in frames)
    assert all(item.frame_id.startswith("frame_") for item in frames)
    assert all(item.storage_ref and Path(item.storage_ref).is_file() for item in frames)
    assert "https://" not in repr(frames)


def test_targeted_frame_stage_replay_identity_is_stable_for_same_media_and_plan(tmp_path):
    first = _context(tmp_path / "first")
    second = _context(tmp_path / "second")

    KnowledgeDirectedFrameExtractionStage(_TargetedExtractor()).execute(first)
    KnowledgeDirectedFrameExtractionStage(_TargetedExtractor()).execute(second)

    assert [(item.frame_id, item.timestamp_ms, item.image_hash) for item in first.artifacts.frames] == [
        (item.frame_id, item.timestamp_ms, item.image_hash) for item in second.artifacts.frames
    ]


def test_pipeline_wires_targeted_frame_stage_after_semantic_segmentation(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'targeted-stage.db'}", enable_qdrant=False)
    names = [runner.name for runner in application._pipeline._stages]  # noqa: SLF001

    assert names.index("semantic_segmentation") < names.index("knowledge_frame") < names.index("ocr")
    assert names.index("ocr") < names.index("vision") < names.index("multimodal_context")
    assert names.index("vision") < names.index("temporal_window") < names.index("semantic_context")

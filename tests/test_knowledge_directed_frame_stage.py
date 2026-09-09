from pathlib import Path
from types import SimpleNamespace

import pytest

from stock_content.adapters.media.frame import FfmpegFrameExtractor
from stock_content.api.dependencies import build_application
from stock_content.application.pipeline import PipelineContext, RuntimeWorkspace
from stock_content.application.stages import ClaimVisualBindingStage, KnowledgeDirectedFrameExtractionStage
from stock_content.domain.artifacts import (
    FrameArtifact,
    MediaArtifact,
    OCRArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    VisionArtifact,
)
from stock_content.domain.claim_draft import ClaimOccurrenceDraft
from stock_content.domain.semantic_segment import materialize_semantic_segments


class _TargetedExtractor:
    def __init__(self):
        self.interval_calls = 0
        self.targeted_calls = 0

    def extract(self, *_args, **_kwargs):
        self.interval_calls += 1
        raise AssertionError("interval extraction must not run in the targeted stage")

    def extract_targeted(self, _video_path, output_dir, requests, *, existing_image_hashes):
        self.targeted_calls += 1
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
    extractor = _TargetedExtractor()
    result = KnowledgeDirectedFrameExtractionStage(extractor).execute(context)

    assert result.produced_artifacts
    assert extractor.interval_calls == 0
    assert extractor.targeted_calls == 1
    assert len(context.state.knowledge_evidence_windows) == 1
    frames = context.artifacts.frames
    assert all(item.producer_stage == "knowledge_frame" for item in frames)
    assert all(item.semantic_segment_ids for item in frames)
    assert all(item.evidence_window_ids for item in frames)
    assert all(item.planner_version == "knowledge-frame-plan.v1" for item in frames)
    assert all(item.planner_request_id.startswith("kfr_") for item in frames)
    assert all(item.frame_id.startswith("frame_") for item in frames)
    assert all(item.storage_ref and Path(item.storage_ref).is_file() for item in frames)
    assert "https://" not in repr(frames)


def test_targeted_frame_stage_replay_identity_is_stable_for_same_media_and_plan(tmp_path):
    first = _context(tmp_path / "first")
    second = _context(tmp_path / "second")

    KnowledgeDirectedFrameExtractionStage(_TargetedExtractor()).execute(first)
    KnowledgeDirectedFrameExtractionStage(_TargetedExtractor()).execute(second)

    assert [(item.frame_id, item.artifact_id, item.planner_request_id) for item in first.artifacts.frames] == [
        (item.frame_id, item.artifact_id, item.planner_request_id) for item in second.artifacts.frames
    ]


def test_pipeline_wires_targeted_frame_stage_after_semantic_segmentation(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'targeted-stage.db'}", enable_qdrant=False)
    names = [runner.name for runner in application._pipeline._stages]  # noqa: SLF001

    assert "frame" not in names
    assert names.index("semantic_segmentation") < names.index("semantic_context")
    assert names.index("semantic_context") < names.index("atomic_claim_extraction")
    assert names.index("atomic_claim_extraction") < names.index("atomic_claim_validation")
    assert names.index("atomic_claim_validation") < names.index("knowledge_frame") < names.index("frame_fixture")
    assert names.index("frame_fixture") < names.index("ocr")
    assert names.index("ocr") < names.index("vision") < names.index("transcript_visual_crosscheck")
    assert names.index("transcript_visual_crosscheck") < names.index("claim_visual_binding")
    assert names.index("claim_visual_binding") < names.index("multimodal_context")
    assert names.index("claim_visual_binding") < names.index("temporal_window")


def test_targeted_stage_uses_each_transcript_grounded_draft_as_its_own_window(tmp_path):
    context = _context(tmp_path)
    semantic = context.state.semantic_segments[0]
    # This models a long semantic chapter with two unrelated propositions;
    # the planner must not give both the chapter-wide visual evidence.
    context.state.claim_drafts = [
        SimpleNamespace(
            semantic_segment_id=semantic.semantic_segment_id,
            evidence_segment_indices=[0],
            normalized_statement="开场",
            conclusion="开场",
        ),
        SimpleNamespace(
            semantic_segment_id=semantic.semantic_segment_id,
            evidence_segment_indices=[1],
            normalized_statement="600519收入增长12%",
            conclusion="600519收入增长12%",
        ),
    ]
    KnowledgeDirectedFrameExtractionStage(_TargetedExtractor()).execute(context)

    windows = context.state.knowledge_evidence_windows
    assert len(windows) == 2
    assert windows[0].knowledge_identity != windows[1].knowledge_identity
    assert len(context.state.claim_evidence_window_ids) == 2
    assert set(context.state.claim_evidence_window_ids[0]).isdisjoint(context.state.claim_evidence_window_ids[1])
    # Every normal knowledge window gets its centre plus the required nearby
    # +/-1 second samples (clamping is allowed at media boundaries).
    requests_by_window = {}
    for frame in context.artifacts.frames:
        for window_id in frame.evidence_window_ids:
            requests_by_window.setdefault(window_id, []).append(frame.timestamp_ms)
    for window in windows:
        timestamps = requests_by_window[context.state.claim_evidence_window_ids[windows.index(window)][0]]
        assert window.center_ms in timestamps
        assert len(timestamps) >= 2


def test_crosschecked_visual_evidence_is_bound_to_its_own_claim_window(tmp_path):
    context = _context(tmp_path)
    first, second = context.state.semantic_segments[0], context.state.semantic_segments[0]
    context.state.claim_drafts = [
        ClaimOccurrenceDraft(
            semantic_segment_id=first.semantic_segment_id, knowledge_kind="CLAIM", claim_type="OPINION"
        ),
        ClaimOccurrenceDraft(
            semantic_segment_id=second.semantic_segment_id, knowledge_kind="CLAIM", claim_type="OPINION"
        ),
    ]
    context.state.claim_evidence_window_ids = {0: ("window-one",), 1: ("window-two",)}
    context.state.transcript_visual_crosschecks = [
        {"frame_id": "frame-one", "relation": "SUPPORTS"},
        {"frame_id": "frame-two", "relation": "SUPPORTS"},
    ]
    for frame_id, window_id, timestamp in (("frame-one", "window-one", 1_000), ("frame-two", "window-two", 5_000)):
        frame = FrameArtifact(
            artifact_id=f"artifact-{frame_id}",
            artifact_type="frame",
            media_artifact_id="media-targeted",
            frame_id=frame_id,
            timestamp_ms=timestamp,
            image_hash=frame_id,
            storage_ref="fixture",
            evidence_window_ids=(window_id,),
        )
        context.artifacts.add("frames", frame)
        context.artifacts.add(
            "ocr",
            OCRArtifact(
                artifact_id=f"ocr-{frame_id}",
                artifact_type="ocr",
                frame_artifact_id=frame.artifact_id,
                frame_id=frame_id,
                timestamp_ms=timestamp,
                image_hash=frame_id,
                evidence_window_ids=(window_id,),
                text=frame_id,
                bbox=[0, 0, 1, 1],
                confidence_score=0.9,
                engine="paddle",
                engine_version="1",
            ),
        )

    ClaimVisualBindingStage().execute(context)

    assert [anchor.frame_id for anchor in context.state.claim_drafts[0].visual_anchors] == ["frame-one"]
    assert [anchor.frame_id for anchor in context.state.claim_drafts[1].visual_anchors] == ["frame-two"]


@pytest.mark.parametrize(
    ("claim_nature", "source_label", "expected"),
    [
        ("ATTRIBUTED_SECONDARY_POLICY_REPORT", "displayed secondary policy page", True),
        ("ATTRIBUTED_SECONDARY_MACRO_FACT_REPORT", "displayed secondary macro page", True),
        ("SOURCE_FORECAST", "displayed secondary page", False),
        ("SOURCE_INTERPRETIVE_CAUSAL_THESIS", "speaker interpretation", False),
    ],
)
def test_displayed_secondary_page_binding_excludes_speaker_thesis_and_forecast(
    tmp_path, claim_nature, source_label, expected
):
    context = _context(tmp_path)
    semantic = context.state.semantic_segments[0]
    draft = ClaimOccurrenceDraft(
        semantic_segment_id=semantic.semantic_segment_id,
        knowledge_kind="CLAIM",
        claim_type="OPINION",
        bundle_v2={
            "claim_nature": claim_nature,
            "source_grade": "SECONDARY",
            "attribution": {"attributed": True, "source_label": source_label},
        },
    )
    context.state.claim_drafts = [draft]
    context.state.claim_evidence_window_ids = {0: ("owned-window",)}
    context.state.transcript_visual_crosschecks = [
        {"frame_id": "secondary-frame", "relation": "SUPPORTS_DISPLAYED_SECONDARY"}
    ]
    frame = FrameArtifact(
        artifact_id="secondary-frame-artifact", artifact_type="frame", media_artifact_id="media-targeted",
        frame_id="secondary-frame", timestamp_ms=3_000, image_hash="image", storage_ref="fixture",
        evidence_window_ids=("owned-window",),
    )
    context.artifacts.add("frames", frame)
    context.artifacts.add(
        "ocr",
        OCRArtifact(
            artifact_id="secondary-ocr", artifact_type="ocr", frame_artifact_id=frame.artifact_id,
            frame_id=frame.frame_id, timestamp_ms=3_000, image_hash="image", evidence_window_ids=("owned-window",),
            text="2030年目标9800", bbox=[0, 0, 1, 1], confidence_score=0.9, engine="paddle", engine_version="1",
        ),
    )
    context.artifacts.add(
        "vision",
        VisionArtifact(
            artifact_id="secondary-vision", artifact_type="vision", frame_artifact_id=frame.artifact_id,
            frame_id=frame.frame_id, timestamp_ms=3_000, image_hash="image", evidence_window_ids=("owned-window",),
            label="secondary page", labels=["secondary-news-page"], confidence_score=0.9,
            model_name="terra", model_version="1",
        ),
    )

    ClaimVisualBindingStage().execute(context)

    assert bool(context.state.claim_drafts[0].visual_anchors) is expected


def test_targeted_stage_fails_closed_for_live_media_without_semantic_windows(tmp_path):
    context = _context(tmp_path)
    context.state.semantic_segments = []

    with pytest.raises(ValueError, match="semantic evidence windows"):
        KnowledgeDirectedFrameExtractionStage(_TargetedExtractor()).execute(context)


def test_production_graph_never_wires_interval_extraction(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'targeted-only.db'}", enable_qdrant=False)
    stages = [runner._stage for runner in application._pipeline._stages]  # noqa: SLF001

    assert not any(type(stage).__name__ == "FrameExtractionStage" for stage in stages)
    targeted = next(stage for stage in stages if stage.name == "knowledge_frame")
    assert isinstance(targeted._extractor, FfmpegFrameExtractor)  # noqa: SLF001

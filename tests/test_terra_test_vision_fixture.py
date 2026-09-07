from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stock_content.adapters.media.terra_test_vision import TerraTestVisionFixtureAnalyzer
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import OCRStage, TranscriptVisualCrosscheckStage, VisionStage
from stock_content.domain.artifacts import FrameArtifact, TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.semantic_segment import build_semantic_segment_artifact, materialize_semantic_segments


class _IndependentPaddleOcr:
    def recognize(self, _path: str) -> dict:
        return {
            "text": "贵州茅台 600519 收入增长12%",
            "blocks": [{"text": "贵州茅台 600519 收入增长12%", "score": 0.99, "bbox": [1, 2, 30, 40]}],
            "engine": "paddleocr",
            "engine_version": "3.1",
        }


def _fixture(*, frame_id: str, timestamp_ms: int, content_hash: str, version: str = "2026-09-08") -> dict:
    return {
        "schema_version": "terra-test-vision-fixture.v1",
        "environment": "test",
        "model_name": "terra-test-substitute",
        "model_version": version,
        "frames": [
            {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "frame_content_hash": content_hash,
                "bbox": [10, 20, 300, 200],
                "label": "price_chart",
                "labels": ["price_chart", "stock_quote"],
                "visual_summary": "贵州茅台 600519 收入增长12%的图表",
                "themes": ["market"],
                "symbols": ["600519"],
                "confidence_score": 0.88,
                "narration_aligned": True,
                "model_name": "terra-test-substitute",
                "model_version": version,
                "environment": "test",
            }
        ],
    }


def _context(tmp_path: Path) -> tuple[PipelineContext, str]:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"reviewed-frame")
    content_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    transcript = TranscriptArtifact(
        artifact_id="transcript",
        artifact_type="transcript",
        media_artifact_id="media",
        asr_model="fixture",
        asr_model_version="1",
        segments=[TranscriptSegmentItem(segment_index=0, start_seconds=1, end_seconds=3, text="600519收入增长12%")],
    )
    semantic = materialize_semantic_segments(transcript, [])[0]
    context = PipelineContext(task_id="terra-test", source={})
    context.artifacts.transcript = transcript
    context.artifacts.semantic_segments = build_semantic_segment_artifact(transcript, [])
    context.state.segments = list(transcript.segments)
    context.state.transcript = "600519收入增长12%"
    context.state.semantic_segments = [semantic]
    frame = FrameArtifact(
        artifact_id="frame-artifact",
        artifact_type="frame",
        frame_id="frame-1",
        timestamp_ms=2_000,
        image_hash=content_hash,
        semantic_segment_ids=(semantic.semantic_segment_id,),
        evidence_window_ids=("window-1",),
    )
    context.artifacts.frames = [frame]
    context.state.frames = [
        {
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms,
            "image_hash": content_hash,
            "image_path": str(image),
            "semantic_segment_ids": [semantic.semantic_segment_id],
            "evidence_window_ids": ["window-1"],
        }
    ]
    return context, content_hash


def _write_fixture(tmp_path: Path, fixture: dict) -> Path:
    path = tmp_path / "terra-visual.fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


def test_explicit_terra_fixture_flows_through_vision_and_c4_audit_without_replacing_ocr(tmp_path):
    context, content_hash = _context(tmp_path)
    analyzer = TerraTestVisionFixtureAnalyzer.from_json_file(
        _write_fixture(tmp_path, _fixture(frame_id="frame-1", timestamp_ms=2_000, content_hash=content_hash)),
        environment="test",
    )

    OCRStage(_IndependentPaddleOcr()).execute(context)
    VisionStage(analyzer).execute(context)
    TranscriptVisualCrosscheckStage().execute(context)

    vision = context.artifacts.vision[0]
    assert vision.model_name == "terra-test-substitute"
    assert vision.payload["environment"] == "test"
    assert vision.payload["bbox"] == [10, 20, 300, 200]
    assert vision.payload["fixture_content_hash"] == analyzer.identity["fixture_content_hash"]
    assert context.artifacts.ocr[0].engine == "paddleocr"
    identity = context.artifacts.transcript_visual_crosscheck.visual_identity
    assert identity["vision_fixture_content_hash"] == analyzer.identity["fixture_content_hash"]
    assert context.artifacts.transcript_visual_crosscheck.eligible_frame_ids == ("frame-1",)


def test_terra_fixture_rejects_non_test_environment_schema_and_frame_membership(tmp_path):
    context, content_hash = _context(tmp_path)
    fixture = _fixture(frame_id="frame-1", timestamp_ms=2_000, content_hash=content_hash)
    with pytest.raises(ValueError, match="environment='test'"):
        TerraTestVisionFixtureAnalyzer(fixture, environment="production")

    fixture["environment"] = "production"
    with pytest.raises(ValueError, match="environment must be test"):
        TerraTestVisionFixtureAnalyzer(fixture, environment="test")

    fixture = _fixture(frame_id="unknown-frame", timestamp_ms=2_000, content_hash=content_hash)
    analyzer = TerraTestVisionFixtureAnalyzer(fixture, environment="test")
    with pytest.raises(ValueError, match="not a materialized frame"):
        VisionStage(analyzer).execute(context)


def test_terra_fixture_identity_changes_with_version_or_content_and_rejects_missing_provenance(tmp_path):
    _context(tmp_path)
    content_hash = "a" * 64
    baseline = _fixture(frame_id="frame-1", timestamp_ms=2_000, content_hash=content_hash)
    changed_version = _fixture(frame_id="frame-1", timestamp_ms=2_000, content_hash=content_hash, version="2026-09-09")
    changed_content = _fixture(frame_id="frame-1", timestamp_ms=2_000, content_hash="b" * 64)
    assert TerraTestVisionFixtureAnalyzer(baseline, environment="test").identity["fixture_content_hash"] != (
        TerraTestVisionFixtureAnalyzer(changed_version, environment="test").identity["fixture_content_hash"]
    )
    assert TerraTestVisionFixtureAnalyzer(baseline, environment="test").identity["fixture_content_hash"] != (
        TerraTestVisionFixtureAnalyzer(changed_content, environment="test").identity["fixture_content_hash"]
    )

    del baseline["frames"][0]["bbox"]
    with pytest.raises(ValueError, match="invalid schema"):
        TerraTestVisionFixtureAnalyzer(baseline, environment="test")

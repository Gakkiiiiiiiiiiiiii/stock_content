from __future__ import annotations

from pathlib import Path

import pytest

from stock_content.adapters.media.vision import HttpVisionAnalyzer
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import OCRStage, VisionStage
from stock_content.domain.artifacts import FrameArtifact, MediaArtifact


def _context(tmp_path: Path) -> PipelineContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fixture-image")
    context = PipelineContext(task_id="visual-c3", source={})
    context.state.transcript = "口播提到 600519 的收入增长。"
    context.state.frames = [
        {
            "frame_id": "frame-target",
            "timestamp_ms": 2_000,
            "image_path": str(image),
            "image_hash": "image-hash",
            "semantic_segment_ids": ["semantic-1"],
            "evidence_window_ids": ["window-1"],
        }
    ]
    context.artifacts.media = MediaArtifact(artifact_id="media-c3", artifact_type="media")
    context.artifacts.frames = [
        FrameArtifact(
            artifact_id="frame-artifact-c3",
            artifact_type="frame",
            media_artifact_id="media-c3",
            frame_id="frame-target",
            timestamp_ms=2_000,
            image_hash="image-hash",
            semantic_segment_ids=("semantic-1",),
            evidence_window_ids=("window-1",),
        )
    ]
    return context


class _Ocr:
    def recognize(self, _path: str, _image_hash: str = "") -> dict:
        return {
            "text": "贵州茅台 600519",
            "blocks": [{"text": "贵州茅台 600519", "score": 0.99, "bbox": [1, 2, 30, 40]}],
            "engine": "paddleocr",
            "engine_version": "3.1",
        }


class _Vision:
    def analyze(self, _path: str, _transcript: str) -> dict:
        return {
            "visual_summary": "行情图显示贵州茅台",
            "labels": ["price_chart"],
            "themes": ["market"],
            "symbols": ["600519"],
            "confidence_score": 0.88,
            "narration_aligned": True,
            "model": "gpt-vision-fixture",
            "model_version": "2026-09-07",
        }


def test_targeted_ocr_and_vision_artifacts_keep_complete_frame_provenance(tmp_path):
    context = _context(tmp_path)
    OCRStage(_Ocr()).execute(context)
    VisionStage(_Vision()).execute(context)

    ocr = context.artifacts.ocr[0]
    vision = context.artifacts.vision[0]
    assert (ocr.frame_id, ocr.timestamp_ms, ocr.image_hash) == ("frame-target", 2_000, "image-hash")
    assert ocr.semantic_segment_ids == ("semantic-1",)
    assert ocr.evidence_window_ids == ("window-1",)
    assert ocr.bbox == [1, 2, 30, 40] and ocr.confidence_score == 0.99
    assert ocr.engine == "paddleocr" and ocr.engine_version == "3.1"
    assert vision.labels == ["price_chart"] and vision.confidence_score == 0.88
    assert vision.model_name == "gpt-vision-fixture" and vision.model_version == "2026-09-07"
    assert vision.semantic_segment_ids == ("semantic-1",)
    assert len(context.state.frame_insights) == 1
    assert context.state.frame_insights[0]["ocr_text"] == "贵州茅台 600519"
    assert context.state.frame_insights[0]["visual_summary"] == "行情图显示贵州茅台"


def test_malformed_model_visual_data_fails_closed(tmp_path):
    context = _context(tmp_path)

    class BadOcr(_Ocr):
        def recognize(self, _path: str, _image_hash: str = "") -> dict:
            return {
                "blocks": [{"text": "x", "score": float("nan"), "bbox": [1, 2, 3, 4]}],
                "engine": "p",
                "engine_version": "3",
            }

    with pytest.raises(ValueError, match="OCR score"):
        OCRStage(BadOcr()).execute(context)

    class BadVision(_Vision):
        def analyze(self, _path: str, _transcript: str) -> dict:
            return {**super().analyze(_path, _transcript), "labels": []}

    with pytest.raises(ValueError, match="vision labels"):
        VisionStage(BadVision()).execute(_context(tmp_path / "bad-vision"))


def test_paddle_blank_frames_emit_no_ocr_artifacts_across_targeted_batch(tmp_path):
    """A legal empty recognition is absence of evidence, not fake OCR text.

    The live XiaoE regression contained blank/transition frames in its
    84-frame claim-directed batch.  Exercise the whole batch to make sure no
    single blank result aborts a resumed production task or writes a blank
    OCRArtifact that a later stage could mistake for evidence.
    """

    context = _context(tmp_path)
    context.state.frames = [
        {
            "frame_id": f"frame-{index}",
            "timestamp_ms": index * 1_000,
            "image_path": context.state.frames[0]["image_path"],
            "image_hash": f"hash-{index}",
        }
        for index in range(84)
    ]

    class BlankPaddle:
        def recognize(self, _path: str, _image_hash: str = "") -> dict:
            return {
                "blocks": [{"text": "   ", "score": 0.0, "bbox": [0, 0, 1, 1]}],
                "engine": "paddleocr",
                "engine_version": "3.7.0",
                "requested_device": "gpu:0",
                "actual_device": "gpu:0",
                "runtime_identity": {"actual_device": "gpu:0", "paddle_version": "3.3.0"},
            }

    OCRStage(BlankPaddle()).execute(context)

    assert context.state.ocr_evidence == []
    assert context.artifacts.ocr == []
    assert len(context.state.frame_insights) == 84
    assert all(
        item["ocr_text"] == "" and item["ocr_evidence"] == {"blocks": []}
        for item in context.state.frame_insights
    )


def test_blank_ocr_block_does_not_hide_a_non_string_schema_error(tmp_path):
    context = _context(tmp_path)

    class BadBlankPaddle:
        def recognize(self, _path: str, _image_hash: str = "") -> dict:
            return {
                "blocks": [{"text": None, "score": 0.0, "bbox": [0, 0, 1, 1]}],
                "engine": "paddleocr",
                "engine_version": "3.7.0",
            }

    with pytest.raises(ValueError, match="OCR text must be a non-empty string"):
        OCRStage(BadBlankPaddle()).execute(context)


def test_http_vision_schema_is_strict_and_model_version_is_configurable(tmp_path, monkeypatch):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"visual_summary":"chart","labels":["price_chart"],'
                                '"themes":[],"symbols":["600519"],"confidence_score":0.9,'
                                '"narration_aligned":true}'
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("stock_content.adapters.media.vision.httpx.post", lambda *args, **kwargs: Response())
    analyzer = HttpVisionAnalyzer("https://vision.invalid/v1", "gpt-test", model_version="revision-7")
    result = analyzer.analyze(str(image), "口播")
    assert result["model"] == "gpt-test" and result["model_version"] == "revision-7"

    with pytest.raises(RuntimeError, match="CONTENT_VISION_URL"):
        HttpVisionAnalyzer(model="gpt-test").analyze(str(image), "口播")
    with pytest.raises(ValueError, match="labels"):
        analyzer._validate({**result, "labels": []})  # noqa: SLF001 - schema boundary probe


def test_no_visual_path_remains_a_deterministic_empty_optional_output():
    context = PipelineContext(task_id="no-visual", source={})
    VisionStage(_Vision()).execute(context)
    assert context.artifacts.vision == []
    assert context.state.frame_insights == []

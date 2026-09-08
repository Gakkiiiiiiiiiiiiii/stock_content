from __future__ import annotations

import json
import struct
import zlib
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from stock_content.adapters.media.ocr_worker import Worker, _probe_png
from stock_content.api.readiness import _ocr_heartbeat
from stock_content.application.pipeline import ContentPipeline, PipelineContext
from stock_content.application.service import ContentApplication
from stock_content.application.stage_runner import StageRunner, _checkpoint_identity
from stock_content.application.stages import OCRStage
from stock_content.domain.artifacts import OCRArtifact, deserialize_artifact, serialize_artifact
from stock_content.domain.checkpoint import CheckpointValidationError, build_checkpoint

_IDENTITY = {
    "requested_device": "gpu:0",
    "actual_device": "gpu:0",
    "paddle_version": "3.3.0",
    "paddleocr_version": "3.7.0",
    "compiled_cuda": "true",
    "cuda_version": "12.9",
    "cudnn_version": "9.9",
    "device_count": "1",
}


class _RuntimeProbe:
    def __init__(self, observed):
        self.observed = observed
        self.probes = 0

    def start_and_probe(self):
        self.probes += 1
        return dict(self.observed)


class _RuntimeReportingOcrStage:
    name = "ocr"

    def __init__(self, observed_on_execute):
        self._engine = _RuntimeProbe(observed_on_execute)
        self._observed_on_execute = observed_on_execute
        self.runs = 0

    def execute(self, context):
        self.runs += 1
        context.options["ocr_runtime_identity"] = dict(self._observed_on_execute)
        return context


def _ocr_context(*, identity=None):
    options = {
        "pipeline_config": {
            "ocr_engine": "paddleocr",
            "ocr_engine_version": "3.7.0",
            "ocr_device": "gpu:0",
            "ocr_require_gpu": True,
        }
    }
    if identity is not None:
        options["ocr_runtime_identity"] = dict(identity)
    return PipelineContext(task_id="ocr-checkpoint", source={}, options=options)


def _legacy_ocr_checkpoint(context):
    """Model-compatible historical OCR record with no runtime seal."""
    identity = _checkpoint_identity(context)
    return build_checkpoint(stage="ocr", **identity)


def test_stage_runner_seals_post_execution_ocr_runtime_identity_for_resume():
    """The checkpoint is built after the OCR stage reports its real runtime."""
    stage = _RuntimeReportingOcrStage(_IDENTITY)
    context = _ocr_context()
    StageRunner(stage).execute(context)

    record = context.checkpoints[-1]
    assert record.model_identity["ocr_actual_device"] == "gpu:0"
    assert '"cuda_version":"12.9"' in record.model_identity["ocr_runtime_identity"]

    # With the exact same live runtime, service restore accepts the completed
    # StageRunner checkpoint and the generic pipeline resume skips execution.
    stage._engine.observed = dict(_IDENTITY)
    resume_context = _ocr_context()
    holder = SimpleNamespace(_pipeline=SimpleNamespace(_stages=[StageRunner(stage)]))
    from stock_content.application.service import _validate_visual_checkpoint_identity

    ContentApplication._observe_ocr_runtime_for_resume(holder, [record], resume_context)
    _validate_visual_checkpoint_identity([record], resume_context)
    assert stage._engine.probes == 1
    ContentPipeline([StageRunner(stage)]).process(context, resume=True)
    assert stage.runs == 1


@pytest.mark.parametrize(
    "changed_key", ["actual_device", "paddle_version", "paddleocr_version", "cuda_version", "cudnn_version"]
)
def test_stage_runner_ocr_checkpoint_rejects_changed_or_empty_runtime_on_restore(changed_key):
    stage = _RuntimeReportingOcrStage(_IDENTITY)
    context = _ocr_context()
    StageRunner(stage).execute(context)
    record = context.checkpoints[-1]
    holder = SimpleNamespace(_pipeline=SimpleNamespace(_stages=[StageRunner(stage)]))
    from stock_content.application.service import _validate_visual_checkpoint_identity

    changed = dict(_IDENTITY)
    changed[changed_key] = "gpu:1" if changed_key == "actual_device" else "changed"
    stage._engine.observed = changed
    with pytest.raises(CheckpointValidationError, match="OCR (GPU runtime unavailable|runtime identity incompatible)"):
        resume_context = _ocr_context()
        ContentApplication._observe_ocr_runtime_for_resume(holder, [record], resume_context)
        _validate_visual_checkpoint_identity([record], resume_context)

    # A legacy OCR checkpoint that never sealed an observed runtime cannot be
    # promoted to a required-GPU result merely because a current worker is up.
    legacy = _RuntimeReportingOcrStage({})
    legacy_context = _ocr_context()
    StageRunner(legacy).execute(legacy_context)
    legacy_record = legacy_context.checkpoints[-1]
    stage._engine.observed = dict(_IDENTITY)
    with pytest.raises(CheckpointValidationError, match="OCR runtime identity incompatible"):
        resume_context = _ocr_context()
        ContentApplication._observe_ocr_runtime_for_resume(holder, [legacy_record], resume_context)
        _validate_visual_checkpoint_identity([legacy_record], resume_context)


@pytest.mark.parametrize("fixture", [{"transcript": ""}, {"segments": []}])
def test_legacy_transcript_fixtures_resume_without_gpu_runtime_probe(fixture):
    """Transcript/segments fixtures predate the explicit offline flag."""
    from stock_content.application.service import _validate_visual_checkpoint_identity

    record = _legacy_ocr_checkpoint(_ocr_context())
    stage = _RuntimeReportingOcrStage(_IDENTITY)
    holder = SimpleNamespace(_pipeline=SimpleNamespace(_stages=[StageRunner(stage)]))
    context = _ocr_context()
    context.options.update(fixture)

    ContentApplication._observe_ocr_runtime_for_resume(holder, [record], context)
    _validate_visual_checkpoint_identity([record], context)

    assert stage._engine.probes == 0


def test_only_recognized_fixture_inputs_bypass_gpu_resume_probe():
    """Arbitrary truthy values must not create a production replay bypass."""
    record = _legacy_ocr_checkpoint(_ocr_context())
    stage = _RuntimeReportingOcrStage(_IDENTITY)
    holder = SimpleNamespace(_pipeline=SimpleNamespace(_stages=[StageRunner(stage)]))
    context = _ocr_context()
    context.options["note"] = "transcript"

    ContentApplication._observe_ocr_runtime_for_resume(holder, [record], context)

    assert stage._engine.probes == 1


def test_failed_ocr_checkpoint_is_a_retry_boundary_not_runtime_provenance():
    """A failed pre-execution record must not prevent the OCR retry itself."""
    from stock_content.application.service import _validate_visual_checkpoint_identity
    from stock_content.domain.checkpoint import build_checkpoint

    failed = build_checkpoint(stage="ocr", status="FAILED", error="worker exited")
    _validate_visual_checkpoint_identity([failed], _ocr_context())


def test_gpu_heartbeat_is_the_only_readiness_proof(tmp_path):
    path = tmp_path / "ocr.json"
    path.write_text(
        json.dumps(
            {
                "profile": "ocr",
                "health_code": "READY",
                "observed_at": datetime.now(UTC).isoformat(),
                "requested_device": "gpu:0",
                "actual_device": "gpu:0",
                "runtime_identity": _IDENTITY,
            }
        ),
        encoding="utf-8",
    )
    ready, details = _ocr_heartbeat(str(path))
    assert ready and details["paddle_version"] == "3.3.0"
    path.write_text(
        json.dumps(
            {
                "profile": "ocr",
                "health_code": "READY",
                "observed_at": datetime.now(UTC).isoformat(),
                "requested_device": "gpu:0",
                "actual_device": "cpu",
                "runtime_identity": _IDENTITY,
            }
        ),
        encoding="utf-8",
    )
    assert _ocr_heartbeat(str(path))[0] is False


def test_ocr_artifact_is_backward_compatible_and_gpu_identity_is_immutable(tmp_path):
    legacy = deserialize_artifact({"artifact_id": "ocr-legacy", "artifact_type": "ocr", "engine": "paddleocr"})
    assert isinstance(legacy, OCRArtifact) and legacy.runtime_identity == {} and legacy.actual_device == ""
    artifact = OCRArtifact(
        artifact_id="ocr-gpu",
        artifact_type="ocr",
        engine="paddleocr",
        engine_version="3.7.0",
        requested_device="gpu:0",
        actual_device="gpu:0",
        runtime_identity=_IDENTITY,
    )
    restored = deserialize_artifact(serialize_artifact(artifact))
    assert restored == artifact

    from stock_content.api.dependencies import STAGE_VERSIONS
    from stock_content.domain.checkpoint import CheckpointValidationError, build_checkpoint, validate_resume

    legacy_record = build_checkpoint(stage="ocr", stage_version="4.0.0", output_artifacts=[legacy])
    with pytest.raises(CheckpointValidationError, match="stage version incompatible"):
        validate_resume([legacy_record], {legacy.artifact_id: legacy}, stage_versions=STAGE_VERSIONS)


def test_stage_rejects_actual_cpu_for_required_gpu_frame():
    class CpuOcr:
        def recognize(self, *_args):
            return {
                "text": "x",
                "blocks": [],
                "engine": "paddleocr",
                "engine_version": "3.7.0",
                "requested_device": "gpu:0",
                "actual_device": "cpu",
                "runtime_identity": _IDENTITY,
            }

    context = PipelineContext(task_id="gpu", source={})
    context.state.frames = [{"frame_id": "f", "image_path": "f.png"}]
    with pytest.raises(ValueError, match="actual device"):
        OCRStage(CpuOcr()).execute(context)


def test_checkpoint_omits_empty_ocr_runtime_but_seals_observed_identity():
    context = PipelineContext(task_id="checkpoint", source={})
    assert "ocr_runtime_identity" not in _checkpoint_identity(context)["model_identity"]
    context.options["ocr_runtime_identity"] = _IDENTITY
    identity = _checkpoint_identity(context)["model_identity"]
    assert identity["ocr_actual_device"] == "gpu:0"
    assert '"paddle_version":"3.3.0"' in identity["ocr_runtime_identity"]


def test_worker_health_probe_is_a_valid_png_with_decodable_pixels():
    """A corrupt embedded probe must not reach Paddle's libpng decoder."""

    image = _probe_png()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    position, chunks = 8, {}
    while position < len(image):
        length = struct.unpack(">I", image[position : position + 4])[0]
        kind = image[position + 4 : position + 8]
        payload = image[position + 8 : position + 8 + length]
        checksum = struct.unpack(">I", image[position + 8 + length : position + 12 + length])[0]
        assert zlib.crc32(kind + payload) & 0xFFFFFFFF == checksum
        chunks.setdefault(kind, []).append(payload)
        position += 12 + length

    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"][0]
    )
    assert (width, height, bit_depth, color_type, compression, filter_method, interlace) == (160, 64, 8, 2, 0, 0, 0)
    scanlines = zlib.decompress(b"".join(chunks[b"IDAT"]))
    assert len(scanlines) == height * (1 + width * 3)
    assert all(scanlines[row * (1 + width * 3)] == 0 for row in range(height))


def test_worker_parses_paddleocr_v3_nested_result_payload(monkeypatch):
    class Result:
        json = {
            "res": {
                "rec_texts": ["600519"],
                "rec_scores": [0.98],
                "rec_boxes": [[[1, 2], [3, 4], [5, 6], [7, 8]]],
            }
        }

    class Predictor:
        def predict(self, _path):
            return [Result()]

    worker = Worker()
    worker._predictor = Predictor()
    monkeypatch.setattr(worker, "_verify_runtime", lambda: _IDENTITY)
    assert worker._predict("probe.png") == [
        {"text": "600519", "score": 0.98, "bbox": [[1, 2], [3, 4], [5, 6], [7, 8]]}
    ]

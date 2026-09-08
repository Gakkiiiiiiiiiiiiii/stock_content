from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import pytest

from stock_content.adapters.media.ocr import OcrRuntimeError, PaddleOcrEngine, _ocr_worker_environment

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


def test_paddle_ocr_uses_one_isolated_worker_and_keeps_frame_specific_output(tmp_path):
    created, requests = [], []

    class Pipe:
        def write(self, value):
            requests.append(json.loads(value))

        def flush(self):
            return None

    class Output:
        def readline(self):
            request = requests[-1]
            if request["op"] == "health":
                return json.dumps({"ok": True, "runtime_identity": _IDENTITY}) + "\n"
            return (
                json.dumps(
                    {
                        "ok": True,
                        "runtime_identity": _IDENTITY,
                        "text": "recognized:" + request["frame_path"],
                        "blocks": [
                            {"text": "recognized:" + request["frame_path"], "score": 0.91, "bbox": [1, 2, 30, 40]}
                        ],
                    }
                )
                + "\n"
            )

    class Process:
        stdin, stdout = Pipe(), Output()

        def poll(self):
            return None

        def terminate(self):
            return None

    def build_process(*args, **kwargs):
        created.append((args, kwargs))
        return Process()

    engine = PaddleOcrEngine(
        python_path="C:/ocr/python.exe", heartbeat_path=str(tmp_path / "heartbeat.json"), process_factory=build_process
    )

    first = engine.recognize("first-frame.jpg")
    second = engine.recognize("second-frame.jpg")

    assert len(created) == 1
    assert created[0][0][0] == ["C:/ocr/python.exe", "-m", "stock_content.adapters.media.ocr_worker"]
    assert requests == [
        {"op": "health"},
        {"op": "recognize", "frame_path": "first-frame.jpg", "image_hash": ""},
        {"op": "recognize", "frame_path": "second-frame.jpg", "image_hash": ""},
    ]
    assert first == {
        "text": "recognized:first-frame.jpg",
        "blocks": [{"text": "recognized:first-frame.jpg", "score": 0.91, "bbox": [1, 2, 30, 40]}],
        "engine": "paddleocr",
        "engine_version": "3.7.0",
        "requested_device": "gpu:0",
        "actual_device": "gpu:0",
        "runtime_identity": _IDENTITY,
    }
    assert second == {
        "text": "recognized:second-frame.jpg",
        "blocks": [{"text": "recognized:second-frame.jpg", "score": 0.91, "bbox": [1, 2, 30, 40]}],
        "engine": "paddleocr",
        "engine_version": "3.7.0",
        "requested_device": "gpu:0",
        "actual_device": "gpu:0",
        "runtime_identity": _IDENTITY,
    }
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["health_code"] == "READY" and heartbeat["actual_device"] == "gpu:0"


def test_ocr_worker_discards_high_volume_stderr_instead_of_leaving_an_unread_pipe():
    """Paddle diagnostics must never block the stdout JSON protocol."""

    created = []

    class Pipe:
        def write(self, _value):
            return None

        def flush(self):
            return None

    class Output:
        def readline(self):
            return json.dumps({"ok": True, "runtime_identity": _IDENTITY}) + "\n"

    class Process:
        stdin, stdout = Pipe(), Output()

        def poll(self):
            return None

        def terminate(self):
            return None

    engine = PaddleOcrEngine(
        python_path="C:/ocr/python.exe",
        process_factory=lambda *args, **kwargs: (created.append((args, kwargs)) or Process()),
    )

    engine.start_and_probe()

    assert created[0][1]["stderr"] is subprocess.DEVNULL


def test_successful_recognition_refreshes_ocr_heartbeat(tmp_path, monkeypatch):
    requests = []
    observed = iter(
        [
            "2026-09-08T00:00:00Z",
            "2026-09-08T00:01:00Z",
            "2026-09-08T00:02:00Z",
        ]
    )

    class Clock:
        def isoformat(self):
            return next(observed).replace("Z", "+00:00")

    class Pipe:
        def write(self, value):
            requests.append(json.loads(value))

        def flush(self):
            return None

    class Output:
        def readline(self):
            request = requests[-1]
            return json.dumps(
                {
                    "ok": True,
                    "runtime_identity": _IDENTITY,
                    "text": "ok" if request["op"] == "recognize" else "",
                    "blocks": [],
                }
            ) + "\n"

    class Process:
        stdin, stdout = Pipe(), Output()

        def poll(self):
            return None

        def terminate(self):
            return None

    monkeypatch.setattr(
        "stock_content.adapters.media.ocr.datetime",
        type("ClockModule", (), {"now": staticmethod(lambda _tz: Clock())}),
    )
    heartbeat_path = tmp_path / "ocr-heartbeat.json"
    engine = PaddleOcrEngine(
        python_path="C:/ocr/python.exe",
        heartbeat_path=str(heartbeat_path),
        process_factory=lambda *_args, **_kwargs: Process(),
    )

    engine.start_and_probe()
    started_at = json.loads(heartbeat_path.read_text(encoding="utf-8"))["observed_at"]
    engine.recognize("frame.jpg")
    refreshed_at = json.loads(heartbeat_path.read_text(encoding="utf-8"))["observed_at"]

    assert started_at == "2026-09-08T00:00:00Z"
    assert refreshed_at == "2026-09-08T00:01:00Z"


def test_cpu_or_failed_worker_is_fail_closed_before_recognition():
    class Pipe:
        def write(self, _value):
            return None

        def flush(self):
            return None

    class Output:
        def readline(self):
            return json.dumps({"ok": True, "runtime_identity": {**_IDENTITY, "actual_device": "cpu"}}) + "\n"

    class Process:
        stdin, stdout = Pipe(), Output()

        def poll(self):
            return None

    with pytest.raises(OcrRuntimeError, match="OCR_DEVICE_MISMATCH"):
        PaddleOcrEngine(
            python_path="C:/ocr/python.exe", process_factory=lambda *_args, **_kwargs: Process()
        ).start_and_probe()


def test_parent_import_graph_has_no_paddle_or_torch():
    tree = ast.parse(Path("src/stock_content/adapters/media/ocr.py").read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imports |= {
        node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not {"paddle", "paddleocr", "torch", "pyannote"} & imports


def test_worker_environment_does_not_forward_service_secrets(monkeypatch):
    monkeypatch.setenv("CONTENT_BILIBILI_COOKIE", "must-not-cross-boundary")
    monkeypatch.setenv("USERPROFILE", "C:/Users/ocr-runtime")
    environment = _ocr_worker_environment("C:/ocr/python.exe", "gpu:0", True)
    assert "CONTENT_BILIBILI_COOKIE" not in environment
    assert environment["CONTENT_OCR_DEVICE"] == "gpu:0"
    assert environment["HOME"] == "C:\\Users\\ocr-runtime"
    assert environment["USERPROFILE"] == "C:\\Users\\ocr-runtime"
    assert environment["PADDLE_PDX_CACHE_HOME"] == "C:\\Users\\ocr-runtime\\.paddlex"
    assert environment["MODELSCOPE_CACHE"] == "C:\\Users\\ocr-runtime\\.paddlex\\modelscope"
    assert os.environ["CONTENT_BILIBILI_COOKIE"] not in json.dumps(environment)


def test_worker_environment_uses_explicit_ocr_cache_without_profile_or_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("CONTENT_OCR_CACHE_HOME", str(tmp_path / "isolated-cache"))
    monkeypatch.setenv("CONTENT_XIAOE_COOKIE", "must-not-cross-boundary")

    environment = _ocr_worker_environment("C:/ocr/python.exe", "gpu:0", True)

    assert environment["PADDLE_PDX_CACHE_HOME"] == str(tmp_path / "isolated-cache")
    assert environment["MODELSCOPE_CACHE"] == str(tmp_path / "isolated-cache" / "modelscope")
    assert environment["HOME"] == str(tmp_path)
    assert environment["USERPROFILE"] == str(tmp_path)
    assert "CONTENT_XIAOE_COOKIE" not in environment
    assert os.environ["CONTENT_XIAOE_COOKIE"] not in json.dumps(environment)

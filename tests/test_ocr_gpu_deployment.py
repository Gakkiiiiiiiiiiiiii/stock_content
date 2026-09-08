from __future__ import annotations

import os
from pathlib import Path

from stock_content.adapters.media.ocr import _ocr_worker_environment


def test_video_worker_wires_an_isolated_cp311_gpu_ocr_runtime_and_browser():
    root = Path(__file__).parents[1]
    image = (root / "docker" / "Dockerfile.video").read_text(encoding="utf-8")
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim AS python311" in image
    assert "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04" in image
    assert "COPY --from=python311 /usr/local /usr/local" in image
    assert "/usr/local/bin/python -m venv /opt/ocr" in image
    assert "cp311-cp311-manylinux2014_x86_64.whl" in image
    video_environment = image.split("/opt/video/bin/pip install", maxsplit=1)[1].split(
        "/usr/local/bin/python -m venv /opt/ocr", maxsplit=1
    )[0]
    assert "paddlepaddle" not in video_environment
    assert "playwright install --with-deps chromium" in image
    assert "PLAYWRIGHT_BROWSERS_PATH=/ms-playwright" in image
    assert "CONTENT_OCR_PYTHON: /opt/ocr/bin/python" in compose
    assert "CONTENT_OCR_DEVICE: gpu:0" in compose
    assert 'CONTENT_OCR_REQUIRE_GPU: "true"' in compose
    assert "CONTENT_OCR_HEARTBEAT_FILE: /var/run/stock-content/ocr-heartbeat.json" in compose
    assert compose.count("CONTENT_OCR_HEARTBEAT_FILE: /var/run/stock-content/ocr-heartbeat.json") == 2
    assert "CONTENT_OCR_LIBRARY_PATH:" in compose
    assert "content-worker-state:/var/run/stock-content:ro" in compose
    assert "content-ocr-cache:/var/cache/stock-content/ocr" in compose
    assert "driver: nvidia" in compose
    assert "capabilities: [gpu]" in compose


def test_standalone_ocr_image_uses_cp311_not_ubuntu_default_python():
    root = Path(__file__).parents[1]
    image = (root / "docker" / "Dockerfile.ocr-gpu").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim AS python311" in image
    assert "/usr/local/bin/python -m venv /opt/ocr" in image
    assert "python3 -m venv /opt/ocr" not in image
    assert "CONTENT_OCR_LIBRARY_PATH=" in image


def test_scrubbed_ocr_child_receives_only_explicit_native_library_paths(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "host-torch-libraries")
    monkeypatch.setenv("CONTENT_OCR_LIBRARY_PATH", os.pathsep.join(("/cuda", "/ocr/cudnn")))

    environment = _ocr_worker_environment("/opt/ocr/bin/python", "gpu:0", True)

    assert environment["LD_LIBRARY_PATH"] == os.pathsep.join(("/cuda", "/ocr/cudnn"))
    assert "host-torch-libraries" not in environment["LD_LIBRARY_PATH"]

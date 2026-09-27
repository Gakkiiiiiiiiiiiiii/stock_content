from __future__ import annotations

import os
from pathlib import Path

import yaml

from stock_content.adapters.media.ocr import _ocr_worker_environment


def test_docker_context_excludes_private_runtime_artifacts():
    root = Path(__file__).parents[1]
    exclusions = set((root / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".env", ".codex-runs/"} <= exclusions
    assert "src/" not in exclusions
    assert "contracts/" not in exclusions


def test_video_worker_wires_an_isolated_cp311_gpu_ocr_runtime_and_browser():
    root = Path(__file__).parents[1]
    image = (root / "docker" / "Dockerfile.video").read_text(encoding="utf-8")
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim-bookworm AS python311" in image
    assert "nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04" in image
    assert "COPY --from=python311 /usr/local /usr/local" in image
    assert image.index('/usr/local/bin/python -c "import ssl, _hashlib"') < image.index(
        "/opt/video/bin/pip install"
    )
    assert "/usr/local/bin/python -m venv /opt/ocr" in image
    assert (
        "/opt/ocr/bin/pip install --no-cache-dir --no-deps paddlepaddle-gpu==3.3.0 "
        "-i https://www.paddlepaddle.org.cn/packages/stable/cu129/"
    ) in image
    assert "paddlepaddle_gpu-3.3.0-cp311-cp311-manylinux2014_x86_64.whl" not in image
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

    assert "FROM python:3.11-slim-bookworm AS python311" in image
    assert image.index('/usr/local/bin/python -c "import ssl, _hashlib"') < image.index(
        "/opt/ocr/bin/pip install"
    )
    assert "/usr/local/bin/python -m venv /opt/ocr" in image
    assert (
        "/opt/ocr/bin/pip install --no-cache-dir --no-deps paddlepaddle-gpu==3.3.0 "
        "-i https://www.paddlepaddle.org.cn/packages/stable/cu129/"
    ) in image
    assert "paddlepaddle_gpu-3.3.0-cp311-cp311-manylinux2014_x86_64.whl" not in image
    assert "python3 -m venv /opt/ocr" not in image
    assert "CONTENT_OCR_LIBRARY_PATH=" in image


def test_ocr_images_check_pinned_gpu_dependencies_without_build_time_driver_load():
    root = Path(__file__).parents[1]
    locked = (root / "requirements" / "ocr-gpu-cu129.lock").read_text(encoding="utf-8").splitlines()
    assert "paddlepaddle-gpu==3.3.0" in locked
    assert "paddleocr==3.7.0" in locked
    for name in ("Dockerfile.video", "Dockerfile.ocr-gpu"):
        image = (root / "docker" / name).read_text(encoding="utf-8")
        steps = (
            "/opt/ocr/bin/pip install --no-cache-dir -r requirements/ocr-gpu-cu129.lock",
            "/opt/ocr/bin/pip install --no-cache-dir .",
            "/opt/ocr/bin/pip check",
        )
        assert [image.index(step) for step in steps] == sorted(image.index(step) for step in steps)
        assert "/opt/ocr/bin/pip install --no-deps ." not in image
        assert '/opt/ocr/bin/python -c "import paddle"' not in image
        assert "CONTENT_OCR_REQUIRE_GPU=true" in image
    video = (root / "docker" / "Dockerfile.video").read_text(encoding="utf-8")
    diagnostic = (root / "docker" / "Dockerfile.ocr-gpu").read_text(encoding="utf-8")
    assert 'CMD ["python", "-m", "stock_content.workers.content_worker"]' in video
    assert 'CMD ["/opt/ocr/bin/python", "-m", "stock_content.adapters.media.ocr_worker"]' in diagnostic


def test_scrubbed_ocr_child_receives_only_explicit_native_library_paths(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "host-torch-libraries")
    monkeypatch.setenv("CONTENT_OCR_LIBRARY_PATH", os.pathsep.join(("/cuda", "/ocr/cudnn")))

    environment = _ocr_worker_environment("/opt/ocr/bin/python", "gpu:0", True)

    assert environment["LD_LIBRARY_PATH"] == os.pathsep.join(("/cuda", "/ocr/cudnn"))
    assert "host-torch-libraries" not in environment["LD_LIBRARY_PATH"]


def test_ocr_child_receives_only_explicit_temp_path(monkeypatch):
    monkeypatch.setenv("TMPDIR", "/work")
    monkeypatch.setenv("TEMP", "host-temp-secret")
    monkeypatch.setenv("TMP", "host-tmp-secret")
    monkeypatch.setenv("CONTENT_BILIBILI_COOKIE", "credential-secret")

    environment = _ocr_worker_environment("/opt/ocr/bin/python", "gpu:0", True)

    assert environment["TMPDIR"] == "/work"
    assert "TEMP" not in environment
    assert "TMP" not in environment
    assert "CONTENT_BILIBILI_COOKIE" not in environment
    assert "credential-secret" not in str(environment)

    monkeypatch.delenv("TMPDIR")
    fallback = _ocr_worker_environment("/opt/ocr/bin/python", "gpu:0", True)
    assert "TMPDIR" not in fallback
    if os.name == "nt":
        assert fallback["TEMP"] == "host-temp-secret"
        assert fallback["TMP"] == "host-tmp-secret"
    else:
        assert "TEMP" not in fallback
        assert "TMP" not in fallback
    assert "CONTENT_BILIBILI_COOKIE" not in fallback


def test_compose_shares_durable_raw_storage_read_only_with_api():
    root = Path(__file__).parents[1]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    migration = services["content-migrate"]
    api = services["stock-content"]
    video = services["stock-content-video-worker"]

    assert "content-raw-storage" in compose["volumes"]
    assert migration["command"] == [
        "sh", "-ec", "chown 10001:10001 /data /var/run/stock-content && exec stock-content-migrate"
    ]
    assert "content-raw-storage:/data" in migration["volumes"]
    assert "content-worker-state:/var/run/stock-content" in migration["volumes"]
    assert api["environment"]["CONTENT_RAW_STORAGE_DIR"] == "/data"
    assert video["environment"]["CONTENT_RAW_STORAGE_DIR"] == "/data"
    assert video["environment"]["TMPDIR"] == "/work"
    assert "content-raw-storage:/data:ro" in api["volumes"]
    assert "content-raw-storage:/data" in video["volumes"]
    assert video["read_only"] is True
    assert all(not mount.startswith("/data:") for mount in video["tmpfs"])
    assert video["command"][:2] == ["sh", "-ec"]
    startup = video["command"][2]
    assert "mkdir -p /data/raw /data/frames" in startup
    for directory in ("/work", "/data/raw", "/data/frames"):
        assert f'tempfile.TemporaryFile(dir="{directory}").close()' in startup
    assert startup.endswith("&& exec stock-content-worker")
    assert "$" not in startup
    assert api["depends_on"]["content-migrate"]["condition"] == "service_completed_successfully"
    assert video["depends_on"]["content-migrate"]["condition"] == "service_completed_successfully"
    assert api["secrets"] == ["content-service-api-key"]

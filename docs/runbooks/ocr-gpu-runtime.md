# Isolated OCR GPU runtime

OCR uses `CONTENT_OCR_DEVICE=gpu:0` and `CONTENT_OCR_REQUIRE_GPU=true` by
default. Create its dedicated Windows venv from the repository root with:

```powershell
.\scripts\install-ocr-gpu-cu129.ps1 -Python C:\Python311\python.exe
$env:CONTENT_OCR_PYTHON = "$PWD\.venv-ocr-cu129\Scripts\python.exe"
$env:CONTENT_OCR_RUNTIME_PATH = "$PWD\.venv-ocr-cu129\Scripts;$PWD\.venv-ocr-cu129\Library\bin"
$env:CONTENT_OCR_HEARTBEAT_FILE = "C:\ProgramData\stock-content\ocr-heartbeat.json"
$env:CONTENT_OCR_DEVICE = "gpu:0"
$env:CONTENT_OCR_REQUIRE_GPU = "true"
& $env:CONTENT_OCR_PYTHON -m stock_content.cli.ocr_doctor
```

The lock pins PaddleOCR 3.7.0 and the cu129-compatible CUDA runtime 12.9.37
and cuDNN 9.9.0.52. Paddle 3.3.0 cu129 is deliberately installed from the
official Paddle artifact rather than ordinary PyPI requirements. The OCR venv
must not contain Torch; the ASR/Pyannote environment must not contain Paddle.
`/health/video-ingestion-ready` reads only the heartbeat. A missing, stale,
CPU, malformed, or non-READY heartbeat yields HTTP 503 and video work must not
be claimed. The API and parent worker do not import Paddle.

## Linux Compose deployment

Start the video profile on an NVIDIA Container Toolkit host. The
`stock-content-video-worker` image contains two isolated Python 3.11 venvs:
`/opt/video` for media/browser work and `/opt/ocr` for Paddle. Compose grants
the worker one NVIDIA GPU and sets `CONTENT_OCR_PYTHON=/opt/ocr/bin/python`,
`CONTENT_OCR_DEVICE=gpu:0`, and `CONTENT_OCR_REQUIRE_GPU=true`. The worker
cannot claim video work until its real OCR probe writes a fresh `READY`
heartbeat with actual `gpu:0`. The API reads that heartbeat through the shared
read-only `content-worker-state` volume.

`CONTENT_OCR_LIBRARY_PATH` is passed verbatim as the OCR subprocess's
`LD_LIBRARY_PATH`; keep it limited to CUDA/cuDNN and OCR-venv directories. Do
not place the ASR/Torch venv or a broad host library path there. OCR model
caches persist only in the `content-ocr-cache` volume and are writable by the
non-root worker user. Chromium is installed in the video image during build so
the independent authenticated Playwright session remains available at runtime.

Validate a built standalone diagnostic image before rollout (with a GPU
available):

```sh
docker build -f docker/Dockerfile.ocr-gpu -t stock-content-ocr-gpu .
docker run --rm --gpus all -i stock-content-ocr-gpu \
  /opt/ocr/bin/python -m stock_content.cli.ocr_doctor
```

An unavailable GPU, cp311/Paddle mismatch, missing native library, stale
heartbeat, or a CPU report is a deployment failure, not permission to fall
back to CPU.

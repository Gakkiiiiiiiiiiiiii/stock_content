"""The only process allowed to import Paddle for content OCR."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any


class Worker:
    def __init__(self) -> None:
        self.device = os.getenv("CONTENT_OCR_DEVICE", "gpu:0")
        self.require_gpu = os.getenv("CONTENT_OCR_REQUIRE_GPU", "true").lower() == "true"
        self._predictor: Any | None = None
        self._identity: dict[str, str] | None = None

    def health(self) -> dict[str, Any]:
        self._initialize()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            probe = Path(handle.name)
            handle.write(_probe_png())
        try:
            self._predict(str(probe))
        finally:
            probe.unlink(missing_ok=True)
        return {"ok": True, "runtime_identity": self._verify_runtime()}

    def recognize(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._initialize()
        path = payload.get("frame_path")
        if not isinstance(path, str) or not path:
            return {"ok": False, "code": "OCR_PROTOCOL_INVALID"}
        blocks = self._predict(path)
        return {
            "ok": True,
            "blocks": blocks,
            "text": "\n".join(item["text"] for item in blocks),
            "runtime_identity": self._verify_runtime(),
        }

    def _initialize(self) -> None:
        if self._predictor is not None:
            return
        with contextlib.redirect_stdout(sys.stderr):
            import paddle
            from paddleocr import PaddleOCR

            paddle.set_device(self.device)
            self._identity = _identity(paddle, self.device)
            _require_gpu(self._identity, self.require_gpu)
            self._predictor = PaddleOCR(
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    def _verify_runtime(self) -> dict[str, str]:
        import paddle

        identity = _identity(paddle, self.device)
        _require_gpu(identity, self.require_gpu)
        if self._identity is not None and identity != self._identity:
            raise RuntimeError("OCR_RUNTIME_IDENTITY_CHANGED")
        return identity

    def _predict(self, path: str) -> list[dict[str, Any]]:
        assert self._predictor is not None
        with contextlib.redirect_stdout(sys.stderr):
            result = self._predictor.predict(path)
        blocks = []
        for item in result:
            payload = item.json if hasattr(item, "json") else item
            # PaddleOCR 3.x result objects expose recognition fields below
            # ``res``; retain support for direct mapping test doubles.
            if isinstance(payload, dict):
                payload = payload.get("res", payload)
            for text, score, bbox in zip(
                payload.get("rec_texts", []), payload.get("rec_scores", []), payload.get("rec_boxes", []), strict=False
            ):
                blocks.append({"text": str(text), "score": float(score), "bbox": bbox})
        self._verify_runtime()
        return blocks


def _identity(paddle: Any, requested_device: str) -> dict[str, str]:
    version = getattr(paddle, "version", object())
    cuda = getattr(version, "cuda", lambda: "")()
    cudnn = getattr(version, "cudnn", lambda: "")()
    return {
        "requested_device": requested_device,
        "actual_device": str(paddle.get_device()),
        "paddle_version": str(getattr(paddle, "__version__", "unknown")),
        "paddleocr_version": importlib.metadata.version("paddleocr"),
        "compiled_cuda": str(bool(paddle.is_compiled_with_cuda())).lower(),
        "cuda_version": str(cuda or "unknown"),
        "cudnn_version": str(cudnn or "unknown"),
        "device_count": str(paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0),
    }


def _require_gpu(identity: dict[str, str], required: bool) -> None:
    if not required:
        return
    if (
        identity["requested_device"] != "gpu:0"
        or not identity["actual_device"].lower().startswith("gpu:0")
        or identity["compiled_cuda"] != "true"
        or int(identity["device_count"]) < 1
    ):
        raise RuntimeError("OCR_GPU_UNAVAILABLE")


def main() -> None:
    worker = Worker()
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            response = (
                worker.health()
                if payload.get("op") == "health"
                else worker.recognize(payload)
                if payload.get("op") == "recognize"
                else {"ok": False, "code": "OCR_PROTOCOL_INVALID"}
            )
        except Exception as exc:  # protocol output is code-only: no environment or request secrets leak.
            response = {
                "ok": False,
                "code": str(exc) if str(exc).startswith("OCR_") else "OCR_INITIALIZATION_OR_INFERENCE_FAILED",
            }
        print(json.dumps(response, separators=(",", ":")), flush=True)


def _probe_png() -> bytes:
    """Build a small, dependency-free PNG for the worker's real inference probe.

    Keeping the encoder here avoids depending on Pillow in the isolated OCR
    runtime.  The explicit chunk CRCs and zlib stream make the probe valid for
    libpng-based Paddle backends, unlike an arbitrary 1x1 byte literal.
    """

    width, height, scale = 160, 64, 6
    pixels = bytearray(b"\xff" * (width * height * 3))
    glyphs = {
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
        "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    }
    for index, character in enumerate("OCR"):
        left = 20 + index * 45
        for row, bits in enumerate(glyphs[character]):
            for column, bit in enumerate(bits):
                if bit != "1":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        x, y = left + column * scale + dx, 11 + row * scale + dy
                        offset = (y * width + x) * 3
                        pixels[offset : offset + 3] = b"\x00\x00\x00"
    scanlines = b"".join(
        b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3] for row in range(height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        return struct.pack(">I", len(payload)) + kind + payload + checksum

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(
        b"IDAT", zlib.compress(scanlines, level=9)
    ) + chunk(b"IEND", b"")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OcrRuntimeError(RuntimeError):
    """A fail-closed OCR runtime/protocol failure with a stable health code."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


class PaddleOcrEngine:
    """JSON-line client for an isolated, long-lived Paddle OCR runtime.

    This parent adapter intentionally contains no Paddle import: CUDA DLLs
    cannot enter the same process as the ASR/Pyannote Torch runtime.
    """

    def __init__(
        self,
        *,
        python_path: str | None = None,
        device: str | None = None,
        require_gpu: bool | None = None,
        heartbeat_path: str | None = None,
        process_factory: Any | None = None,
    ) -> None:
        self._python_path = python_path or os.getenv("CONTENT_OCR_PYTHON", "")
        self._device = device or os.getenv("CONTENT_OCR_DEVICE", "gpu:0")
        self._require_gpu = _env_bool("CONTENT_OCR_REQUIRE_GPU", True) if require_gpu is None else require_gpu
        self._heartbeat_path = heartbeat_path or os.getenv("CONTENT_OCR_HEARTBEAT_FILE", "")
        self._process_factory = process_factory or subprocess.Popen
        self._process: Any | None = None
        self._runtime_identity: dict[str, str] = {}

    @property
    def runtime_identity(self) -> dict[str, str]:
        return dict(self._runtime_identity)

    def start_and_probe(self) -> dict[str, str]:
        """Initialize once and prove a GPU prediction before work is claimed."""
        response = self._request({"op": "health"})
        candidate = _reported_identity(response)
        try:
            identity = _runtime_identity(response)
        except OcrRuntimeError as exc:
            self._write_heartbeat(exc.code, candidate)
            raise
        self._runtime_identity = identity
        self._write_heartbeat("READY", identity)
        return identity

    def recognize(self, frame_path: str, image_hash: str = "") -> dict[str, Any]:
        if not self._runtime_identity:
            self.start_and_probe()
        response = self._request({"op": "recognize", "frame_path": str(Path(frame_path)), "image_hash": image_hash})
        candidate = _reported_identity(response)
        try:
            identity = _runtime_identity(response)
        except OcrRuntimeError as exc:
            self._write_heartbeat(exc.code, candidate)
            raise
        if identity != self._runtime_identity:
            self._write_heartbeat("OCR_RUNTIME_IDENTITY_CHANGED", identity)
            raise OcrRuntimeError("OCR_RUNTIME_IDENTITY_CHANGED")
        # The readiness proof represents a living, proven GPU runtime rather
        # than merely its startup event.  A successful prediction is the only
        # safe point at which to extend that proof.
        self._write_heartbeat("READY", identity)
        return {
            "text": str(response.get("text") or ""),
            "blocks": list(response.get("blocks") or []),
            "engine": "paddleocr",
            "engine_version": identity["paddleocr_version"],
            "requested_device": identity["requested_device"],
            "actual_device": identity["actual_device"],
            "runtime_identity": identity,
        }

    def close(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass
            wait = getattr(self._process, "wait", None)
            if callable(wait):
                try:
                    wait(timeout=5)
                except subprocess.TimeoutExpired:
                    kill = getattr(self._process, "kill", None)
                    if callable(kill):
                        try:
                            kill()
                            wait(timeout=5)
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                except OSError:
                    pass
            self._process = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._process_for_request()
        stdin = getattr(process, "stdin", None)
        stdout = getattr(process, "stdout", None)
        if stdin is None or stdout is None:
            self._fail("OCR_PROTOCOL_UNAVAILABLE")
        try:
            stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stdin.flush()
            line = stdout.readline()
        except (OSError, ValueError) as exc:
            self._fail("OCR_WORKER_IO_FAILED", str(exc))
        if not line:
            self._fail("OCR_WORKER_EXITED")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self._fail("OCR_PROTOCOL_INVALID", str(exc))
        if not isinstance(response, dict) or not response.get("ok"):
            self._fail(str(response.get("code") or "OCR_WORKER_FAILED"))
        return response

    def _process_for_request(self) -> Any:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self._python_path:
            self._fail("OCR_PYTHON_NOT_CONFIGURED")
        command = [self._python_path, "-m", "stock_content.adapters.media.ocr_worker"]
        self._process = self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Paddle/PaddleX emits verbose progress logs on stderr.  stderr
            # is not part of the JSON protocol and leaving an unread PIPE can
            # block a healthy worker when its OS pipe buffer fills.  Discard
            # it instead, which also prevents dependency diagnostics from
            # reaching the service process or its logs.
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=_ocr_worker_environment(self._python_path, self._device, self._require_gpu),
        )
        return self._process

    def _fail(self, code: str, message: str = "") -> None:
        self._write_heartbeat(code, self._runtime_identity)
        raise OcrRuntimeError(code, message)

    def _write_heartbeat(self, code: str, identity: dict[str, str]) -> None:
        if not self._heartbeat_path:
            return
        path = Path(self._heartbeat_path)
        payload = {
            "profile": "ocr",
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "health_code": code,
            "requested_device": self._device,
            "actual_device": identity.get("actual_device", ""),
            "runtime_identity": identity,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass


def _runtime_identity(response: dict[str, Any]) -> dict[str, str]:
    identity = _reported_identity(response)
    if not identity:
        raise OcrRuntimeError("OCR_RUNTIME_IDENTITY_MISSING")
    required = {
        "requested_device",
        "actual_device",
        "paddle_version",
        "paddleocr_version",
        "compiled_cuda",
        "cuda_version",
        "cudnn_version",
        "device_count",
    }
    if required - set(identity):
        raise OcrRuntimeError("OCR_RUNTIME_IDENTITY_MISSING")
    if identity["requested_device"] != "gpu:0" or not identity["actual_device"].lower().startswith("gpu:0"):
        raise OcrRuntimeError("OCR_DEVICE_MISMATCH")
    if identity["compiled_cuda"].lower() != "true" or int(identity["device_count"]) < 1:
        raise OcrRuntimeError("OCR_GPU_UNAVAILABLE")
    return identity


def _reported_identity(response: dict[str, Any]) -> dict[str, str]:
    value = response.get("runtime_identity")
    return {str(key): str(item) for key, item in value.items() if item is not None} if isinstance(value, dict) else {}


def _ocr_worker_environment(python_path: str, device: str, require_gpu: bool) -> dict[str, str]:
    """Pass a minimal non-secret environment; never inherit service credentials."""
    executable = Path(python_path).resolve()
    paths = [str(executable.parent), str(executable.parent.parent)]
    paths.extend(part for part in os.getenv("CONTENT_OCR_RUNTIME_PATH", "").split(os.pathsep) if part)
    environment = {
        "PATH": os.pathsep.join(dict.fromkeys(paths)),
        "CONTENT_OCR_DEVICE": device,
        "CONTENT_OCR_REQUIRE_GPU": "true" if require_gpu else "false",
        "PYTHONUNBUFFERED": "1",
    }
    # The parent service deliberately does not inherit LD_LIBRARY_PATH. Pass
    # only an explicit OCR-native lookup list so Paddle can locate CUDA and
    # cuDNN under Linux without admitting Torch libraries from the video/ASR
    # environment.
    library_paths = [part for part in os.getenv("CONTENT_OCR_LIBRARY_PATH", "").split(os.pathsep) if part]
    if library_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(library_paths))
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME"):
        if os.getenv(name):
            environment[name] = str(os.environ[name])
    environment.update(_ocr_model_cache_environment())
    return environment


def _ocr_model_cache_environment() -> dict[str, str]:
    """Locate Paddle/PaddleX caches without forwarding the user environment.

    PaddleOCR imports ModelScope, which calls ``Path.home()`` even when all
    OCR weights are already cached.  The worker intentionally receives a
    scrubbed environment, so pass only explicit cache locations.  Deployments
    can isolate those caches with ``CONTENT_OCR_CACHE_HOME``.
    """

    configured = os.getenv("CONTENT_OCR_CACHE_HOME", "").strip()
    profile = os.getenv("USERPROFILE", "").strip()
    if configured:
        cache_home = Path(configured)
        home = Path(profile) if profile else cache_home.parent
    elif profile:
        cache_home = Path(profile) / ".paddlex"
        home = Path(profile)
    elif local_app_data := os.getenv("LOCALAPPDATA", "").strip():
        cache_home = Path(local_app_data) / "stock_content" / "ocr-cache"
        home = Path(local_app_data).parent.parent
    else:
        # ModelScope must not call Path.home() in a profile-less service.
        cache_home = Path(tempfile.gettempdir()) / "stock_content" / "ocr-cache"
        home = cache_home.parent
    return {
        # ModelScope evaluates its Path.home()-based default eagerly even when
        # MODELSCOPE_CACHE is supplied.  On Windows, pathlib resolves home
        # from USERPROFILE rather than HOME, so both carry only this path.
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PADDLE_PDX_CACHE_HOME": str(cache_home),
        "MODELSCOPE_CACHE": str(cache_home / "modelscope"),
    }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["OcrRuntimeError", "PaddleOcrEngine"]

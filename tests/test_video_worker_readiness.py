from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from stock_content.api.readiness import _video_components, _video_worker_heartbeat
from stock_content.application.readiness_service import ReadinessDependencies
from stock_content.application.source_resolution_service import credential_allowlist_from_environment
from stock_content.workers import content_worker


def _heartbeat(
    *, observed_at: datetime | None = None, actual_device: str = "gpu:0", credential_ref: str = "xiaoe-state"
):
    return {
        "schema": "video-worker-readiness.v1",
        "profile": "video",
        "health_code": "READY",
        "observed_at": (observed_at or datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
        "xiaoe_page": {
            "enabled": True,
            "ready": True,
            "credential_ref": credential_ref,
            "credential_provider": "file-secret",
            "materializer_identity": "XiaoeMaterializer.local.v1",
        },
        "frame_extractor": {"ready": True, "identity": "FfmpegFrameExtractor.extract_targeted.v1"},
        "ocr": {"health_code": "READY", "requested_device": "gpu:0", "actual_device": actual_device},
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _configure(monkeypatch, path: Path) -> None:
    monkeypatch.setenv("CONTENT_VIDEO_WORKER_HEARTBEAT_FILE", str(path))
    monkeypatch.setenv("CONTENT_VIDEO_HEARTBEAT_MAX_AGE_SECONDS", "120")
    monkeypatch.setenv("CONTENT_XIAOE_PAGE_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CONTENT_XIAOE_CREDENTIAL_REF", "xiaoe-state")
    monkeypatch.setenv("CONTENT_XIAOE_CREDENTIAL_PROVIDER", "file-secret")
    monkeypatch.setenv("CONTENT_INGESTION_CREDENTIAL_REFS", "other-ref")
    monkeypatch.setenv("CONTENT_INGESTION_CREDENTIAL_PROVIDERS", "other-provider")


def test_api_readiness_uses_worker_proof_not_storage_state_file(tmp_path, monkeypatch):
    heartbeat = tmp_path / "video-worker-heartbeat.json"
    _configure(monkeypatch, heartbeat)
    # The API deliberately has no state-file mount.  A valid worker proof must
    # still make the Xiaoe browser component ready.
    monkeypatch.delenv("CONTENT_XIAOE_STORAGE_STATE_FILE", raising=False)
    _write(heartbeat, _heartbeat())

    ready, proof = _video_worker_heartbeat(str(heartbeat))
    application = type(
        "App",
        (),
        {"_tasks": type("Tasks", (), {"_sessions": object(), "claim_pending": lambda *_args: None})()},
    )()
    components = _video_components(ReadinessDependencies(postgres_ok=True), application)

    assert ready is True
    assert proof["health_code"] == "READY"
    assert "credential_ref" not in proof
    assert components["xiaoe_browser"]["ready"] is True


def test_video_worker_heartbeat_fails_closed_when_stale_cpu_or_allowlist_mismatched(tmp_path, monkeypatch):
    heartbeat = tmp_path / "video-worker-heartbeat.json"
    _configure(monkeypatch, heartbeat)

    _write(heartbeat, _heartbeat(observed_at=datetime.now(UTC) - timedelta(seconds=121)))
    assert _video_worker_heartbeat(str(heartbeat))[0] is False

    _write(heartbeat, _heartbeat(actual_device="cpu"))
    assert _video_worker_heartbeat(str(heartbeat))[0] is False

    _write(heartbeat, _heartbeat(credential_ref="not-allowlisted"))
    assert _video_worker_heartbeat(str(heartbeat))[0] is False


def test_page_state_reference_and_provider_are_added_to_api_allowlist(monkeypatch):
    monkeypatch.setenv("CONTENT_INGESTION_CREDENTIAL_REFS", "other-ref")
    monkeypatch.setenv("CONTENT_INGESTION_CREDENTIAL_PROVIDERS", "other-provider")
    monkeypatch.setenv("CONTENT_XIAOE_CREDENTIAL_REF", "xiaoe-page-state")
    monkeypatch.setenv("CONTENT_XIAOE_CREDENTIAL_PROVIDER", "worker-secret-provider")

    references, providers = credential_allowlist_from_environment()

    assert {"other-ref", "xiaoe-page-state"} <= references
    assert {"other-provider", "worker-secret-provider"} <= providers


def test_worker_heartbeat_copies_only_nonsecret_ocr_device_attestation(tmp_path, monkeypatch):
    state = tmp_path / "private-state.json"
    state.write_text('{"cookies":[{"value":"do-not-copy"}]}', encoding="utf-8")
    ocr = tmp_path / "ocr-heartbeat.json"
    _write(
        ocr,
        {
            "health_code": "READY",
            "requested_device": "gpu:0",
            "actual_device": "gpu:0",
        },
    )
    monkeypatch.setenv("CONTENT_XIAOE_PAGE_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CONTENT_XIAOE_STORAGE_STATE_FILE", str(state))
    monkeypatch.setenv("CONTENT_XIAOE_CREDENTIAL_REF", "xiaoe-state")
    monkeypatch.setenv("CONTENT_XIAOE_CREDENTIAL_PROVIDER", "file-secret")
    monkeypatch.setenv("CONTENT_XIAOE_PAGE_URL_TEMPLATE", "https://example.xiaoeknow.com/{source_ref}")
    monkeypatch.setenv("CONTENT_OCR_HEARTBEAT_FILE", str(ocr))
    monkeypatch.setattr(content_worker, "page_resolver_from_environment", lambda: object())
    monkeypatch.setattr(content_worker.shutil, "which", lambda _binary: "/usr/bin/ffmpeg")

    payload = content_worker._video_readiness_payload()

    assert payload["health_code"] == "READY"
    assert payload["ocr"] == {"health_code": "READY", "requested_device": "gpu:0", "actual_device": "gpu:0"}
    assert "do-not-copy" not in json.dumps(payload)

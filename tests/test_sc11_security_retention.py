from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from stock_content.adapters.retention.in_memory_tombstones import InMemoryTombstoneRepository
from stock_content.api.errors import install_error_handlers
from stock_content.application.retention_service import RetentionService
from stock_content.application.service import _assert_checkpoint_has_no_signed_url
from stock_content.application.source_resolution_service import normalize_command
from stock_content.domain.lineage import ContentSnapshot
from stock_content.domain.models import ContentTask
from stock_content.domain.retention import RetentionCandidate, RetentionClass, RetentionPolicy
from stock_content.domain.security_redaction import redact_for_serialization, redact_text
from stock_content.domain.source_materialization import CredentialReference

NOW = datetime(2026, 9, 7, tzinfo=UTC)


def _candidate(artifact_class: RetentionClass = RetentionClass.RAW_MEDIA) -> RetentionCandidate:
    return RetentionCandidate("artifact-1", artifact_class, "a" * 64, "b" * 64, "lineage-1", NOW - timedelta(days=8))


def test_api_reference_allowlist_and_credential_hash_boundary() -> None:
    reference = CredentialReference(credential_ref="xiaoe-session-a", provider="file-secret")
    command = normalize_command(
        source_type="xiaoe", source_ref="course-1/lesson-1", credential_ref=reference,
        allowed_credential_refs=frozenset({"xiaoe-session-a"}), allowed_credential_providers=frozenset({"file-secret"}),
    )
    assert command.credential_ref_hash and "xiaoe-session-a" not in json.dumps(asdict(command))
    with pytest.raises(ValueError, match="not allowlisted"):
        normalize_command(
            source_type="xiaoe", source_ref="course-1/lesson-1", credential_ref=reference,
            allowed_credential_refs=frozenset(), allowed_credential_providers=frozenset({"file-secret"}),
        )


def test_redaction_and_checkpoint_guard_cover_api_log_db_checkpoint_snapshot_and_bundle_surfaces() -> None:
    canary_url = "https://cdn.example/media.m3u8?signature=secret-url-canary#fragment"
    canary_cookie = "secret-cookie-canary"
    payload = {"locator": canary_url, "Authorization": "Bearer secret-token-canary", "Cookie": canary_cookie}
    redacted = redact_for_serialization(payload)
    assert canary_url not in json.dumps(redacted) and canary_cookie not in json.dumps(redacted)
    assert redact_text(canary_url) == "https://cdn.example/media.m3u8"
    task = ContentTask("task-1", "xiaoe", canary_url)
    snapshot = ContentSnapshot("snapshot-1", "xiaoe", canary_url, "hash")
    for surface in (task.to_dict(), snapshot.to_dict(), redacted):
        text = json.dumps(surface, default=str)
        assert "secret-url-canary" not in text and "secret-cookie-canary" not in text
    with pytest.raises(ValueError, match="CHECKPOINT_CONTAINS_SECRET_OR_SIGNED_LOCATOR"):
        _assert_checkpoint_has_no_signed_url({"resolver_url": canary_url})

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/failure")
    def failure():
        raise HTTPException(400, detail={"code": "BAD_REQUEST", "details": payload})

    response = TestClient(app).get("/failure")
    assert "secret-url-canary" not in response.text and "secret-cookie-canary" not in response.text


def test_retention_defaults_validation_dry_run_and_idempotent_audit_tombstone() -> None:
    policy = RetentionPolicy.from_environment()
    assert policy.days == {
        RetentionClass.RAW_MEDIA: 7, RetentionClass.SUBTITLE: 30,
        RetentionClass.TRANSCRIPT: 90, RetentionClass.KNOWLEDGE: 3650,
    }
    with pytest.raises(ValueError):
        RetentionPolicy({item: 0 for item in RetentionClass})
    repository = InMemoryTombstoneRepository()
    service = RetentionService(policy, repository)
    planned = service.plan(_candidate(), now=NOW)
    assert planned.dry_run and planned.action == "AUDIT_TOMBSTONE_PLANNED" and not repository.items
    first = service.plan(_candidate(), now=NOW, dry_run=False)
    second = service.plan(_candidate(), now=NOW + timedelta(days=1), dry_run=False)
    assert first.tombstone == second.tombstone and len(repository.items) == 1
    serialized = json.dumps(first.tombstone.to_dict())
    assert "locator" not in serialized and "secret" not in serialized
    assert service.plan(_candidate(RetentionClass.KNOWLEDGE), now=NOW).action == "KEEP"


def test_artifact_scanner_fails_without_echo_and_ignores_runtime_media(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "clean.json").write_text('{"url":"https://example.test/path"}', encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "scan_artifacts.py"
    clean = subprocess.run([sys.executable, str(script), str(root)], text=True, capture_output=True, check=False)
    assert clean.returncode == 0
    (root / "bad.json").write_text("token=secret-token-canary", encoding="utf-8")
    failed = subprocess.run([sys.executable, str(script), str(root)], text=True, capture_output=True, check=False)
    assert failed.returncode == 1 and "secret-token-canary" not in failed.stdout + failed.stderr
    cache = root / "media"
    cache.mkdir()
    (cache / "ignored.txt").write_text("token=secret-token-canary", encoding="utf-8")
    (root / "bad.json").unlink()
    assert subprocess.run([sys.executable, str(script), str(root)], check=False).returncode == 0


def test_video_runtime_policy_is_explicit_configuration_not_host_enforcement_claim() -> None:
    root = Path(__file__).parents[1]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (root / "docker" / "Dockerfile.video").read_text(encoding="utf-8")
    policy = (root / "docker" / "video-worker-egress-policy.yaml").read_text(encoding="utf-8")
    for expected in ("read_only: true", "user: \"10001:10001\"", "pids: 128", "xiaoe-storage-state"):
        assert expected in compose
    assert "USER content" in dockerfile and "mkdir /work /data" in dockerfile
    for expected in ("egress-proxy-or-cni-required", "loopback", "cloud-metadata", "secret-store", "browser_limits"):
        assert expected in policy

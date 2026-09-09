"""Run one authorized Xiaoe ingestion through the public HTTP and SQL boundaries.

This is an operator runner, not a test fixture: it never supplies transcript,
frames, claims, knowledge, or Bundle rows.  The deployed video worker must
materialize and process the licensed page using its mounted Playwright state.
Only safe identifiers and aggregate SQL readback evidence are written to the
requested report file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

# A source checkout can run this operator command before it is installed as an
# editable package.  Prefer this checkout over another EPIC worktree that may
# be installed in the invoking shell.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from stock_content.adapters.postgres.models import (  # noqa: E402
    ClaimArtifactMemberRow,
    ClaimOccurrenceRow,
    ClaimStateEventRow,
    ContentArtifactRow,
    ContentKnowledgeBundleRow,
    ContentSnapshotRow,
    ContentTaskEffectRow,
    ContentTaskRow,
    FinancialClaimRow,
    KnowledgeUnitRow,
    OcrEvidenceRow,
    SignalOutboxRow,
    SourceArtifactMetadataRow,
    VideoAssetRow,
    VideoFrameRow,
    VideoSegmentRow,
    VisionEvidenceRow,
)
from stock_content.application.source_resolution_service import canonical_xiaoe_page_ref  # noqa: E402


class E2EError(RuntimeError):
    """A stable, non-secret runner failure."""


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise E2EError(f"{name} is required")
    return value


def _token() -> str:
    path = Path(_required("CONTENT_E2E_SERVICE_API_KEY_FILE"))
    if not path.is_file() or path.is_symlink():
        raise E2EError("CONTENT_E2E_SERVICE_API_KEY_FILE must be a regular file")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise E2EError("CONTENT_E2E_SERVICE_API_KEY_FILE is empty")
    return value


def _headers(token: str, *, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Caller-Service": os.getenv("CONTENT_E2E_CALLER_SERVICE", "stock_agent"),
        "Content-Type": "application/json",
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _body(source_ref: str, idempotency_key: str) -> dict[str, Any]:
    return {
        "source_type": "xiaoe",
        "source_ref": source_ref,
        "part": 1,
        "transcript_policy": "subtitle_first",
        "options": {"language": os.getenv("CONTENT_E2E_LANGUAGE", "zh")},
        "credential_ref": {
            "credential_ref": os.getenv("CONTENT_XIAOE_CREDENTIAL_REF", "xiaoe-storage-state"),
            "provider": "file-secret",
        },
        "idempotency_key": idempotency_key,
    }


def _response_json(response: httpx.Response, action: str) -> dict[str, Any]:
    if response.status_code >= 400:
        # Do not relay response bodies: an upstream proxy might echo a URL or
        # authorization header.  HTTP status and action are enough to debug
        # the report without enlarging the secret boundary.
        raise E2EError(f"{action} failed with HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise E2EError(f"{action} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise E2EError(f"{action} returned a non-object JSON response")
    return payload


def _poll(
    client: httpx.Client, base_url: str, headers: dict[str, str], task_id: str, timeout_seconds: int
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        task = _response_json(client.get(f"{base_url}/v1/content/ingestions/{task_id}", headers=headers), "task poll")
        status = str(task.get("status") or "")
        if status == "SUCCEEDED":
            return task
        if status in {"FAILED", "CANCELLED", "LEASE_LOST"}:
            raise E2EError(f"video task reached terminal status {status}")
        if time.monotonic() >= deadline:
            raise E2EError("video task timed out before a terminal state")
        time.sleep(min(5.0, max(0.2, float(os.getenv("CONTENT_E2E_POLL_SECONDS", "2")))))


def _database_evidence(
    database_url: str, *, task_id: str, video_id: str, snapshot_id: str, bundle_id: str
) -> dict[str, int]:
    """Read committed rows only; the runner never writes to the database."""
    sessions = sessionmaker(bind=create_engine(database_url, future=True), expire_on_commit=False)
    with sessions() as session:
        snapshot = session.get(ContentSnapshotRow, snapshot_id)
        artifact_map = dict(snapshot.artifact_ids or {}) if snapshot is not None else {}
        artifact_ids = list(artifact_map.values())
        claim_ids = list(
            session.scalars(
                select(ClaimArtifactMemberRow.claim_id).where(
                    ClaimArtifactMemberRow.artifact_id == str(artifact_map.get("claims") or "__missing__")
                )
            )
        )
        return {
            "task": int(
                session.scalar(
                    select(func.count()).select_from(ContentTaskRow).where(ContentTaskRow.task_id == task_id)
                )
                or 0
            ),
            "task_effect": int(
                session.scalar(
                    select(func.count())
                    .select_from(ContentTaskEffectRow)
                    .where(ContentTaskEffectRow.task_id == task_id)
                )
                or 0
            ),
            "source": int(
                session.scalar(
                    select(func.count())
                    .select_from(SourceArtifactMetadataRow)
                    .where(SourceArtifactMetadataRow.artifact_id == str(artifact_map.get("source") or "__missing__"))
                )
                or 0
            ),
            "video": int(
                session.scalar(
                    select(func.count()).select_from(VideoAssetRow).where(VideoAssetRow.video_id == video_id)
                )
                or 0
            ),
            "transcript_segment": int(
                session.scalar(
                    select(func.count()).select_from(VideoSegmentRow).where(VideoSegmentRow.video_id == video_id)
                )
                or 0
            ),
            "knowledge": int(
                session.scalar(
                    select(func.count()).select_from(KnowledgeUnitRow).where(KnowledgeUnitRow.video_id == video_id)
                )
                or 0
            ),
            "frame": int(
                session.scalar(
                    select(func.count()).select_from(VideoFrameRow).where(VideoFrameRow.video_id == video_id)
                )
                or 0
            ),
            "ocr": int(
                session.scalar(
                    select(func.count())
                    .select_from(OcrEvidenceRow)
                    .join(VideoFrameRow)
                    .where(VideoFrameRow.video_id == video_id)
                )
                or 0
            ),
            "vision": int(
                session.scalar(
                    select(func.count())
                    .select_from(VisionEvidenceRow)
                    .join(VideoFrameRow)
                    .where(VideoFrameRow.video_id == video_id)
                )
                or 0
            ),
            "snapshot": int(
                session.scalar(
                    select(func.count())
                    .select_from(ContentSnapshotRow)
                    .where(ContentSnapshotRow.content_snapshot_id == snapshot_id)
                )
                or 0
            ),
            "snapshot_artifact": int(
                session.scalar(
                    select(func.count())
                    .select_from(ContentArtifactRow)
                    .where(ContentArtifactRow.artifact_id.in_(artifact_ids or ["__missing__"]))
                )
                or 0
            ),
            "claim": int(
                session.scalar(
                    select(func.count())
                    .select_from(FinancialClaimRow)
                    .where(FinancialClaimRow.claim_id.in_(claim_ids or ["__missing__"]))
                )
                or 0
            ),
            "occurrence": int(
                session.scalar(
                    select(func.count())
                    .select_from(ClaimOccurrenceRow)
                    .where(ClaimOccurrenceRow.claim_id.in_(claim_ids or ["__missing__"]))
                )
                or 0
            ),
            "claim_state": int(
                session.scalar(
                    select(func.count())
                    .select_from(ClaimStateEventRow)
                    .where(ClaimStateEventRow.claim_id.in_(claim_ids or ["__missing__"]))
                )
                or 0
            ),
            "outbox": int(
                session.scalar(
                    select(func.count())
                    .select_from(SignalOutboxRow)
                    .where(SignalOutboxRow.content_snapshot_id == snapshot_id)
                )
                or 0
            ),
            "bundle": int(
                session.scalar(
                    select(func.count())
                    .select_from(ContentKnowledgeBundleRow)
                    .where(ContentKnowledgeBundleRow.bundle_id == bundle_id)
                )
                or 0
            ),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = _required("CONTENT_E2E_CONTENT_BASE_URL").rstrip("/")
    token = _token()
    source_ref = args.source_ref or _required("CONTENT_E2E_XIAOE_SOURCE_REF")
    # Accept the human page URL for convenience but send only the safe,
    # durable identity over HTTP.  A pre-normalized product/lesson pair is
    # equally accepted for automation.
    if source_ref.startswith(("https://", "http://")):
        source_ref = canonical_xiaoe_page_ref(source_ref)
    timeout_seconds = args.timeout_seconds
    idempotency_key = args.idempotency_key
    headers = _headers(token, idempotency_key=idempotency_key)
    with httpx.Client(timeout=httpx.Timeout(15.0, read=60.0)) as client:
        video_ready = _response_json(client.get(f"{base_url}/health/video-ingestion-ready"), "video readiness")
        if not video_ready.get("ready"):
            raise E2EError("video ingestion readiness is not ready")
        bundle_ready = _response_json(
            client.get(f"{base_url}/health/knowledge-bundle-ready", headers=headers), "Bundle readiness"
        )
        if not bundle_ready.get("ready"):
            raise E2EError("Bundle readiness is not ready")
        queued = _response_json(
            client.post(f"{base_url}/v1/content/ingestions", headers=headers, json=_body(source_ref, idempotency_key)),
            "ingestion",
        )
        task_id = str(queued.get("task_id") or "")
        if not task_id:
            raise E2EError("ingestion response omitted task_id")
        task = _poll(client, base_url, headers, task_id, timeout_seconds)
        result = dict(task.get("result") or {})
        video_id = str(result.get("video_id") or "")
        snapshot_id = str(result.get("content_snapshot_id") or "")
        if not video_id or not snapshot_id:
            raise E2EError("successful task omitted video_id or content_snapshot_id")
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        bundle_body = {
            "content_snapshot_id": snapshot_id,
            "query": "xiaoe investment knowledge extraction",
            "symbol": "UNSPECIFIED",
            "business_as_of": now,
            "knowledge_as_of": now,
            "availability_as_of": now,
            "minimum_support_status": "SOURCE_SUPPORTED",
            "max_items": 100,
            "policy": "PUBLIC_STRICT",
            "policy_version": "content-bundle-policy.v2",
            "contract_version": "content-knowledge-bundle.v2",
        }
        bundle = _response_json(
            client.post(f"{base_url}/v1/content/knowledge-bundles", headers=headers, json=bundle_body), "Bundle create"
        )
        bundle_id = str(bundle.get("bundle_id") or "")
        if not bundle_id:
            raise E2EError("Bundle response omitted bundle_id")
        reread = _response_json(
            client.get(f"{base_url}/v1/content/knowledge-bundles/{bundle_id}", headers=headers), "Bundle readback"
        )
    if reread.get("bundle_hash") != bundle.get("bundle_hash"):
        raise E2EError("immutable Bundle readback hash mismatch")
    database_evidence = _database_evidence(
        _required("CONTENT_E2E_DATABASE_URL"),
        task_id=task_id,
        video_id=video_id,
        snapshot_id=snapshot_id,
        bundle_id=bundle_id,
    )
    if any(value <= 0 for value in database_evidence.values()):
        raise E2EError("database readback is incomplete")
    return {
        "runner": "xiaoe-http-e2e.v1",
        "executed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_type": "xiaoe",
        "source_ref": source_ref,
        "task_id": task_id,
        "task_status": task.get("status"),
        "video_id": video_id,
        "content_snapshot_id": snapshot_id,
        "bundle_id": bundle_id,
        "bundle_hash": bundle.get("bundle_hash"),
        "bundle_item_count": len(bundle.get("items") or []),
        "database_readback": database_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one authorized Xiaoe HTTP/queue/PostgreSQL E2E")
    parser.add_argument("--report", type=Path, required=True, help="non-secret JSON report path")
    parser.add_argument("--source-ref", help="stable product/lesson identity or public Xiaoe course page URL")
    parser.add_argument("--idempotency-key", default=f"xiaoe-http-e2e-{int(time.time())}")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    try:
        report = run(args)
    except E2EError as exc:
        raise SystemExit(f"XIAOE_HTTP_E2E_FAILED: {exc}") from exc
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"task_id": report["task_id"], "bundle_id": report["bundle_id"], "status": "SUCCEEDED"}))


if __name__ == "__main__":
    main()

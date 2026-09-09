from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentTaskRow
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.api.main import create_app
from stock_content.application.service import ContentApplication
from stock_content.application.source_resolution_service import normalize_command, request_hash_for
from stock_content.domain.source_materialization import CredentialReference
from stock_content.ports.repositories import IdempotencyConflict


class _RecordingApplication:
    def __init__(self) -> None:
        self.commands = []

    def enqueue_ingestion(self, command):
        self.commands.append(command)
        return {"task_id": f"task-{len(self.commands)}", "status": "PENDING"}

    def get_task(self, task_id):
        return {"task_id": task_id, "status": "PENDING"}


class _TestAuthorizer:
    """Explicit test boundary; production uses file-backed ServiceAuthorizer."""

    def configured(self):
        return True

    def authorize(self, _authorization, caller, *, required_caller=None):
        return required_caller or caller or "stock_agent"


def _client(application):
    return TestClient(create_app(application, authorizer=_TestAuthorizer()))


class _NullPipeline:
    _stages: list[object] = []


class _NullRepository:
    pass


def _application(tasks) -> ContentApplication:
    return ContentApplication(tasks, _NullRepository(), _NullRepository(), _NullRepository(), _NullPipeline())


def test_canonical_and_legacy_bilibili_use_one_command_shape():
    application = _RecordingApplication()
    client = _client(application)
    canonical = client.post(
        "/v1/content/ingestions",
        json={"source_type": "bilibili", "source_ref": "BV1abc", "part": 1,
              "transcript_policy": "subtitle_first", "options": {"language": "zh"}},
    )
    legacy = client.post("/api/v1/videos/bilibili/ingest", json={"bv_id": "BV1abc"})
    assert canonical.status_code == legacy.status_code == 200
    assert application.commands[0].source_type == application.commands[1].source_type == "bilibili"
    assert application.commands[0].canonical_source_ref == application.commands[1].canonical_source_ref == "BV1abc"
    assert application.commands[1].part == 1
    assert application.commands[1].transcript_policy == "subtitle_first"


def test_legacy_xiaoe_signed_locator_requires_secret_reference_and_never_records_url(monkeypatch):
    application = _RecordingApplication()
    client = _client(application)
    monkeypatch.setenv("CONTENT_XIAOE_HLS_CREDENTIAL_REF", "authorized-xiaoe-hls")
    secret_url = "https://m.xiaoe-tech.com/media/lesson.m3u8?signature=secret-canary#fragment"
    response = client.post(
        "/api/v1/videos/xiaoe/ingest",
        json={
            "m3u8_url": secret_url,
            "credential_ref": {"credential_ref": "authorized-xiaoe-hls", "provider": "file-secret"},
        },
    )
    assert response.status_code == 200
    command = application.commands[0]
    assert command.source_type == "xiaoe_hls"
    assert command.canonical_source_ref == "https://m.xiaoe-tech.com/media/lesson.m3u8"
    assert "secret-canary" not in repr(command)
    assert command.credential_ref_hash and command.locator_secret_hash is None
    rejected = client.post("/api/v1/videos/xiaoe/ingest", json={"m3u8_url": secret_url})
    assert rejected.status_code == 422


def test_xiaoe_page_url_is_reduced_to_stable_identity_before_durable_queue(monkeypatch, tmp_path):
    """HTTP ingress persists an idempotent, secret-free page task.

    This intentionally stops at the durable queue boundary.  The production
    HTTP runner exercises the separately deployed media worker; this test
    proves a retry cannot create a second logical task before that worker
    claims it.
    """
    monkeypatch.setenv("CONTENT_INGESTION_CREDENTIAL_REFS", "xiaoe-storage-state")
    database = Database(f"sqlite:///{tmp_path / 'xiaoe-ingestion.db'}")
    database.create_schema()
    application = _application(PostgresContentTaskRepository(database.session_factory))
    client = _client(application)
    page_url = (
        "https://appaoswidcd4711.h5.xiaoeknow.com/p/course/video/"
        "v_6a9e9ff1e4b0694c5c07a1f1?product_id=p_6a5ed542e4b0694c352d9382"
    )
    body = {
        "source_type": "xiaoe",
        "source_ref": page_url,
        "part": 1,
        "transcript_policy": "subtitle_first",
        "options": {"language": "zh"},
        "credential_ref": {"credential_ref": "xiaoe-storage-state", "provider": "file-secret"},
    }
    first = client.post("/v1/content/ingestions", json=body, headers={"Idempotency-Key": "xiaoe-http-e2e"})
    second = client.post("/v1/content/ingestions", json=body, headers={"Idempotency-Key": "xiaoe-http-e2e"})
    assert first.status_code == second.status_code == 200
    assert first.json()["task_id"] == second.json()["task_id"]
    with database.session_factory() as session:
        rows = list(session.scalars(select(ContentTaskRow)))
    assert len(rows) == 1
    assert rows[0].source_type == "xiaoe"
    assert rows[0].source_ref == "p_6a5ed542e4b0694c352d9382/v_6a9e9ff1e4b0694c5c07a1f1"
    assert page_url not in repr(rows[0])
    assert "xiaoe-storage-state" not in repr(rows[0])


def test_bilibili_xor_and_idempotency_header_body_errors_are_stable():
    client = _client(_RecordingApplication())
    invalid = client.post(
        "/api/v1/videos/bilibili/ingest",
        json={"url": "https://www.bilibili.com/video/BV1a", "bv_id": "BV1a"},
    )
    mismatch = client.post(
        "/v1/content/ingestions", headers={"Idempotency-Key": "header"},
        json={"source_type": "bilibili", "source_ref": "BV1a", "part": 1,
              "transcript_policy": "subtitle_first", "options": {}, "idempotency_key": "body"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "INVALID_INGESTION_REQUEST"
    assert mismatch.status_code == 409
    assert mismatch.json()["error"]["code"] == "IDEMPOTENCY_KEY_MISMATCH"


@pytest.mark.parametrize(
    "source_ref,expected",
    [
        ("av170001", "av170001"),
        ("https://www.bilibili.com/video/av170001", "https://www.bilibili.com/video/av170001"),
        ("https://b23.tv/fixture", "https://b23.tv/fixture"),
        ("https://www.bilibili.com/video/BV1fixture?p=2", "https://www.bilibili.com/video/BV1fixture?p=2"),
    ],
)
def test_canonical_bilibili_admits_worker_resolver_inputs_without_ingress_redirect(source_ref, expected):
    command = normalize_command(source_type="bilibili", source_ref=source_ref, part=1)
    assert command.canonical_source_ref == expected
    if source_ref.endswith("?p=2"):
        assert command.part == 2


def test_bilibili_cookie_reference_is_allowlisted_and_is_not_queued_as_a_cookie(monkeypatch):
    monkeypatch.setenv("CONTENT_BILIBILI_CREDENTIAL_REF", "authorized-bili-cookie")
    application = _RecordingApplication()
    client = _client(application)
    response = client.post(
        "/v1/content/ingestions",
        json={
            "source_type": "bilibili", "source_ref": "BV1fixture", "part": 1,
            "transcript_policy": "subtitle_first", "options": {},
            "credential_ref": {"credential_ref": "authorized-bili-cookie", "provider": "file-secret"},
        },
    )
    assert response.status_code == 200
    queued = application.commands[0]
    assert queued.credential_ref_hash
    assert "authorized-bili-cookie" not in repr(queued)


def test_request_hash_is_deterministic_and_repository_reserves_key(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'ingestion.db'}")
    database.create_schema()
    app = _application(PostgresContentTaskRepository(database.session_factory))
    credential = CredentialReference(credential_ref="vault://content/lesson", provider="vault")
    first = normalize_command(
        source_type="xiaoe", source_ref="course-1/lesson-1", part=1, transcript_policy="subtitle_first",
        options={"language": "zh"}, idempotency_key="same-key", credential_ref=credential,
    )
    duplicate = normalize_command(
        source_type="xiaoe", source_ref="course-1/lesson-1", part=1, transcript_policy="subtitle_first",
        options={"language": "zh"}, idempotency_key="same-key", credential_ref=credential,
    )
    assert request_hash_for(first) == request_hash_for(duplicate)
    created = app.enqueue_ingestion(first)
    assert app.enqueue_ingestion(duplicate)["task_id"] == created["task_id"]
    changed = normalize_command(
        source_type="xiaoe", source_ref="course-1/lesson-1", part=2, transcript_policy="subtitle_first",
        options={"language": "zh"}, idempotency_key="same-key", credential_ref=credential,
    )
    with pytest.raises(IdempotencyConflict):
        app.enqueue_ingestion(changed)
    with database.session_factory() as session:
        row = session.scalar(select(ContentTaskRow))
        serialized = repr({"source_ref": row.source_ref, "options": row.options, "credential": row.credential_ref_hash})
    assert "vault://content/lesson" not in serialized
    assert row.credential_ref_hash and row.source_identity_hash and row.request_hash


def test_migration_and_contract_are_explicitly_canonical_only():
    root = Path(__file__).parents[1]
    migration = (root / "migrations" / "030_content_ingestion_canonicalization.sql").read_text(encoding="utf-8")
    contract = (root / "contracts" / "content-ingestion.v1.json").read_text(encoding="utf-8")
    assert "legacy_unresolved" in migration
    assert "Cookie" not in migration and "signed_url" not in migration
    assert "credential_ref" in contract and "storage_state" not in contract


def test_epic043_numbered_schema_migrations_are_ddl_only():
    root = Path(__file__).parents[1]
    for name in ("030_content_ingestion_canonicalization.sql", "032_atomic_claim_grounding_projection.sql"):
        sql = (root / "migrations" / name).read_text(encoding="utf-8")
        statements = [
            statement.strip()
            for statement in re.sub(r"--[^\n]*", "", sql).split(";")
            if statement.strip()
        ]
        assert statements
        assert all(re.match(r"^(ALTER|CREATE)\b", statement, re.IGNORECASE) for statement in statements)
        assert not re.search(r"\b(?:UPDATE|INSERT|DELETE|MERGE)\b", "\n".join(statements), re.IGNORECASE)


def test_epic043_legacy_backfill_is_explicit_and_not_a_schema_bootstrap_step():
    root = Path(__file__).parents[1]
    operational = (root / "scripts" / "backfill_epic043_legacy_rows.py").read_text(encoding="utf-8")
    bootstrap = (root / "src" / "stock_content" / "adapters" / "postgres" / "migration_ledger.py").read_text(
        encoding="utf-8"
    )
    assert "--confirm" in operational
    assert "--database-url" in operational
    assert "backfill_epic043_legacy_rows" not in bootstrap

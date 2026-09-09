"""HTTP adapters for the canonical content-ingestion command."""
from __future__ import annotations

from typing import Callable, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from stock_content.application.source_resolution_service import (
    IngestionValidationError,
    canonical_bilibili_ref,
    canonical_xiaoe_hls_ref,
    canonical_xiaoe_page_ref,
    command_with_legacy_policy,
    credential_allowlist_from_environment,
    normalize_command,
)
from stock_content.domain.source_materialization import CredentialReference
from stock_content.ports.repositories import IdempotencyConflict


class CanonicalIngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["bilibili", "xiaoe"]
    source_ref: str = Field(min_length=1)
    credential_ref: CredentialReference | None = None
    part: int = Field(default=1, ge=1)
    transcript_policy: Literal["subtitle_first"] = "subtitle_first"
    options: dict = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=128)


class LegacyIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    bv_id: str | None = None
    m3u8_url: str | None = None
    credential_ref: CredentialReference | None = None
    options: dict = Field(default_factory=dict)


def _error(status: int, code: str, message: str) -> HTTPException:
    # Existing API errors use FastAPI's ``detail`` envelope. Keep that shape
    # while making ingestion failures programmatically stable and secret-free.
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _idempotency_key(header: str | None, body: str | None) -> str | None:
    if header and body and header != body:
        raise _error(409, "IDEMPOTENCY_KEY_MISMATCH", "header and body Idempotency-Key differ")
    return header or body


def _legacy_options(options: dict) -> tuple[dict, int, str, str | None]:
    options = dict(options or {})
    allowed = {
        "language", "part", "transcript_policy", "idempotency_key", "metadata", "transcript",
        "offline_fixture", "asr_model", "asr_model_version", "quant_market_snapshot_ids", "code_sha",
        "available_from", "trace_id",
    }
    if set(options) - allowed:
        raise _error(422, "INVALID_INGESTION_REQUEST", "unsupported legacy ingestion options")
    part = options.pop("part", 1)
    policy = options.pop("transcript_policy", "subtitle_first")
    body_key = options.pop("idempotency_key", None)
    return options, part, policy, body_key


def create_ingestions_router(application_for_request: Callable[[], object]) -> APIRouter:
    router = APIRouter()

    def enqueue(command):
        try:
            return application_for_request().enqueue_ingestion(command)
        except IngestionValidationError as exc:
            raise _error(422, exc.code, str(exc)) from exc
        except IdempotencyConflict as exc:
            raise _error(409, exc.code, str(exc)) from exc

    @router.post("/v1/content/ingestions")
    def create_ingestion(
        request: CanonicalIngestionRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        try:
            allowed_refs, allowed_providers = credential_allowlist_from_environment()
            command = normalize_command(
                source_type=request.source_type,
                source_ref=request.source_ref,
                part=request.part,
                transcript_policy=request.transcript_policy,
                options=request.options,
                idempotency_key=_idempotency_key(idempotency_key, request.idempotency_key),
                credential_ref=request.credential_ref,
                allowed_credential_refs=allowed_refs,
                allowed_credential_providers=allowed_providers,
                trace_id=http_request.state.trace_id,
                decision_id=http_request.state.decision_id,
            )
        except IngestionValidationError as exc:
            raise _error(422, exc.code, str(exc)) from exc
        return enqueue(command_with_legacy_policy(command))

    @router.get("/v1/content/ingestions/{task_id}")
    def get_ingestion(task_id: str) -> dict:
        payload = application_for_request().get_task(task_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="task not found")
        return payload

    @router.post("/api/v1/videos/bilibili/ingest")
    def ingest_bilibili(
        request: LegacyIngestRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        options, part, transcript_policy, body_key = _legacy_options(request.options)
        try:
            allowed_refs, allowed_providers = credential_allowlist_from_environment()
            command = normalize_command(
                source_type="bilibili",
                source_ref=canonical_bilibili_ref(url=request.url, bv_id=request.bv_id),
                part=part,
                transcript_policy=transcript_policy,
                options=options,
                idempotency_key=_idempotency_key(idempotency_key, body_key),
                credential_ref=request.credential_ref,
                allowed_credential_refs=allowed_refs,
                allowed_credential_providers=allowed_providers,
                trace_id=http_request.state.trace_id,
                decision_id=http_request.state.decision_id,
                allow_legacy_options=True,
            )
        except IngestionValidationError as exc:
            raise _error(422, exc.code, str(exc)) from exc
        return enqueue(command_with_legacy_policy(command))

    @router.post("/api/v1/videos/xiaoe/ingest")
    def ingest_xiaoe(
        request: LegacyIngestRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        options, part, transcript_policy, body_key = _legacy_options(request.options)
        try:
            allowed_refs, allowed_providers = credential_allowlist_from_environment()
            if bool(request.url) == bool(request.m3u8_url):
                raise IngestionValidationError("exactly one of url or m3u8_url is required")
            if request.url:
                source_type = "xiaoe"
                public_ref = canonical_xiaoe_page_ref(request.url)
                if request.credential_ref is None:
                    raise IngestionValidationError("xiaoe requires an allowlisted credential_ref")
            else:
                source_type = "xiaoe_hls"
                public_ref, locator_secret_input = canonical_xiaoe_hls_ref(request.m3u8_url)
                signed_locator = locator_secret_input is not None
                if signed_locator and request.credential_ref is None:
                    raise IngestionValidationError("signed Xiaoe HLS requires an allowlisted credential_ref")
            command = normalize_command(
                # A public locator is safe to queue without a credential.  A
                # signed locator instead resolves from the worker's secret
                # reference; the request URL itself is never recoverable.
                source_type=source_type,
                source_ref=public_ref,
                part=part,
                transcript_policy=transcript_policy,
                options=options,
                idempotency_key=_idempotency_key(idempotency_key, body_key),
                credential_ref=request.credential_ref,
                allowed_credential_refs=allowed_refs,
                allowed_credential_providers=allowed_providers,
                trace_id=http_request.state.trace_id,
                decision_id=http_request.state.decision_id,
                allow_legacy_options=True,
            )
        except IngestionValidationError as exc:
            raise _error(422, exc.code, str(exc)) from exc
        return enqueue(command_with_legacy_policy(command))

    return router


__all__ = ["create_ingestions_router"]

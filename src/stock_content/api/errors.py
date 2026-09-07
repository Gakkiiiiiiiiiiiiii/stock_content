"""Stable, redacted HTTP error protocol for stock_content."""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from stock_content.adapters.postgres.database import SchemaNotReadyError
from stock_content.api.security import ServiceAuthError
from stock_content.domain.security_redaction import redact_for_serialization
from stock_content.ports.repositories import IdempotencyConflict

_STATUS_DEFAULTS = {
    400: ("BAD_REQUEST", False, "request is invalid"),
    401: ("AUTH_REQUIRED", False, "Bearer authentication is required"),
    403: ("CALLER_FORBIDDEN", False, "caller service is not allowed"),
    404: ("NOT_FOUND", False, "resource was not found"),
    409: ("IDEMPOTENCY_CONFLICT", False, "request conflicts with existing state"),
    415: ("UNSUPPORTED_MEDIA_TYPE", False, "application/json content type is required"),
    422: ("VALIDATION_ERROR", False, "request validation failed"),
    429: ("THROTTLED", True, "request is throttled"),
    502: ("UPSTREAM_FAILURE", True, "upstream dependency failed"),
    503: ("DEPENDENCY_NOT_READY", True, "required dependency is not ready"),
    504: ("UPSTREAM_TIMEOUT", True, "upstream dependency timed out"),
}
_SAFE_MESSAGES = {
    "INVALID_INGESTION_REQUEST": "ingestion request is invalid",
    "IDEMPOTENCY_KEY_MISMATCH": "idempotency key conflicts with existing request",
    "IDEMPOTENCY_CONFLICT": "idempotency key conflicts with existing request",
    "KNOWLEDGE_BUNDLE_NOT_READY": "knowledge bundle service is not ready",
    "CONTENT_SNAPSHOT_MISMATCH": "content snapshot cannot satisfy this request",
    "CONTENT_BUNDLE_BUILD_FAILED": "knowledge bundle could not be built",
}


def envelope(code: str, message: str, retryable: bool, trace_id: str, details: object | None = None) -> dict:
    error = {"code": code, "message": message, "retryable": retryable, "trace_id": trace_id}
    if details is not None:
        error["details"] = details
    return {"error": error}


def _trace(request: Request) -> str:
    return str(getattr(request.state, "trace_id", "unknown"))


def install_error_handlers(app) -> None:
    @app.exception_handler(ServiceAuthError)
    async def service_auth_error(request: Request, exc: ServiceAuthError):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else {}
        return JSONResponse(
            envelope(exc.code, exc.message, exc.status_code >= 500, _trace(request)), exc.status_code, headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _exc: RequestValidationError):
        return JSONResponse(envelope("VALIDATION_ERROR", "request validation failed", False, _trace(request)), 422)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        fallback = _STATUS_DEFAULTS.get(exc.status_code, ("INTERNAL_ERROR", True, "internal service error"))
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        code = str(detail.get("code") or fallback[0])
        message = _SAFE_MESSAGES.get(code, fallback[2])
        # Details are only emitted when an adapter deliberately supplied a
        # bounded structured object; strings often contain upstream payloads.
        safe_details = detail.get("details") if isinstance(detail.get("details"), (dict, list)) else None
        if safe_details is not None:
            safe_details = redact_for_serialization(safe_details)
        headers = dict(exc.headers or {})
        if exc.status_code == 401:
            headers.setdefault("WWW-Authenticate", "Bearer")
        return JSONResponse(
            envelope(code, message, fallback[1], _trace(request), safe_details), exc.status_code, headers
        )

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_error(request: Request, _exc: IdempotencyConflict):
        return JSONResponse(
            envelope(
                "IDEMPOTENCY_CONFLICT", "idempotency key conflicts with an existing request", False, _trace(request)
            ),
            409,
        )

    @app.exception_handler(SchemaNotReadyError)
    async def schema_error(request: Request, _exc: SchemaNotReadyError):
        return JSONResponse(envelope("SCHEMA_NOT_READY", "required schema is not ready", True, _trace(request)), 503)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _exc: Exception):
        return JSONResponse(envelope("INTERNAL_ERROR", "internal service error", True, _trace(request)), 500)


__all__ = ["envelope", "install_error_handlers"]

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from stock_content.api.dependencies import build_application
from stock_content.application.service import ContentApplication

SERVICE_NAME = "stock_content"
SERVICE_VERSION = "1.0.0"
CONTRACT_VERSIONS = ["content.v1", "content-factor-signal.v2"]


class IngestRequest(BaseModel):
    url: str | None = None
    bv_id: str | None = None
    m3u8_url: str | None = None
    options: dict = Field(default_factory=dict)


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: dict = Field(default_factory=dict)
    limit: int = Field(default=20, ge=1, le=100)
    intent: str | None = None


class ContentSignalRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    start: datetime
    end: datetime
    minimum_support_status: str = "SOURCE_SUPPORTED"


def _with_idempotency(options: dict, idempotency_key: str | None) -> dict:
    # §33：Content ingest 支持 Idempotency-Key，避免重试产生重复任务。
    merged = dict(options or {})
    if idempotency_key and not merged.get("idempotency_key"):
        merged["idempotency_key"] = idempotency_key
    return merged


def create_app(service: ContentApplication | None = None) -> FastAPI:
    app = FastAPI(title="stock_content", version="1.0.0")
    application = service or build_application()

    @app.middleware("http")
    async def trace_headers(request: Request, call_next):
        # §32：统一 Trace Headers，全链路保持同一 trace_id。
        trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
        response: Response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        if request.headers.get("x-decision-id"):
            response.headers["x-decision-id"] = request.headers["x-decision-id"]
        response.headers["x-caller-service"] = request.headers.get("x-caller-service", "")
        return response

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok", "service": SERVICE_NAME, "contract_version": "content.v1"}

    @app.get("/health/version")
    def health_version() -> dict:
        # §106 Release Version
        return {
            "service": SERVICE_NAME,
            "service_version": SERVICE_VERSION,
            "git_commit": os.getenv("CONTENT_GIT_COMMIT", "unknown"),
            "contract_versions": CONTRACT_VERSIONS,
        }

    @app.post("/api/v1/videos/bilibili/ingest")
    def ingest_bilibili(request: IngestRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict:
        source_ref = request.url or request.bv_id
        if not source_ref:
            raise HTTPException(status_code=422, detail="url or bv_id is required")
        options = _with_idempotency(request.options, idempotency_key)
        return application.enqueue("bilibili", source_ref, options)

    @app.post("/api/v1/videos/xiaoe/ingest")
    def ingest_xiaoe(request: IngestRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict:
        if not request.m3u8_url:
            raise HTTPException(status_code=422, detail="m3u8_url is required")
        options = _with_idempotency(request.options, idempotency_key)
        return application.enqueue("xiaoe_hls", request.m3u8_url, options)

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(task_id: str) -> dict:
        payload = application.get_task(task_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="task not found")
        return payload

    @app.get("/api/v1/videos/{video_id}")
    def get_video(video_id: str) -> dict:
        payload = application.get_video(video_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="video not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.get("/api/v1/videos")
    def list_videos(limit: int = 50) -> dict:
        safe_limit = max(1, min(limit, 200))
        return {"contract_version": "content.v1", "items": application.list_videos(safe_limit)}

    @app.get("/api/v1/videos/{video_id}/segments")
    def get_video_segments(video_id: str) -> dict:
        items = application.get_segments(video_id)
        if items is None:
            raise HTTPException(status_code=404, detail="video not found")
        return {"contract_version": "content.v1", "video_id": video_id, "items": items}

    @app.get("/api/v1/videos/{video_id}/chapters")
    def get_video_chapters(video_id: str) -> dict:
        items = application.get_chapters(video_id)
        if items is None:
            raise HTTPException(status_code=404, detail="video not found")
        return {"contract_version": "content.v1", "video_id": video_id, "items": items}

    @app.get("/api/v1/videos/{video_id}/summary")
    def get_video_summary(video_id: str) -> dict:
        payload = application.get_summary(video_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="summary not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.get("/api/v1/videos/{video_id}/knowledge")
    def list_video_knowledge(video_id: str, limit: int = 100) -> dict:
        items = application.list_video_knowledge(video_id, max(1, min(limit, 500)))
        if items is None:
            raise HTTPException(status_code=404, detail="video not found")
        return {"contract_version": "content.v1", "video_id": video_id, "items": items}

    @app.get("/api/v1/knowledge/{knowledge_uid}")
    def get_knowledge(knowledge_uid: str) -> dict:
        payload = application.get_knowledge(knowledge_uid)
        if payload is None:
            raise HTTPException(status_code=404, detail="knowledge unit not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.post("/api/v1/knowledge/search")
    def search_knowledge(request: KnowledgeSearchRequest) -> dict:
        items = application.search_knowledge(request.query, request.filters, request.limit)
        return {
            "contract_version": "content.v1",
            "items": items,
            "limit": request.limit,
            "intent": request.intent or "research",
            "filters": request.filters,
        }

    @app.post("/internal/v1/factor-signals")
    def factor_signals(request: ContentSignalRequest) -> dict:
        start = request.start.replace(tzinfo=request.start.tzinfo or UTC)
        end = request.end.replace(tzinfo=request.end.tzinfo or UTC)
        items = application.factor_signals(request.symbols, start, end, request.minimum_support_status)
        return {"contract_version": "content-factor-signal.v2", "items": items}

    return app


app = create_app()

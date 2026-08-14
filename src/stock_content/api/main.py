from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stock_content.api.dependencies import build_application
from stock_content.application.service import ContentApplication


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


def create_app(service: ContentApplication | None = None) -> FastAPI:
    app = FastAPI(title="stock_content", version="1.0.0")
    application = service or build_application()

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok", "service": "stock_content", "contract_version": "content.v1"}

    @app.post("/api/v1/videos/bilibili/ingest")
    def ingest_bilibili(request: IngestRequest) -> dict:
        source_ref = request.url or request.bv_id
        if not source_ref:
            raise HTTPException(status_code=422, detail="url or bv_id is required")
        return application.enqueue("bilibili", source_ref, request.options)

    @app.post("/api/v1/videos/xiaoe/ingest")
    def ingest_xiaoe(request: IngestRequest) -> dict:
        if not request.m3u8_url:
            raise HTTPException(status_code=422, detail="m3u8_url is required")
        return application.enqueue("xiaoe_hls", request.m3u8_url, request.options)

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

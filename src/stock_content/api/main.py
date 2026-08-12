from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stock_content.application.service import ContentApplication

app = FastAPI(title="stock_content", version="1.0.0")
service = ContentApplication()


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
    start: str
    end: str
    minimum_support_status: str = "SOURCE_SUPPORTED"


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "service": "stock_content", "contract_version": "content.v1"}


@app.post("/api/v1/videos/bilibili/ingest")
def ingest_bilibili(request: IngestRequest) -> dict:
    source_ref = request.url or request.bv_id
    if not source_ref:
        raise HTTPException(status_code=422, detail="url or bv_id is required")
    return service.enqueue("bilibili", source_ref, request.options)


@app.post("/api/v1/videos/xiaoe/ingest")
def ingest_xiaoe(request: IngestRequest) -> dict:
    if not request.m3u8_url:
        raise HTTPException(status_code=422, detail="m3u8_url is required")
    return service.enqueue("xiaoe_hls", request.m3u8_url, request.options)


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    payload = service.get_task(task_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="task not found")
    return payload


@app.post("/api/v1/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest) -> dict:
    # The persisted knowledge adapter is added together with the database copy.
    return {"items": [], "limit": request.limit, "intent": request.intent or "research", "filters": request.filters}


@app.post("/internal/v1/factor-signals")
def factor_signals(request: ContentSignalRequest) -> dict:
    return {"contract_version": "content-factor-signal.v1", "items": []}

from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError

from stock_content.api.dependencies import build_application
from stock_content.application.service import ContentApplication
from stock_content.domain.claims import FinancialClaim

SERVICE_NAME = "stock_content"
SERVICE_VERSION = "1.0.0"
CONTRACT_VERSIONS = ["content.v1", "content-factor-signal.v3", "content-factor-signal.v4", "content-factor-signal.v5"]


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
    availability_as_of: datetime | None = None
    target_start: str | None = None
    target_end: str | None = None
    temporal_role: str | None = None
    semantic_segment_id: str | None = None
    business_as_of: datetime | None = None
    knowledge_as_of: datetime | None = None
    pit_mode: str | None = None


class ContentSignalRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    start: datetime
    end: datetime
    minimum_support_status: str = "SOURCE_SUPPORTED"
    availability_as_of: datetime | None = None
    pit_mode: str | None = None


class ClaimRequest(BaseModel):
    # §8：FinancialClaim 登记（P1-1）。
    claim_type: str
    subject_type: str
    subject_id: str
    predicate: str
    value: Any = None
    unit: str | None = None
    fact_time: str | None = None
    published_at: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    source_confidence: float = 0.5
    extractor_confidence: float = 0.5


class VerificationRetryRequest(BaseModel):
    claim_id: str | None = None


class ReplayRequest(BaseModel):
    mode: str = "VERIFY_LINEAGE"
    pipeline_version: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


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
        decision_id = request.headers.get("x-decision-id") or None
        request.state.trace_id = trace_id
        request.state.decision_id = decision_id
        response: Response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        if decision_id:
            response.headers["x-decision-id"] = decision_id
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
    def ingest_bilibili(
        request: IngestRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        source_ref = request.url or request.bv_id
        if not source_ref:
            raise HTTPException(status_code=422, detail="url or bv_id is required")
        options = _with_idempotency(request.options, idempotency_key)
        options["trace_id"] = http_request.state.trace_id
        if http_request.state.decision_id:
            options["decision_id"] = http_request.state.decision_id
        return application.enqueue("bilibili", source_ref, options)

    @app.post("/api/v1/videos/xiaoe/ingest")
    def ingest_xiaoe(
        request: IngestRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        if not request.m3u8_url:
            raise HTTPException(status_code=422, detail="m3u8_url is required")
        options = _with_idempotency(request.options, idempotency_key)
        options["trace_id"] = http_request.state.trace_id
        if http_request.state.decision_id:
            options["decision_id"] = http_request.state.decision_id
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

    # Register the static search route before /knowledge/{knowledge_uid};
    # otherwise FastAPI treats the literal "search" as a knowledge id.
    @app.get("/api/v1/knowledge/search")
    def search_knowledge_get(
        query: str,
        availability_as_of: datetime | None = None,
        target_start: str | None = None,
        target_end: str | None = None,
        temporal_role: str | None = None,
        semantic_segment_id: str | None = None,
        business_as_of: datetime | None = None,
        knowledge_as_of: datetime | None = None,
        pit_mode: str | None = None,
        limit: int = 20,
    ) -> dict:
        try:
            items = application.search_knowledge(
                query, {}, max(1, min(limit, 100)),
                availability_as_of=availability_as_of,
                target_start=target_start,
                target_end=target_end,
                temporal_role=temporal_role,
                semantic_segment_id=semantic_segment_id,
                business_as_of=business_as_of,
                knowledge_as_of=knowledge_as_of,
                pit_mode=pit_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "contract_version": "content.v1", "items": items,
            "limit": max(1, min(limit, 100)), "intent": "research",
            "filters": {
                "availability_as_of": availability_as_of, "target_start": target_start,
                "target_end": target_end, "temporal_role": temporal_role,
                "semantic_segment_id": semantic_segment_id, "business_as_of": business_as_of,
                "knowledge_as_of": knowledge_as_of, "pit_mode": pit_mode,
            },
        }

    @app.get("/api/v1/knowledge/{knowledge_uid}")
    def get_knowledge(knowledge_uid: str) -> dict:
        payload = application.get_knowledge(knowledge_uid)
        if payload is None:
            raise HTTPException(status_code=404, detail="knowledge unit not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.post("/api/v1/knowledge/search")
    def search_knowledge(request: KnowledgeSearchRequest) -> dict:
        effective_filters = {
            **dict(request.filters or {}),
            **{
                key: value for key, value in {
                    "availability_as_of": request.availability_as_of,
                    "target_start": request.target_start,
                    "target_end": request.target_end,
                    "temporal_role": request.temporal_role,
                    "semantic_segment_id": request.semantic_segment_id,
                    "business_as_of": request.business_as_of,
                    "knowledge_as_of": request.knowledge_as_of,
                    "pit_mode": request.pit_mode,
                }.items() if value is not None
            },
        }
        try:
            items = application.search_knowledge(request.query, effective_filters, request.limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "contract_version": "content.v1",
            "items": items,
            "limit": request.limit,
            "intent": request.intent or "research",
            "filters": effective_filters,
        }

    @app.get("/api/v1/videos/{video_id}/snapshots")
    def list_video_snapshots(video_id: str) -> dict:
        # §4 P0-2：同一 source 的历次处理产物版本列表。
        items = application.list_snapshots_for_video(video_id)
        if items is None:
            raise HTTPException(status_code=404, detail="video not found")
        return {"contract_version": "content.v1", "video_id": video_id, "items": items}

    @app.get("/api/v1/content-snapshots/{content_snapshot_id}")
    def get_content_snapshot(content_snapshot_id: str) -> dict:
        payload = application.get_content_snapshot(content_snapshot_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="content snapshot not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.get("/api/v1/content-snapshots/{content_snapshot_id}/lineage")
    def get_content_snapshot_lineage(content_snapshot_id: str) -> dict:
        payload = application.get_snapshot_lineage(content_snapshot_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="content snapshot not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.post("/api/v1/content-snapshots/{content_snapshot_id}/replay")
    def replay_content_snapshot(content_snapshot_id: str, request: ReplayRequest | None = None) -> dict:
        result = application.replay_content_snapshot(
            content_snapshot_id,
            mode=request.mode if request else None,
            pipeline_version=request.pipeline_version if request else None,
            overrides=request.overrides if request else None,
        )
        error = result.get("error")
        if error:
            status = {
                "SNAPSHOT_NOT_FOUND": 404,
                "INVALID_REPLAY_MODE": 422,
                "INVALID_REPLAY_REQUEST": 422,
                "REPLAY_ARTIFACT_MISSING": 409,
                "REPLAY_ARTIFACT_HASH_MISMATCH": 409,
                "REPLAY_LINEAGE_CYCLE": 409,
                "REPLAY_LINEAGE_REFERENCE_MISSING": 409,
                "REPLAY_LINEAGE_REFERENCE_INVALID": 409,
                "REPLAY_IDENTITY_MISMATCH": 409,
                "REPLAY_NONDETERMINISTIC": 409,
                "REPLAY_INPUT_UNAVAILABLE": 424,
                "REPLAY_UNAVAILABLE": 503,
            }.get(str(error), 500)
            raise HTTPException(status_code=status, detail=result)
        return {"contract_version": "content.v1", **result}

    @app.get("/api/v1/content-snapshots/{content_snapshot_id}/signals")
    def get_snapshot_signals(content_snapshot_id: str, claim_id: str | None = None) -> dict:
        if application.get_content_snapshot(content_snapshot_id) is None:
            raise HTTPException(status_code=404, detail="content snapshot not found")
        return {
            "contract_version": "content-factor-signal.v4",
            "content_snapshot_id": content_snapshot_id,
            "items": application.get_snapshot_signals(content_snapshot_id, claim_id),
        }

    @app.get("/api/v1/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict:
        payload = application.get_artifact(artifact_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.get("/api/v1/artifacts/{artifact_id}/lineage")
    def get_artifact_lineage(artifact_id: str) -> dict:
        payload = application.get_artifact_lineage(artifact_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.post("/api/v1/claims")
    def register_claim(request: ClaimRequest, http_request: Request) -> dict:
        # §5 P1-1：claim 登记 -> 验证生命周期 + 冲突检测。
        try:
            claim = FinancialClaim(
                claim_type=request.claim_type,
                subject_type=request.subject_type,
                subject_id=request.subject_id,
                predicate=request.predicate,
                value=request.value,
                unit=request.unit,
                fact_time=date.fromisoformat(request.fact_time) if request.fact_time else None,
                evidence_refs=request.evidence_refs,
                source_confidence=request.source_confidence,
                extractor_confidence=request.extractor_confidence,
            )
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "contract_version": "content.v1",
            **application.register_claim(claim, trace_id=getattr(http_request.state, "trace_id", None)),
        }

    @app.get("/api/v1/claims/{claim_id}")
    def get_claim(claim_id: str) -> dict:
        payload = application.get_claim(claim_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="claim not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.get("/api/v1/claims/{claim_id}/evidence")
    def get_claim_evidence(claim_id: str) -> dict:
        payload = application.get_claim_evidence(claim_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="claim not found")
        return {"contract_version": "content.v1", "claim_id": claim_id, "items": payload}

    @app.get("/api/v1/claims/{claim_id}/verifications")
    def get_claim_verifications(claim_id: str) -> dict:
        payload = application.get_claim_verifications(claim_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="claim not found")
        return {"contract_version": "content.v1", "claim_id": claim_id, "items": payload}

    @app.get("/api/v1/claims/{claim_id}/verification")
    def get_claim_verification(claim_id: str) -> dict:
        payload = application.get_claim_verification(claim_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="claim verification not found")
        return {"contract_version": "content.v1", "data": payload}

    @app.get("/api/v1/signals/{signal_id}/lineage")
    def get_signal_lineage(signal_id: str) -> dict:
        payload = application.get_signal_lineage(signal_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="signal not found")
        return {"contract_version": "content-factor-signal.v4", "data": payload}

    @app.post("/api/v1/verification/retry")
    def retry_verification(request: VerificationRetryRequest) -> dict:
        result = application.retry_verification(request.claim_id)
        if result.get("error") == "CLAIM_NOT_FOUND":
            raise HTTPException(status_code=404, detail="claim not found")
        return {"contract_version": "content.v1", **result}

    @app.get("/api/v1/conflicts")
    def list_conflicts(status: str | None = None) -> dict:
        try:
            items = application.list_conflicts(status)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"contract_version": "content.v1", "items": items}

    @app.post("/internal/v1/factor-signals")
    def factor_signals(request: ContentSignalRequest) -> dict:
        start = request.start.replace(tzinfo=request.start.tzinfo or UTC)
        end = request.end.replace(tzinfo=request.end.tzinfo or UTC)
        items = application.factor_signals(request.symbols, start, end, request.minimum_support_status)
        return {"contract_version": "content-factor-signal.v3", "items": items}

    @app.post("/internal/v1/factor-signals/v4")
    def factor_signals_v4(request: ContentSignalRequest) -> dict:
        start = request.start.replace(tzinfo=request.start.tzinfo or UTC)
        end = request.end.replace(tzinfo=request.end.tzinfo or UTC)
        return {
            "contract_version": "content-factor-signal.v4",
            "items": application.factor_signals_v4(request.symbols, start, end),
        }

    @app.post("/internal/v1/factor-signals/v5")
    def factor_signals_v5(request: ContentSignalRequest) -> dict:
        start = request.start.replace(tzinfo=request.start.tzinfo or UTC)
        end = request.end.replace(tzinfo=request.end.tzinfo or UTC)
        try:
            items = application.factor_signals_v5(
                request.symbols,
                start,
                end,
                request.minimum_support_status,
                availability_as_of=request.availability_as_of,
                pit_mode=request.pit_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"contract_version": "content-factor-signal.v5", "items": items}

    return app


app = create_app()

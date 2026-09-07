from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from stock_content.domain.knowledge_bundle import KnowledgeBundleRequest


class BundleRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_snapshot_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    business_as_of: datetime
    knowledge_as_of: datetime
    availability_as_of: datetime
    minimum_support_status: str = Field(min_length=1)
    max_items: int = Field(ge=1, le=100)
    policy: str = "PUBLIC_STRICT"
    policy_version: str = "content-bundle-policy.v1"


def create_knowledge_bundles_router(application):
    router = APIRouter(prefix="/v1/content/knowledge-bundles", tags=["knowledge-bundles"])

    @router.post("")
    def create(body: BundleRequestBody) -> dict:
        try:
            return application().create_knowledge_bundle(KnowledgeBundleRequest(**body.model_dump()))
        except ValueError as exc:
            code = str(exc).split(":", 1)[0]
            if code == "KNOWLEDGE_BUNDLE_PRODUCER_NOT_CONFIGURED":
                raise HTTPException(status_code=503, detail={"code": "KNOWLEDGE_BUNDLE_NOT_READY"}) from exc
            if code in {"AVAILABILITY_AS_OF_EXCEEDED", "KNOWLEDGE_AS_OF_EXCEEDED"}:
                raise HTTPException(status_code=409, detail={"code": "CONTENT_SNAPSHOT_MISMATCH"}) from exc
            raise HTTPException(status_code=422, detail={"code": "CONTENT_BUNDLE_BUILD_FAILED"}) from exc

    @router.get("/{bundle_id}")
    def get(bundle_id: str) -> dict:
        payload = application().get_knowledge_bundle(bundle_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="bundle not found")
        return payload

    return router

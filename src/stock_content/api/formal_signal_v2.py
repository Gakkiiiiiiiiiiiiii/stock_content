"""Formal v2 signal query request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stock_content.domain.bitemporal_query import FORMAL_CONTRACT, PUBLIC_STRICT, FormalContentSignalQueryV2
from stock_content.domain.signal_contract_v5_1 import CONTRACT_CHECKSUM


class FormalSignalQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract: str = Field(..., min_length=1)
    request_id: str = Field(..., min_length=1)
    symbols: list[str] = Field(..., min_length=1)
    business_as_of: datetime
    knowledge_as_of: datetime
    availability_as_of: datetime
    content_snapshot_id: str = Field(..., min_length=1)
    pit_mode: str = Field(..., min_length=1)
    min_support: int = Field(..., ge=1)
    signal_policy_version: str = "signal-policy.v1"
    start: datetime | None = None
    end: datetime | None = None

    def to_domain(self) -> FormalContentSignalQueryV2:
        return FormalContentSignalQueryV2(**self.model_dump())


def formal_manifest(
    query: FormalContentSignalQueryV2, *, producer_commit: str, items: list[dict[str, Any]]
) -> dict[str, Any]:
    clocks = {
        field: getattr(query, field).isoformat().replace("+00:00", "Z")
        for field in ("business_as_of", "knowledge_as_of", "availability_as_of")
    }
    return {
        "contract": FORMAL_CONTRACT,
        "contract_checksum": CONTRACT_CHECKSUM,
        "authority": "FORMAL_FACT",
        "formal_eligible": True,
        "query_id": query.query_id,
        "content_snapshot_id": query.content_snapshot_id,
        **clocks,
        "pit_mode": PUBLIC_STRICT,
        "signal_policy_version": query.signal_policy_version,
        "producer_commit": producer_commit,
        "signal_count": len(items),
    }


__all__ = ["FormalSignalQueryRequest", "formal_manifest"]

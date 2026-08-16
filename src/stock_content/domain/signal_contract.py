"""content-factor-signal 契约治理（详细修改方案 §7）。

从 v2 演进为 v3：新增 signal_schema_version / producer_version /
claim_id / content_snapshot_id / producer 血缘块。
Factor 侧必须可以拒绝不支持的 major version。
"""
from __future__ import annotations

import os
from typing import Any

SIGNAL_SCHEMA_VERSION = "content-factor-signal.v3"
SIGNAL_SCHEMA_MAJOR = 3
PRODUCER_VERSION = "1.0.0"
SERVICE_VERSION = "1.0.0"

_DIRECTION_BY_SENTIMENT = {"BULLISH": "LONG", "BEARISH": "SHORT"}


def signal_major_version(schema_version: str) -> int | None:
    """解析 schema 版本号的 major（'content-factor-signal.v3' -> 3）。"""
    try:
        return int(str(schema_version).rsplit("v", 1)[-1])
    except (ValueError, IndexError):
        return None


def accepts_schema_version(schema_version: str, *, max_supported_major: int = SIGNAL_SCHEMA_MAJOR) -> bool:
    """消费方（Factor）拒绝不支持的 major version。"""
    major = signal_major_version(schema_version)
    return major is not None and 1 <= major <= max_supported_major


def upgrade_signal_v3(item: dict[str, Any], *, code_sha: str | None = None) -> dict[str, Any]:
    """将既有 signal payload 升级为 v3（保持旧字段向后兼容）。

    P0 C-03：ContentSnapshot 缺失的结果不得作为 v3 正常信号发送，
    只能以 DEGRADED_NO_SNAPSHOT 显式降级，consumer 必须拒绝当正常信号消费。
    """
    attributes = item.get("provenance") or {}
    sentiment = str(item.get("sentiment") or "")
    resolved_code_sha = code_sha if code_sha is not None else os.getenv("CONTENT_GIT_COMMIT", "unknown")
    content_snapshot_id = item.get("content_snapshot_id")
    return {
        **item,
        "signal_id": item.get("signal_id") or item.get("knowledge_uid"),
        "signal_schema_version": SIGNAL_SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "signal_status": "NORMAL" if content_snapshot_id else "DEGRADED_NO_SNAPSHOT",
        "content_snapshot_id": content_snapshot_id,
        "claim_id": item.get("claim_id"),
        "symbol": item.get("symbol") or item.get("subject_key"),
        "event_time": item.get("market_fact_date") or item.get("as_of_time"),
        "published_at": item.get("available_from"),
        "signal_type": item.get("knowledge_kind") or item.get("kind"),
        "direction": _DIRECTION_BY_SENTIMENT.get(sentiment, "NEUTRAL"),
        "magnitude": item.get("event_strength", item.get("confidence")),
        "confidence": item.get("confidence"),
        "support_status": item.get("support_status"),
        "evidence_refs": list(item.get("evidence_ids") or []),
        "market_snapshot_id": item.get("market_snapshot_id"),
        "market_data_version": item.get("market_data_version"),
        "producer": {
            "service_version": SERVICE_VERSION,
            "code_sha": resolved_code_sha,
            "model_id": attributes.get("model"),
            "prompt_version": attributes.get("prompt_version"),
        },
    }


def is_normal_v3_signal(signal: dict[str, Any]) -> bool:
    """consumer 判定：只有携带 content_snapshot_id 的 NORMAL 信号才可当正常 v3 信号消费。"""
    return (
        signal.get("signal_schema_version") == SIGNAL_SCHEMA_VERSION
        and signal.get("signal_status") == "NORMAL"
        and bool(signal.get("content_snapshot_id"))
    )


__all__ = [
    "PRODUCER_VERSION",
    "SERVICE_VERSION",
    "SIGNAL_SCHEMA_MAJOR",
    "SIGNAL_SCHEMA_VERSION",
    "accepts_schema_version",
    "is_normal_v3_signal",
    "signal_major_version",
    "upgrade_signal_v3",
]

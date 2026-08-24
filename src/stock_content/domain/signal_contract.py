"""content-factor-signal 契约治理（详细修改方案 §7）。

从 v2 演进为 v3：新增 signal_schema_version / producer_version /
claim_id / content_snapshot_id / producer 血缘块。
Factor 侧必须可以拒绝不支持的 major version。
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Any

SIGNAL_SCHEMA_VERSION = "content-factor-signal.v3"
SIGNAL_SCHEMA_MAJOR = 3
PRODUCER_VERSION = "1.0.0"
SERVICE_VERSION = "1.0.0"
SIGNAL_SCHEMA_V4 = "content-factor-signal.v4"
SIGNAL_SCHEMA_V4_MAJOR = 4

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


def signal_id_v4(
    *,
    content_snapshot_id: str,
    claim_id: str,
    policy_version: str,
    signal_type: str = "FACT",
    verification_artifact_id: str | None = None,
) -> str:
    """Deterministic id; verification artifact is captured by snapshot lineage."""
    raw = content_snapshot_id + claim_id + policy_version + signal_type
    return "signal-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def upgrade_signal_v4(
    item: dict[str, Any],
    *,
    content_snapshot_id: str | None = None,
    claim_id: str | None = None,
    verification_artifact_id: str | None = None,
    policy_version: str = "signal_policy.v1",
) -> dict[str, Any]:
    """Normalize a v4 payload and enforce its lineage references."""
    snapshot = content_snapshot_id or item.get("content_snapshot_id") or item.get("snapshot_id")
    claim = claim_id or item.get("claim_id")
    verification = verification_artifact_id or item.get("verification_artifact_id")
    if not snapshot or not claim or not verification:
        raise ValueError("content-factor-signal.v4 requires snapshot, claim and verification artifact")
    payload = dict(item)
    producer = dict(payload.get("producer") or {})
    support = dict(payload.get("support") or {})
    verification_payload = dict(payload.get("verification") or {})
    source = dict(payload.get("source") or {})
    signal_type = str(payload.get("signal_type") or payload.get("fact_category") or "FACT")
    truth_scope = {
        "FACT": "FACT",
        "FORECAST": "AUTHOR_FORECAST",
        "OPINION": "AUTHOR_OPINION",
        "INFERENCE": "SYSTEM_INFERENCE",
    }.get(signal_type, "UNKNOWN")
    event_time = payload.get("event_time") or payload.get("market_fact_date") or payload.get("published_at")
    published_at = payload.get("published_at") or payload.get("available_from") or event_time or "unknown"
    available_from = payload.get("available_from") or published_at or event_time or "unknown"
    effective_decision_id = payload.get("decision_id") or "decision-" + hashlib.sha256(
        f"{snapshot}:{claim}:{policy_version}:{signal_type}".encode()
    ).hexdigest()[:32]
    producer_defaults = {
        "service": producer.get("service") or "stock_content",
        "service_version": producer.get("service_version") or SERVICE_VERSION,
        "code_sha": producer.get("code_sha") or "unknown",
        "pipeline_version": producer.get("pipeline_version") or "pipeline.v3",
        "model_id": producer.get("model_id") or "unknown",
        "prompt_version": producer.get("prompt_version") or "unknown",
        "trace_id": producer.get("trace_id") or "unknown",
        "decision_id": effective_decision_id,
        "decision": producer.get("decision") or payload.get("decision") or "unknown",
    }
    producer_payload = dict(producer_defaults)
    producer_payload.update(producer)
    # Explicit nulls for generated defaults must not erase the normalized
    # values. Nullable producer metadata remains untouched, while the
    # cross-field decision id is finalized below.
    for key, default in producer_defaults.items():
        if producer_payload.get(key) is None:
            producer_payload[key] = default
    payload.update(
        {
            "signal_schema_version": SIGNAL_SCHEMA_V4,
            "signal_policy_version": policy_version,
            "content_snapshot_id": snapshot,
            "snapshot_id": snapshot,
            "claim_id": claim,
            "verification_artifact_id": verification,
            "decision_id": effective_decision_id,
            "signal_status": payload.get("signal_status") or "DEGRADED",
            "truth_scope": payload.get("truth_scope") or truth_scope,
            "signal_type": signal_type,
            "fact_category": payload.get("fact_category") or signal_type,
            "direction": payload.get("direction") or "NEUTRAL",
            "symbol": payload.get("symbol") or payload.get("subject_key") or "unknown",
            "magnitude": payload.get("magnitude", payload.get("event_strength", 0)),
            "confidence": payload.get("confidence", 0),
            "event_time": event_time or "unknown",
            "available_from": available_from,
            "published_at": published_at,
            "source": {
                "source_artifact_id": source.get("source_artifact_id") or f"source-{snapshot}",
                "source_type": source.get("source_type") or "unknown",
                "source_ref": source.get("source_ref") or snapshot,
            },
            "support": {
                "status": support.get("status") or payload.get("support_status") or "UNSUPPORTED",
                "score": support.get("score", payload.get("confidence", 0)),
                "evidence_refs": list(support.get("evidence_refs") or payload.get("evidence_refs") or []),
            },
            "evidence_refs": list(payload.get("evidence_refs") or support.get("evidence_refs") or []),
            "verification": {
                "status": (
                    verification_payload.get("status")
                    or payload.get("verification_status")
                    or "VERIFICATION_PENDING"
                ),
                "provider": verification_payload.get("provider") or "unknown",
                "market_snapshot_id": verification_payload.get("market_snapshot_id"),
                "market_data_version": verification_payload.get("market_data_version"),
                "verification_rule_version": verification_payload.get(
                    "verification_rule_version", "verification_rule.v1"
                ),
            },
            "producer": producer_payload,
            "policy": {
                "signal_policy_version": policy_version,
                **dict(payload.get("policy") or {}),
            },
            "signal_id": item.get("signal_id") or signal_id_v4(
                content_snapshot_id=snapshot,
                claim_id=claim,
                policy_version=policy_version,
                signal_type=signal_type,
                verification_artifact_id=verification,
            ),
        }
    )
    # A caller-supplied producer must not be able to break the cross-field
    # decision invariant while upgrading an otherwise legacy payload.
    payload["producer"]["decision_id"] = effective_decision_id
    return payload


def validate_signal_v4(signal: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(signal, dict):
        raise ValueError("v4 signal must be an object")
    allowed_top_level = {
        "signal_id", "decision_id", "signal_schema_version", "signal_policy_version",
        "content_snapshot_id", "claim_id", "verification_artifact_id", "signal_status",
        "event_type", "truth_scope", "symbol", "signal_type", "fact_category", "direction",
        "magnitude", "confidence", "event_time", "available_from", "published_at", "snapshot_id",
        "source", "support", "verification", "evidence_refs", "producer", "policy",
    }
    _reject_non_string_keys(signal, "signal")
    unexpected = sorted(set(signal) - allowed_top_level)
    if unexpected:
        raise ValueError(f"v4 signal contains unsupported fields: {', '.join(unexpected)}")
    for key in (
        "signal_id",
        "decision_id",
        "signal_schema_version",
        "content_snapshot_id",
        "claim_id",
        "verification_artifact_id",
        "signal_policy_version",
        "signal_status",
        "truth_scope",
        "symbol",
        "signal_type",
        "fact_category",
        "direction",
        "magnitude",
        "confidence",
        "event_time",
        "available_from",
        "published_at",
        "support",
        "verification",
        "evidence_refs",
        "producer",
        "policy",
    ):
        _require_key(signal, key, "signal")
    if signal.get("signal_schema_version") != SIGNAL_SCHEMA_V4:
        raise ValueError("unsupported signal schema major")
    forbidden = {"order_qty", "limit_price", "portfolio_weight", "execute_at"}
    if forbidden.intersection(signal):
        raise ValueError("v4 signal cannot contain trading instruction fields")
    _require_string(signal, "signal_id", min_length=1)
    _require_string(signal, "decision_id", min_length=1)
    _require_string(signal, "signal_policy_version", min_length=1)
    _require_string(signal, "content_snapshot_id", min_length=1)
    _require_string(signal, "claim_id", min_length=1)
    _require_string(signal, "verification_artifact_id", min_length=1)
    _require_enum(signal["signal_status"], {"NORMAL", "DEGRADED", "SUPPRESSED", "CONTRADICTION"}, "signal_status")
    _require_enum(
        signal["truth_scope"],
        {"FACT", "AUTHOR_FORECAST", "AUTHOR_OPINION", "SYSTEM_INFERENCE", "UNKNOWN"},
        "truth_scope",
    )
    _require_string(signal, "symbol", min_length=1)
    _require_enum(signal["signal_type"], {"FACT", "FORECAST", "OPINION", "INFERENCE"}, "signal_type")
    _require_string(signal, "fact_category", min_length=1)
    _require_enum(signal["direction"], {"LONG", "SHORT", "NEUTRAL"}, "direction")
    _require_number(signal["confidence"], "confidence", minimum=0, maximum=1)
    _require_string(signal, "event_time", min_length=1)
    _require_string(signal, "available_from", min_length=1)
    _require_string(signal, "published_at", min_length=1)
    if "event_type" in signal and signal["event_type"] is not None and not isinstance(signal["event_type"], str):
        raise ValueError("v4 signal event_type must be a string or null")
    if "snapshot_id" in signal:
        _require_string(signal, "snapshot_id", min_length=1)
    source = signal.get("source")
    if not isinstance(source, dict):
        raise ValueError("v4 signal requires source lineage")
    _reject_additional_properties(source, {"source_artifact_id", "source_type", "source_ref"}, "source")
    for key in ("source_artifact_id", "source_type", "source_ref"):
        _require_string(source, key, min_length=1, object_name="source")
    support = signal.get("support")
    if not isinstance(support, dict):
        raise ValueError("v4 signal requires valid support status")
    _reject_additional_properties(support, {"status", "score", "evidence_refs"}, "support")
    _require_present(support, "status", "support")
    _require_present(support, "score", "support")
    _require_present(support, "evidence_refs", "support")
    _require_enum(support["status"], {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "AMBIGUOUS"}, "support.status")
    _require_number(support["score"], "support.score", minimum=0, maximum=1)
    _require_string_array(support["evidence_refs"], "support.evidence_refs")
    verification = signal.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("v4 signal requires verification lineage")
    _reject_additional_properties(
        verification,
        {"status", "provider", "market_snapshot_id", "market_data_version", "verification_rule_version"},
        "verification",
    )
    for key in ("status", "provider", "verification_rule_version"):
        _require_present(verification, key, "verification")
        _require_string(verification, key, min_length=1, object_name="verification")
    for key in ("market_snapshot_id", "market_data_version"):
        if key not in verification:
            raise ValueError(f"v4 verification missing {key}")
        _require_nullable_string(verification[key], f"verification.{key}")
    producer = signal.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("v4 signal requires producer lineage")
    _reject_additional_properties(
        producer,
        {
            "service", "service_version", "code_sha", "container_digest", "dependency_lock_hash",
            "python_lock_hash", "pipeline_version", "model_id", "prompt_version", "trace_id",
            "decision_id", "decision",
        },
        "producer",
    )
    for key in (
        "service", "service_version", "code_sha", "pipeline_version", "model_id",
        "prompt_version", "trace_id", "decision_id",
    ):
        _require_present(producer, key, "producer")
        _require_string(producer, key, min_length=1, object_name="producer")
    if "decision" in producer:
        _require_string(producer, "decision", min_length=1, object_name="producer")
    for key in ("container_digest", "dependency_lock_hash", "python_lock_hash"):
        if key in producer:
            _require_nullable_string(producer[key], f"producer.{key}")
    if signal.get("decision_id") != producer.get("decision_id"):
        raise ValueError("v4 decision_id must match producer.decision_id")
    policy = signal.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("v4 policy missing signal_policy_version")
    _require_present(policy, "signal_policy_version", "policy")
    _require_string(policy, "signal_policy_version", min_length=1, object_name="policy")
    _reject_additional_properties(policy, {"signal_policy_version", "forecast_confidence_threshold"}, "policy")
    if "forecast_confidence_threshold" in policy:
        _require_number(
            policy["forecast_confidence_threshold"],
            "policy.forecast_confidence_threshold",
            minimum=0,
            maximum=1,
        )
    _require_string_array(signal["evidence_refs"], "evidence_refs")
    return signal


def _reject_additional_properties(value: dict[str, Any], allowed: set[str], object_name: str) -> None:
    _reject_non_string_keys(value, object_name)
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"v4 {object_name} contains unsupported fields: {', '.join(unexpected)}")


def _reject_non_string_keys(value: dict[Any, Any], object_name: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"v4 {object_name} property names must be strings")


def _require_present(value: dict[str, Any], key: str, object_name: str) -> None:
    if key not in value or value[key] is None:
        raise ValueError(f"v4 {object_name} missing {key}")


def _require_key(value: dict[str, Any], key: str, object_name: str) -> None:
    if key not in value:
        raise ValueError(f"v4 {object_name} missing {key}")


def _require_string(
    value: dict[str, Any],
    key: str,
    *,
    min_length: int = 0,
    object_name: str = "signal",
) -> None:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"v4 {object_name}.{key} must be a string")
    if len(item) < min_length:
        raise ValueError(f"v4 {object_name}.{key} must have minLength {min_length}")


def _require_nullable_string(value: Any, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"v4 {field_name} must be a string or null")


def _require_enum(value: Any, allowed: set[str], field_name: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"v4 {field_name} has unsupported value")


def _require_number(value: Any, field_name: str, *, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise ValueError(f"v4 {field_name} must be a number")
    if value < minimum or value > maximum:
        raise ValueError(f"v4 {field_name} must be between {minimum} and {maximum}")


def _require_string_array(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"v4 {field_name} must be an array")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"v4 {field_name} items must be strings")


__all__ = [
    "PRODUCER_VERSION",
    "SERVICE_VERSION",
    "SIGNAL_SCHEMA_MAJOR",
    "SIGNAL_SCHEMA_VERSION",
    "SIGNAL_SCHEMA_V4",
    "SIGNAL_SCHEMA_V4_MAJOR",
    "accepts_schema_version",
    "is_normal_v3_signal",
    "signal_id_v4",
    "upgrade_signal_v4",
    "validate_signal_v4",
    "signal_major_version",
    "upgrade_signal_v3",
]

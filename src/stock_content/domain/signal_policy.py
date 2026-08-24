"""Pure v4 signal policy over immutable snapshot/claim/verification data."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

SIGNAL_POLICY_VERSION = "signal_policy.v1"
FORECAST_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class SignalDecision:
    status: str
    reason: str
    truth_scope: str
    signal_type: str
    event_type: str | None = None

    @property
    def allowed(self) -> bool:
        return self.status != "SUPPRESSED"


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def signal_id_v4(*, content_snapshot_id: str, claim_id: str, policy_version: str, signal_type: str) -> str:
    raw = content_snapshot_id + claim_id + policy_version + signal_type
    return "signal-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def decision_id_v4(*, content_snapshot_id: str, claim_id: str, policy_version: str, signal_type: str) -> str:
    """Stable audit id for a policy decision (independent of transport)."""
    raw = f"{content_snapshot_id}:{claim_id}:{policy_version}:{signal_type}"
    return "decision-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class SignalPolicy:
    version = SIGNAL_POLICY_VERSION
    forecast_confidence_threshold = FORECAST_CONFIDENCE_THRESHOLD

    def evaluate(self, claim: Any, verification: Any = None, *, snapshot: Any = None) -> SignalDecision:
        category = str(_value(claim, "fact_category", ""))
        claim_type = str(_value(claim, "claim_type", ""))
        support = str(_value(claim, "source_support_status", "UNSUPPORTED"))
        evidence = _value(claim, "evidence_refs", []) or []
        confidence = float(_value(claim, "extractor_confidence", 0) or 0)
        if category == "FACT" or claim_type in {
            "PRICE", "RETURN", "VALUATION", "FINANCIAL_METRIC", "CORPORATE_EVENT", "INDUSTRY_RELATION"
        }:
            if support not in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
                return SignalDecision("SUPPRESSED", "INSUFFICIENT_SOURCE_SUPPORT", "FACT", "FACT")
            status = str(_value(verification, "status", "VERIFICATION_PENDING"))
            if status == "VERIFIED":
                return SignalDecision("NORMAL", "VERIFIED_FACT", "FACT", "FACT")
            if status == "PARTIALLY_VERIFIED":
                return SignalDecision("DEGRADED", "PARTIAL_VERIFICATION", "FACT", "FACT")
            if status == "CONTRADICTED":
                return SignalDecision("CONTRADICTION", "CONTRADICTED_FACT", "FACT", "FACT", "FACT_CONTRADICTION")
            return SignalDecision("SUPPRESSED", status or "VERIFICATION_PENDING", "FACT", "FACT")
        if claim_type == "FORECAST" or category == "FORECAST":
            if support == "SUPPORTED" and evidence and confidence >= self.forecast_confidence_threshold:
                return SignalDecision("NORMAL", "AUTHOR_FORECAST", "AUTHOR_FORECAST", "FORECAST")
            return SignalDecision("SUPPRESSED", "INSUFFICIENT_FORECAST_SUPPORT", "AUTHOR_FORECAST", "FORECAST")
        if claim_type == "OPINION" or category == "OPINION":
            if support == "SUPPORTED" and evidence:
                return SignalDecision("NORMAL", "AUTHOR_OPINION", "AUTHOR_OPINION", "OPINION")
            return SignalDecision("SUPPRESSED", "INSUFFICIENT_OPINION_SUPPORT", "AUTHOR_OPINION", "OPINION")
        if claim_type == "INFERENCE" or category == "INFERENCE":
            if support == "SUPPORTED" and evidence:
                return SignalDecision("NORMAL", "SYSTEM_INFERENCE", "SYSTEM_INFERENCE", "INFERENCE")
            return SignalDecision("SUPPRESSED", "INSUFFICIENT_INFERENCE_SUPPORT", "SYSTEM_INFERENCE", "INFERENCE")
        return SignalDecision("SUPPRESSED", "UNKNOWN_CLAIM_TYPE", "UNKNOWN", "UNKNOWN")

    def build_signal(
        self,
        snapshot: Any,
        claim: Any,
        verification: Any,
        *,
        verification_artifact_id: str | None = None,
        evidence_refs: list[str] | None = None,
        producer: dict[str, Any] | None = None,
        trace_id: str | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        decision = self.evaluate(claim, verification, snapshot=snapshot)
        snapshot_id = str(_value(snapshot, "content_snapshot_id", ""))
        claim_id = str(_value(claim, "claim_id", ""))
        verification_id = verification_artifact_id or str(
            _value(verification, "verification_artifact_id", "")
            or _value(verification, "artifact_id", "")
        )
        if not snapshot_id or not claim_id or not verification_id:
            raise ValueError("signal requires snapshot, claim, and verification artifact references")
        refs = list(evidence_refs or _value(claim, "evidence_refs", []) or [])
        source_type = str(_value(snapshot, "source_type", "unknown") or "unknown")
        source_ref = str(_value(snapshot, "source_ref", snapshot_id) or snapshot_id)
        source_artifact_id = str(_value(snapshot, "source_artifact_id", "") or f"source-{snapshot_id}")
        deterministic_decision_id = decision_id_v4(
            content_snapshot_id=snapshot_id,
            claim_id=claim_id,
            policy_version=self.version,
            signal_type=decision.signal_type,
        )
        manifest = dict(_value(snapshot, "producer_manifest", {}) or {})
        models = dict(_value(snapshot, "model_versions", {}) or {})
        prompts = dict(_value(snapshot, "prompt_versions", {}) or {})
        effective_decision_id = decision_id or deterministic_decision_id
        producer_payload = {
            "service": "stock_content",
            "service_version": "1.0.0",
            "code_sha": str(manifest.get("code_sha") or _value(snapshot, "code_sha", "unknown") or "unknown"),
            "container_digest": manifest.get("container_digest"),
            "dependency_lock_hash": manifest.get("dependency_lock_hash"),
            "pipeline_version": str(_value(snapshot, "pipeline_version", "pipeline.v3")),
            "model_id": str(models.get("llm_model") or _value(claim, "extraction_model_id", "unknown") or "unknown"),
            "prompt_version": str(
                prompts.get("knowledge_extraction")
                or prompts.get("extraction")
                or _value(claim, "extraction_prompt_version", "unknown")
                or "unknown"
            ),
            "trace_id": trace_id or "unknown",
            "decision_id": effective_decision_id,
            "decision": decision.reason,
        }
        producer_payload.update(dict(producer or {}))
        producer_payload["decision_id"] = effective_decision_id
        fact_time = _value(claim, "fact_time")
        published_at = _value(claim, "published_at")
        snapshot_created_at = _value(snapshot, "created_at")
        event_time = fact_time or published_at or snapshot_created_at
        published_time = published_at or snapshot_created_at
        return {
            "signal_id": signal_id_v4(
                content_snapshot_id=snapshot_id,
                claim_id=claim_id,
                policy_version=self.version,
                signal_type=decision.signal_type,
            ),
            "decision_id": effective_decision_id,
            "signal_schema_version": "content-factor-signal.v4",
            "signal_policy_version": self.version,
            "signal_status": decision.status,
            "signal_type": decision.signal_type,
            "fact_category": str(_value(claim, "fact_category", "")),
            "truth_scope": decision.truth_scope,
            "symbol": str(_value(claim, "subject_id", "")),
            "direction": "NEUTRAL",
            "magnitude": _value(claim, "value"),
            "confidence": float(_value(claim, "extractor_confidence", 0) or 0),
            "event_time": str(event_time or ""),
            "available_from": str(snapshot_created_at or ""),
            "published_at": str(published_time or ""),
            "source": {
                "source_artifact_id": source_artifact_id,
                "source_type": source_type,
                "source_ref": source_ref,
            },
            "content_snapshot_id": snapshot_id,
            "snapshot_id": snapshot_id,
            "claim_id": claim_id,
            "verification_artifact_id": verification_id,
            "evidence_refs": refs,
            "support": {
                "status": str(_value(claim, "source_support_status", "UNSUPPORTED")),
                "score": float(_value(claim, "source_confidence", 0) or 0),
                "evidence_refs": refs,
            },
            "verification": {
                "status": str(_value(verification, "status", "")),
                "provider": str(_value(verification, "provider", "quant") or "quant"),
                "market_snapshot_id": _value(verification, "market_snapshot_id"),
                "market_data_version": _value(verification, "market_data_version"),
                "verification_rule_version": _value(verification, "verification_rule_version", "verification_rule.v1"),
            },
            "producer": producer_payload,
            "policy": {
                "signal_policy_version": self.version,
                "forecast_confidence_threshold": self.forecast_confidence_threshold,
            },
            "event_type": decision.event_type,
        }

    def build_initial_signals(
        self, snapshot: Any, claims: list[Any], verification: Any,
        *, trace_id: str | None = None, decision_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Build a deterministic signal projection for every claim in a snapshot."""
        results = {
            str(_value(item, "claim_id", "")): item
            for item in list(_value(verification, "results", []) or [])
        }
        artifact_id = str(_value(verification, "artifact_id", ""))
        signals: list[dict[str, Any]] = []
        seen: set[str] = set()
        for claim in claims:
            claim_id = str(_value(claim, "claim_id", ""))
            result = results.get(claim_id, {"claim_id": claim_id, "status": "VERIFICATION_PENDING"})
            signal = self.build_signal(
                snapshot,
                claim,
                result,
                verification_artifact_id=artifact_id,
                trace_id=trace_id,
                decision_id=decision_id,
            )
            if signal["signal_id"] not in seen:
                signals.append(signal)
                seen.add(signal["signal_id"])
        return signals

    # Explicit names used by initial ingest and the Postgres rebuild command.
    build_initial = build_initial_signals
    rebuild = build_initial_signals


__all__ = [
    "FORECAST_CONFIDENCE_THRESHOLD",
    "SIGNAL_POLICY_VERSION",
    "SignalDecision",
    "SignalPolicy",
    "decision_id_v4",
    "signal_id_v4",
]

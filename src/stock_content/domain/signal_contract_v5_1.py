"""Strict producer contract for formal content-factor signals."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from stock_content.domain.artifacts import canonical_json

CONTRACT_NAME = "content-factor-signal.v5.1"
AUTHORITY_FORMAL = "FORMAL_FACT"
AUTHORITY_COMPATIBILITY = "COMPATIBILITY_READ_ONLY"
PIT_MODE = "PUBLIC_STRICT"

_REQUIRED = {
    "contract",
    "contract_checksum",
    "authority",
    "formal_eligible",
    "signal_id",
    "claim_id",
    "occurrence_id",
    "semantic_segment_id",
    "asserted_at",
    "source_available_at",
    "source_availability_quality",
    "available_from",
    "temporal_bindings",
    "lifecycle_as_of",
    "content_snapshot_id",
    "evidence_refs",
    "producer_commit",
    "signal_policy_version",
    "business_as_of",
    "knowledge_as_of",
    "availability_as_of",
}


def canonical_signal_json(value: Mapping[str, Any]) -> str:
    return canonical_json(dict(value))


def checksum_for_contract(schema: Mapping[str, Any] | None = None) -> str:
    if schema is None:
        schema_path = Path(__file__).parents[3] / "contracts" / "content-factor-signal.v5.1.json"
        if not schema_path.exists():
            # Repository-level contracts are not necessarily included in a
            # wheel; retain the same release checksum in that environment.
            return "bc4a37eda4a48977c11f229eed1166d1b3bd3b04a17c2c96f90666026ebe804c"
        return hashlib.sha256(schema_path.read_bytes()).hexdigest()
    return hashlib.sha256(canonical_json(dict(schema)).encode("utf-8")).hexdigest()


CONTRACT_CHECKSUM = checksum_for_contract()


def formal_signal_id(
    *,
    claim_id: str,
    occurrence_id: str,
    semantic_segment_id: str,
    content_snapshot_id: str,
    business_as_of: str,
    knowledge_as_of: str,
    availability_as_of: str,
    query_id: str,
    signal_policy_version: str,
) -> str:
    """Return the immutable identity of a formal signal projection.

    The formal id is deliberately a new namespace.  In particular, a
    legacy/materialized signal id is never accepted as an input: a projection
    changes identity whenever any PIT clock, snapshot, query or policy changes.
    """
    identity = {
        "claim_id": claim_id,
        "occurrence_id": occurrence_id,
        "semantic_segment_id": semantic_segment_id,
        "content_snapshot_id": content_snapshot_id,
        "business_as_of": business_as_of,
        "knowledge_as_of": knowledge_as_of,
        "availability_as_of": availability_as_of,
        "query_id": query_id,
        "signal_policy_version": signal_policy_version,
    }
    return "sig51_" + hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def signal_checksum(signal: Mapping[str, Any]) -> str:
    payload = dict(signal)
    payload.pop("signal_checksum", None)
    return hashlib.sha256(canonical_signal_json(payload).encode("utf-8")).hexdigest()


def validate_signal_v5_1(signal: Mapping[str, Any], *, expected_checksum: str | None = None) -> dict[str, Any]:
    if not isinstance(signal, Mapping):
        raise ValueError("v5.1 signal must be an object")
    missing = sorted(key for key in _REQUIRED if key not in signal)
    if missing:
        raise ValueError(f"v5.1 signal missing {', '.join(missing)}")
    unexpected = sorted(set(signal) - _REQUIRED)
    if unexpected:
        raise ValueError(f"v5.1 signal contains unsupported fields: {', '.join(unexpected)}")
    if signal["contract"] != CONTRACT_NAME:
        raise ValueError("unsupported formal contract")
    if signal["authority"] != AUTHORITY_FORMAL or signal["formal_eligible"] is not True:
        raise ValueError("formal signal must be FORMAL_FACT and formal_eligible=true")
    checksum = expected_checksum or CONTRACT_CHECKSUM
    if signal["contract_checksum"] != checksum:
        raise ValueError("contract checksum mismatch")
    for key in (
        "signal_id",
        "claim_id",
        "occurrence_id",
        "semantic_segment_id",
        "content_snapshot_id",
        "producer_commit",
        "signal_policy_version",
    ):
        if not isinstance(signal[key], str) or not signal[key]:
            raise ValueError(f"v5.1 {key} must be a non-empty string")
    if signal["source_availability_quality"] not in {
        "EXACT",
        "PROXY",
        "PUBLISHED_TIME_PROXY",
        "INGEST_TIME_UPPER_BOUND",
    }:
        raise ValueError("unsupported source_availability_quality")
    for key in ("business_as_of", "knowledge_as_of", "availability_as_of"):
        _rfc3339_aware(signal[key], key)
    # ``available_from`` is the lower bound used by PIT consumers and is not
    # nullable.  The two source timestamps remain nullable by contract.
    if signal["available_from"] is None:
        raise ValueError("available_from must be an RFC3339 datetime")
    _rfc3339_aware(signal["available_from"], "available_from")
    for key in ("asserted_at", "source_available_at"):
        if signal[key] is not None:
            _rfc3339_aware(signal[key], key)
    if signal["lifecycle_as_of"] is None or not isinstance(signal["lifecycle_as_of"], Mapping):
        raise ValueError("lifecycle_as_of must be an object")
    for key in ("status", "known_from", "artifact_id"):
        if key not in signal["lifecycle_as_of"]:
            raise ValueError(f"lifecycle_as_of missing {key}")
        if not isinstance(signal["lifecycle_as_of"][key], str) or not signal["lifecycle_as_of"][key]:
            raise ValueError(f"lifecycle_as_of.{key} must be a non-empty string")
    _rfc3339_aware(signal["lifecycle_as_of"]["known_from"], "lifecycle_as_of.known_from")
    if not isinstance(signal["temporal_bindings"], list) or not isinstance(signal["evidence_refs"], list):
        raise ValueError("temporal_bindings and evidence_refs must be arrays")
    return dict(signal)


def _rfc3339_aware(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an RFC3339 datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC3339 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


__all__ = [
    "AUTHORITY_COMPATIBILITY",
    "AUTHORITY_FORMAL",
    "CONTRACT_CHECKSUM",
    "CONTRACT_NAME",
    "PIT_MODE",
    "canonical_signal_json",
    "checksum_for_contract",
    "formal_signal_id",
    "signal_checksum",
    "validate_signal_v5_1",
]

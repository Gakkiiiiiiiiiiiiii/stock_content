"""Vendored strict consumer checks used by producer contract CI."""
from __future__ import annotations

from typing import Any

from stock_content.domain.signal_contract_v5_1 import validate_signal_v5_1


REQUIRED = frozenset({
    "contract", "contract_checksum", "authority", "formal_eligible", "signal_id",
    "claim_id", "occurrence_id", "content_snapshot_id", "business_as_of",
    "knowledge_as_of", "availability_as_of",
})
SCHEMA_REQUIRED = frozenset({
    "contract", "contract_checksum", "authority", "formal_eligible", "signal_id", "claim_id",
    "occurrence_id", "semantic_segment_id", "asserted_at", "source_available_at",
    "source_availability_quality", "available_from", "temporal_bindings", "lifecycle_as_of",
    "content_snapshot_id", "evidence_refs", "producer_commit", "signal_policy_version",
    "business_as_of", "knowledge_as_of", "availability_as_of",
})


def accept_formal_signal(payload: dict[str, Any]) -> bool:
    """Strictly accept only complete v5.1 formal facts."""
    if not SCHEMA_REQUIRED.issubset(payload) or set(payload) != SCHEMA_REQUIRED:
        return False
    if not REQUIRED.issubset(payload):
        return False
    try:
        validate_signal_v5_1(payload)
    except (TypeError, ValueError):
        return False
    lifecycle = payload["lifecycle_as_of"]
    return bool(
        isinstance(lifecycle, dict)
        and isinstance(payload["temporal_bindings"], list)
        and isinstance(payload["evidence_refs"], list)
    )


__all__ = ["accept_formal_signal"]

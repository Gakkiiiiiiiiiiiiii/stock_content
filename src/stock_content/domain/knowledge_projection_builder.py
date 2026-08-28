from __future__ import annotations

from typing import Any

from .claim_occurrence import ClaimOccurrence
from .claims import FinancialClaim


class KnowledgeProjectionBuilder:
    def build(
        self,
        claim: FinancialClaim,
        occurrence: ClaimOccurrence | None = None,
        verification: Any = None,
        lifecycle: Any = None,
    ) -> dict[str, Any]:
        """Produce a read/search payload without making it a source of truth."""
        # A canonical claim may have multiple source occurrences.  The
        # occurrence-scoped uid keeps those projections independently
        # addressable while preserving claim_id as the stable semantic key.
        knowledge_uid = occurrence.occurrence_id if occurrence is not None else claim.claim_id
        payload = {
            "knowledge_uid": knowledge_uid,
            "statement": claim.predicate,
            "kind": "CLAIM",
            "knowledge_kind": claim.fact_category,
            "subject": claim.subject_id,
            "subject_key": claim.subject_id,
            "predicate_key": claim.predicate,
            "ticker": claim.ticker,
            "support_status": claim.source_support_status,
            "attributes": {
                "claim_id": claim.claim_id,
                "temporal_bindings": [x.model_dump(mode="json") for x in claim.temporal_bindings],
                "temporal_relations": [x.model_dump(mode="json") for x in claim.temporal_relations],
            },
        }
        if occurrence:
            payload["attributes"].update({
                "occurrence_id": occurrence.occurrence_id,
                "semantic_segment_id": occurrence.semantic_segment_id,
                "asserted_at": (
                    occurrence.times.asserted_at.isoformat()
                    if occurrence.times.asserted_at is not None else None
                ),
                "source_published_at": (
                    occurrence.times.source_published_at.isoformat()
                    if occurrence.times.source_published_at is not None else None
                ),
            })
            payload["available_from"] = occurrence.times.available_from
        if verification is not None:
            payload["attributes"]["verification"] = (
                verification.model_dump(mode="json") if hasattr(verification, "model_dump") else verification
            )
        if lifecycle is not None:
            payload["lifecycle_status"] = getattr(lifecycle, "to_status", lifecycle)
        return payload

    project = build


__all__ = ["KnowledgeProjectionBuilder"]

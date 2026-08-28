"""Projection from grounded extraction to evidence-independent canonical claims."""

from __future__ import annotations

import hashlib
from typing import Any

from .artifacts import canonical_json
from .claim_draft import ClaimOccurrenceDraft
from .claims import CLAIM_CATEGORY, FinancialClaim


def condition_key_of(condition_text: str | None, bindings: list[Any] | None = None) -> str | None:
    condition_bindings = [
        x
        for x in (bindings or [])
        if getattr(getattr(x, "role", None), "value", getattr(x, "role", None)) == "CONDITION_PERIOD"
    ]
    if not condition_text and not condition_bindings:
        return None
    return hashlib.sha256(
        canonical_json(
            {
                "condition": " ".join((condition_text or "").split()).casefold(),
                "bindings": sorted(x.temporal_binding_id for x in condition_bindings),
            }
        ).encode()
    ).hexdigest()


class ClaimCanonicalizer:
    def __init__(self, normalization_version: str | None = None) -> None:
        # The temporal normalizer is the authority for this version.  A
        # configured value is used by the production stage; direct domain
        # callers can derive it from the normalized bindings below.
        self.normalization_version = normalization_version

    def canonicalize(
        self,
        draft: ClaimOccurrenceDraft,
        *,
        evidence_refs: list[str] | None = None,
        temporal_bindings: list[Any] | None = None,
        temporal_relations: list[Any] | None = None,
        normalization_version: str | None = None,
    ) -> FinancialClaim:
        bindings = list(temporal_bindings or [])
        relations = list(temporal_relations or [])
        binding_versions = {
            str(getattr(binding, "normalization_version", ""))
            for binding in bindings
            if getattr(binding, "normalization_version", None)
        }
        configured_version = normalization_version or self.normalization_version
        if configured_version and binding_versions and binding_versions != {configured_version}:
            raise ValueError(
                "canonical claim normalization version does not match temporal bindings"
            )
        effective_normalization_version = (
            configured_version
            or next(iter(binding_versions), None)
            or "normalization.v1"
        )
        claim = FinancialClaim(
            claim_type=draft.claim_type,
            # Stage 2 may not promote an arbitrary knowledge_kind into the
            # canonical fact layer.  Category is a deterministic projection
            # of the claim type, including FINANCIAL_METRIC -> FACT.
            fact_category=CLAIM_CATEGORY[draft.claim_type],
            subject_type=draft.subject_type or "UNKNOWN",
            subject_id=draft.subject_key,
            ticker=draft.subject_key if draft.subject_key else None,
            predicate=draft.predicate_key,
            value=draft.value,
            unit=draft.unit,
            currency=draft.currency,
            condition_text=draft.condition_text,
            invalidation_text=draft.invalidation_text,
            condition_key=condition_key_of(draft.condition_text, bindings),
            temporal_bindings=bindings,
            temporal_relations=relations,
            # Final canonical claims intentionally own no source evidence.
            evidence_refs=[],
            source_support_status="SUPPORTED",
            source_confidence=draft.extraction_confidence,
            extractor_confidence=draft.extraction_confidence,
            claim_schema_version="claim.final.v1",
            normalization_version=effective_normalization_version,
        )
        return claim


canonicalize_claim = ClaimCanonicalizer().canonicalize
canonicalize = canonicalize_claim

__all__ = ["ClaimCanonicalizer", "canonicalize_claim", "canonicalize", "condition_key_of"]

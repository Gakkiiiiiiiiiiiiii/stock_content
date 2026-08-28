from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .artifacts import ClaimOccurrenceArtifact, canonical_json
from .temporal_semantics import AvailabilityQuality, OccurrenceTimes


def assertion_locator_hash_of(
    source_artifact_id: str,
    transcript_artifact_id: str,
    semantic_segment_id: str,
    evidence_refs: list[str],
    temporal_evidence_refs: list[str] | None = None,
) -> str:
    # Temporal refs are another relationship to the same immutable source
    # evidence.  Union them into one coordinate set; role assignment must not
    # alter an assertion locator or occurrence identity.
    stable_refs = sorted(set(evidence_refs) | set(temporal_evidence_refs or []))
    payload = {
        "source_artifact_id": source_artifact_id,
        "transcript_artifact_id": transcript_artifact_id,
        "semantic_segment_id": semantic_segment_id,
        "evidence_refs": stable_refs,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


class ClaimOccurrence(BaseModel):
    occurrence_id: str = ""
    claim_id: str
    source_artifact_id: str
    transcript_artifact_id: str
    semantic_segment_id: str
    assertion_locator_hash: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    condition_evidence_refs: list[str] = Field(default_factory=list)
    invalidation_evidence_refs: list[str] = Field(default_factory=list)
    temporal_evidence_refs: list[str] = Field(default_factory=list)
    times: OccurrenceTimes
    source_support_status: str = "SOURCE_LOCATED"
    source_confidence: float = Field(default=0.0, ge=0, le=1)
    extractor_confidence: float = Field(default=0.0, ge=0, le=1)
    raw_temporal_expressions: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _identity(self) -> "ClaimOccurrence":
        all_refs = self.evidence_refs + self.condition_evidence_refs + self.invalidation_evidence_refs
        locator = self.assertion_locator_hash or assertion_locator_hash_of(
            self.source_artifact_id,
            self.transcript_artifact_id,
            self.semantic_segment_id,
            all_refs,
            self.temporal_evidence_refs,
        )
        if not self.assertion_locator_hash:
            object.__setattr__(self, "assertion_locator_hash", locator)
        if not self.occurrence_id:
            object.__setattr__(
                self,
                "occurrence_id",
                "co_"
                + hashlib.sha256(
                    canonical_json(
                        {
                            "claim_id": self.claim_id,
                            "source_artifact_id": self.source_artifact_id,
                            "assertion_locator_hash": locator,
                        }
                    ).encode()
                ).hexdigest(),
            )
        return self


def occurrence_id_of(claim_id: str, source_artifact_id: str, assertion_locator_hash: str) -> str:
    payload = {
        "claim_id": claim_id,
        "source_artifact_id": source_artifact_id,
        "assertion_locator_hash": assertion_locator_hash,
    }
    return "co_" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()


__all__ = [
    "ClaimOccurrence",
    "ClaimOccurrenceArtifact",
    "OccurrenceTimes",
    "AvailabilityQuality",
    "assertion_locator_hash_of",
    "occurrence_id_of",
]

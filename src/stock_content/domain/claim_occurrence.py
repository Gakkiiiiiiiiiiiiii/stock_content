from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .artifacts import ClaimOccurrenceArtifact, canonical_json
from .temporal_semantics import AvailabilityQuality, OccurrenceTimes

_OCCURRENCE_ID_PREFIX = "co_"
_OCCURRENCE_ID_DIGEST_LENGTH = 61
_OCCURRENCE_ID_MAX_LENGTH = len(_OCCURRENCE_ID_PREFIX) + _OCCURRENCE_ID_DIGEST_LENGTH


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
    secondary_evidence_refs: list[str] = Field(default_factory=list)
    condition_evidence_refs: list[str] = Field(default_factory=list)
    invalidation_evidence_refs: list[str] = Field(default_factory=list)
    temporal_evidence_refs: list[str] = Field(default_factory=list)
    times: OccurrenceTimes
    source_support_status: str = "SOURCE_LOCATED"
    source_confidence: float = Field(default=0.0, ge=0, le=1)
    extractor_confidence: float = Field(default=0.0, ge=0, le=1)
    raw_temporal_expressions: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    primary_quote: str | None = None
    normalized_statement: str | None = None
    grounding_status: str = "LEGACY_UNGROUNDED"
    grounding_reason_codes: list[str] = Field(default_factory=list)
    contradiction_group_id: str | None = None
    claim_schema_version: str = "claim.legacy.v1"
    legacy_grounding_incomplete: bool = True

    @model_validator(mode="after")
    def _identity(self) -> "ClaimOccurrence":
        if self.grounding_status == "GROUNDED":
            if not self.evidence_refs or not self.primary_quote or not self.normalized_statement:
                raise ValueError("grounded occurrence requires primary evidence, quote, and normalized statement")
            if self.legacy_grounding_incomplete:
                raise ValueError("grounded occurrence cannot be marked legacy incomplete")
        all_refs = (
            self.evidence_refs + self.secondary_evidence_refs + self.condition_evidence_refs
            + self.invalidation_evidence_refs
        )
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
                _OCCURRENCE_ID_PREFIX
                + hashlib.sha256(
                    canonical_json(
                        {
                            "claim_id": self.claim_id,
                            "source_artifact_id": self.source_artifact_id,
                            "assertion_locator_hash": locator,
                        }
                    ).encode()
                ).hexdigest()[:_OCCURRENCE_ID_DIGEST_LENGTH],
            )
        return self


def occurrence_id_of(claim_id: str, source_artifact_id: str, assertion_locator_hash: str) -> str:
    payload = {
        "claim_id": claim_id,
        "source_artifact_id": source_artifact_id,
        "assertion_locator_hash": assertion_locator_hash,
    }
    return _OCCURRENCE_ID_PREFIX + hashlib.sha256(
        canonical_json(payload).encode()
    ).hexdigest()[:_OCCURRENCE_ID_DIGEST_LENGTH]


def knowledge_uid_for_occurrence(occurrence_id: str) -> str:
    """Return the <=64-character knowledge key for current and legacy occurrences.

    Earlier occurrences used the full 64-character digest after ``co_``.
    The durable occurrence row keeps that historical identifier, while its
    knowledge projection compacts the digest exactly as current generation
    does.  This makes old snapshot replay and new ingestion converge without
    widening the knowledge-unit primary key.
    """
    value = str(occurrence_id or "")
    if value.startswith(_OCCURRENCE_ID_PREFIX):
        digest = value[len(_OCCURRENCE_ID_PREFIX):]
        if len(digest) >= _OCCURRENCE_ID_DIGEST_LENGTH and all(char in "0123456789abcdefABCDEF" for char in digest):
            return _OCCURRENCE_ID_PREFIX + digest[:_OCCURRENCE_ID_DIGEST_LENGTH].lower()
    if len(value) <= _OCCURRENCE_ID_MAX_LENGTH:
        return value
    return _OCCURRENCE_ID_PREFIX + hashlib.sha256(value.encode("utf-8")).hexdigest()[:_OCCURRENCE_ID_DIGEST_LENGTH]


__all__ = [
    "ClaimOccurrence",
    "ClaimOccurrenceArtifact",
    "OccurrenceTimes",
    "AvailabilityQuality",
    "assertion_locator_hash_of",
    "knowledge_uid_for_occurrence",
    "occurrence_id_of",
]

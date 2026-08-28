"""Fail-closed grounding of Stage 2 coordinate-only drafts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .artifacts import EvidenceItem, TranscriptArtifact, canonical_json
from .claim_draft import ClaimOccurrenceDraft
from .semantic_segment import SemanticSegment


@dataclass(frozen=True)
class GroundedClaimOccurrence:
    draft: ClaimOccurrenceDraft
    evidences: tuple[EvidenceItem, ...]
    primary_evidence_refs: tuple[str, ...] = ()
    condition_evidence_refs: tuple[str, ...] = ()
    invalidation_evidence_refs: tuple[str, ...] = ()
    temporal_evidence_refs: tuple[str, ...] = ()

    @property
    def evidence_refs_by_role(self) -> dict[str, tuple[str, ...]]:
        """Relationship refs; roles intentionally do not affect evidence IDs."""
        return {
            "PRIMARY": self.primary_evidence_refs,
            "CONDITION": self.condition_evidence_refs,
            "INVALIDATION": self.invalidation_evidence_refs,
            "TEMPORAL": self.temporal_evidence_refs,
        }


class ClaimDraftGrounder:
    def ground(
        self, draft: ClaimOccurrenceDraft, transcript: TranscriptArtifact, semantic_segment: SemanticSegment
    ) -> GroundedClaimOccurrence:
        if draft.semantic_segment_id != semantic_segment.semantic_segment_id:
            raise ValueError("draft does not belong to semantic segment")
        by_index = {item.segment_index: item for item in transcript.segments}
        valid_range = set(range(semantic_segment.start_segment_index, semantic_segment.end_segment_index + 1))
        groups = {
            "PRIMARY": draft.evidence_segment_indices,
            "CONDITION": draft.condition_evidence_segment_indices,
            "INVALIDATION": draft.invalidation_evidence_segment_indices,
        }
        all_indices = [index for values in groups.values() for index in values]
        if not all_indices:
            raise ValueError("claim draft requires evidence coordinates")
        if any(index not in by_index or index not in valid_range for index in all_indices):
            raise ValueError("evidence coordinate is absent or outside semantic segment")
        texts = " ".join(by_index[index].raw_text or by_index[index].text for index in sorted(set(all_indices)))
        if draft.conclusion and not self._locatable(draft.conclusion, texts):
            raise ValueError("claim conclusion is not locatable in evidence")
        for field_name, text, coords in (
            ("condition", draft.condition_text, draft.condition_evidence_segment_indices),
            ("invalidation", draft.invalidation_text, draft.invalidation_evidence_segment_indices),
        ):
            if text and (
                not coords
                or not self._locatable(text, " ".join((by_index[i].raw_text or by_index[i].text) for i in coords))
            ):
                raise ValueError(f"{field_name} is not locatable in evidence")
        for expression in draft.temporal_expressions:
            if not expression.evidence_segment_indices or any(
                i not in valid_range or i not in by_index for i in expression.evidence_segment_indices
            ):
                raise ValueError("temporal expression requires in-range evidence coordinates")
            expression_text = " ".join(
                (by_index[i].raw_text or by_index[i].text) for i in expression.evidence_segment_indices
            )
            if not self._locatable(expression.raw_expression, expression_text):
                raise ValueError("temporal expression is not locatable in evidence")
        # Numeric and ticker tokens are hard grounding requirements.
        for token in re.findall(r"(?<!\w)(?:\d+(?:[.,]\d+)?%?|[A-Z]{1,6}(?:[.][A-Z]{1,4})?)(?!\w)", draft.conclusion):
            if token not in texts:
                raise ValueError(f"token {token!r} is not locatable in evidence")
        # Evidence is a source coordinate, not a claim relationship.  Build it
        # exactly once per segment so assigning the same segment to two roles
        # cannot manufacture different evidence identities.
        role_indices = {
            role: tuple(sorted(set(indices))) for role, indices in groups.items()
        }
        temporal_indices = tuple(
            sorted(
                {
                    index
                    for expression in draft.temporal_expressions
                    for index in expression.evidence_segment_indices
                }
            )
        )
        all_evidence_indices = sorted(
            {index for values in role_indices.values() for index in values} | set(temporal_indices)
        )
        evidence_by_index: dict[int, EvidenceItem] = {}
        for index in all_evidence_indices:
            item = by_index[index]
            raw = item.raw_text or item.text
            eid = (
                "ev_"
                + hashlib.sha256(
                    canonical_json(
                        {
                            "source_artifact_id": transcript.artifact_id,
                            "segment_id": item.segment_id,
                            "segment_index": index,
                            "raw_text": raw,
                        }
                    ).encode()
                ).hexdigest()
            )
            evidence_by_index[index] = EvidenceItem(
                evidence_id=eid,
                source_type="ASR",
                source_artifact_id=transcript.artifact_id,
                evidence_text=raw,
                raw_text=raw,
                normalized_text=item.normalized_text or item.text,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                locator={
                    "segment_id": item.segment_id,
                    "segment_index": index,
                    "semantic_segment_id": semantic_segment.semantic_segment_id,
                },
            )
        refs = {index: evidence.evidence_id for index, evidence in evidence_by_index.items()}
        temporal_refs = tuple(refs[index] for index in temporal_indices)
        return GroundedClaimOccurrence(
            draft=draft,
            evidences=tuple(evidence_by_index[index] for index in all_evidence_indices),
            primary_evidence_refs=tuple(refs[index] for index in role_indices["PRIMARY"]),
            condition_evidence_refs=tuple(refs[index] for index in role_indices["CONDITION"]),
            invalidation_evidence_refs=tuple(refs[index] for index in role_indices["INVALIDATION"]),
            temporal_evidence_refs=temporal_refs,
        )

    @staticmethod
    def _locatable(needle: str, haystack: str) -> bool:
        return needle.casefold().strip() in haystack.casefold()

    materialize = ground


__all__ = ["ClaimDraftGrounder", "GroundedClaimOccurrence"]

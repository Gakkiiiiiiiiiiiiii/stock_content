"""Stage 1 semantic segmentation with a fail-closed boundary protocol.

The segmenter deliberately knows nothing about claims.  A model may propose
only boundary coordinates; materialization and coverage validation remain
deterministic domain operations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any

from .artifacts import TranscriptArtifact, artifact_id_of
from .semantic_boundary_validator import validate_boundaries, validate_full_coverage
from .semantic_segment import (
    SemanticBoundary,
    SemanticSegment,
    build_semantic_segment_artifact,
    materialize_semantic_segments,
)


@dataclass(frozen=True)
class SemanticSegmentationResult:
    artifact: Any
    segments: tuple[SemanticSegment, ...]
    metrics: dict[str, float]


class SemanticSegmenter:
    """Produce a full-coverage SemanticSegmentArtifact.

    ``model_gateway`` is optional for deterministic/offline fixtures.  The
    gateway is called once for a short transcript and once per bounded block
    for long input; invalid output gets exactly one schema-repair attempt.
    """

    name = "semantic_segmentation"
    schema_version = "semantic-segment.v1"

    def __init__(
        self,
        model_gateway: Any | None = None,
        *,
        model_id: str = "",
        prompt_version: str = "semantic-segmentation.v1",
        safe_tokens: int = 3200,
        block_tokens: int | None = None,
        segment_overlap: int = 2,
        allow_offline_fixture: bool = True,
    ) -> None:
        if safe_tokens <= 0 or (block_tokens is not None and block_tokens <= 0):
            raise ValueError("token budgets must be positive")
        if segment_overlap < 0:
            raise ValueError("segment_overlap must be non-negative")
        self.model_gateway = model_gateway
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.safe_tokens = safe_tokens
        self.block_tokens = block_tokens or safe_tokens
        self.segment_overlap = segment_overlap
        self.allow_offline_fixture = allow_offline_fixture
        self.last_metrics: dict[str, float] = {}

    def segment(
        self, transcript: TranscriptArtifact, *, offline_fixture: bool = False
    ) -> SemanticSegmentationResult:
        items = list(transcript.segments)
        if not items:
            artifact = build_semantic_segment_artifact(
                transcript, (), model_id=self.model_id, prompt_version=self.prompt_version,
                schema_version=self.schema_version,
            )
            artifact = replace(artifact, parent_artifact_ids=(transcript.artifact_id,), artifact_id="", content_hash="")
            artifact = replace(artifact, artifact_id=artifact_id_of(artifact))
            self.last_metrics = {"segment_count": 0.0, "repair_count": 0.0, "failure_count": 0.0}
            return SemanticSegmentationResult(artifact, (), dict(self.last_metrics))

        available = self.model_gateway is not None and bool(
            getattr(self.model_gateway, "available", lambda: True)()
        )
        if not available and not (self.allow_offline_fixture or offline_fixture):
            raise RuntimeError("semantic segmentation model gateway is unavailable")
        if not available:
            boundaries: list[SemanticBoundary] = []
            repair_count = 0
        elif self._token_count(transcript) <= self.safe_tokens:
            boundaries, repair_count = self._call_with_repair(items, 0, len(items))
        else:
            proposals: list[SemanticBoundary] = []
            repair_count = 0
            for start, end in self._blocks(items):
                block, repairs = self._call_with_repair(items, start, end)
                proposals.extend(block)
                repair_count += repairs
            # Reconciliation is deterministic: coordinates are deduplicated,
            # sorted and validated globally; no proposal is clamped.  A model
            # can place one boundary at either side of an overlap, so nearby
            # conflicting coordinates are adjudicated as one candidate.
            boundaries = self._reconcile(proposals, overlap=self.segment_overlap)

        segments = materialize_semantic_segments(
            transcript,
            boundaries,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
        )
        validate_full_coverage(segments, len(items))
        artifact = build_semantic_segment_artifact(
            transcript,
            boundaries,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
        )
        # Artifact parent linkage is part of the authoritative chain.
        artifact = replace(artifact, parent_artifact_ids=(transcript.artifact_id,), artifact_id="", content_hash="")
        artifact = replace(artifact, artifact_id=artifact_id_of(artifact))
        self.last_metrics = {
            "segment_count": float(len(segments)),
            "repair_count": float(repair_count),
            "failure_count": 0.0,
            "boundary_count": float(len(boundaries)),
        }
        return SemanticSegmentationResult(artifact, tuple(segments), dict(self.last_metrics))

    materialize = segment

    def _blocks(self, items: list[Any]) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = 0
        while start < len(items):
            budget = 0
            end = start
            while end < len(items):
                item = items[end]
                text = item.raw_text or item.text or ""
                cost = max(1, len(text) // 4, len(text.split()))
                if end > start and budget + cost > self.block_tokens:
                    break
                budget += cost
                end += 1
            end = max(start + 1, end)
            end = min(len(items), end)
            blocks.append((start, end))
            if end == len(items):
                break
            start = max(start + 1, end - self.segment_overlap)
        return blocks

    @staticmethod
    def _reconcile(
        proposals: list[SemanticBoundary], *, overlap: int = 2
    ) -> list[SemanticBoundary]:
        """Adjudicate block proposals without depending on response order.

        Exact coordinates are first collapsed and retain a support count (the
        number of blocks that proposed that coordinate).  Coordinates within
        the block overlap form a conflict cluster.  The strongest candidate
        is selected by a total ordering of confidence, repeated support and
        metadata completeness.  A coordinate independently repeated by more
        than one block is retained alongside another independently repeated
        coordinate; this protects genuinely adjacent topic changes from
        being swallowed by reconciliation.  No coordinate is clamped or
        inferred.
        """
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        grouped: dict[int, list[SemanticBoundary]] = {}
        for proposal in proposals:
            if isinstance(proposal.after_segment_index, bool) or not isinstance(
                proposal.after_segment_index, int
            ):
                raise ValueError("boundary after_segment_index must be an int")
            if proposal.after_segment_index < 0:
                raise ValueError("boundary after_segment_index must be non-negative")
            SemanticSegmenter._validate_metadata(proposal)
            grouped.setdefault(proposal.after_segment_index, []).append(proposal)

        candidates: list[tuple[int, int, SemanticBoundary]] = []
        for index, values in grouped.items():
            representative = max(
                values,
                key=lambda item: SemanticSegmenter._proposal_rank(item, len(values)),
            )
            candidates.append((index, len(values), representative))
        candidates.sort(key=lambda item: item[0])

        result: list[SemanticBoundary] = []
        cursor = 0
        while cursor < len(candidates):
            cluster = [candidates[cursor]]
            cursor += 1
            while cursor < len(candidates) and candidates[cursor][0] - cluster[-1][0] <= overlap:
                cluster.append(candidates[cursor])
                cursor += 1

            if len(cluster) == 1:
                selected = cluster
            else:
                # Repeated support is evidence that two close coordinates
                # represent independent boundaries.  A singleton proposal in
                # the same cluster is treated as an overlap disagreement and
                # loses to the repeated candidate(s).
                repeated = [item for item in cluster if item[1] > 1]
                selected = repeated or [
                    max(
                        cluster,
                        key=lambda item: SemanticSegmenter._proposal_rank(
                            item[2], item[1]
                        ),
                    )
                ]
            result.extend(item[2] for item in selected)
        return result

    @staticmethod
    def _proposal_rank(proposal: SemanticBoundary, support: int) -> tuple[Any, ...]:
        """Return a total, order-independent ranking for one proposal."""
        confidence = proposal.confidence
        confidence_rank = confidence if confidence is not None else -math.inf
        return (
            confidence_rank,
            support,
            int(bool(proposal.next_subject)),
            int(bool(proposal.next_topic)),
            proposal.boundary_type or "",
            proposal.next_subject or "",
            proposal.next_topic or "",
        )

    @staticmethod
    def _validate_metadata(proposal: SemanticBoundary) -> None:
        if not isinstance(proposal.boundary_type, str) or not proposal.boundary_type:
            raise ValueError("boundary_type must be a non-empty string")
        for name in ("next_topic", "next_subject"):
            value = getattr(proposal, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
        confidence = proposal.confidence
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("confidence must be a finite number between 0 and 1")

    @staticmethod
    def _token_count(transcript: TranscriptArtifact) -> int:
        text = " ".join(item.raw_text or item.text for item in transcript.segments)
        # Conservative approximation that works for both whitespace-delimited
        # languages and CJK transcripts.
        return max(1, len(text) // 4, len(text.split()))

    def _call_with_repair(self, items: list[Any], start: int, end: int) -> tuple[list[SemanticBoundary], int]:
        prompt = self._prompt(items, start, end)
        response = self._complete(prompt)
        try:
            return self._parse(response, start, end), 0
        except (TypeError, ValueError, json.JSONDecodeError):
            repair = self._complete(prompt + "\n上一次输出无效。仅修复 JSON schema，仍不得输出 claim 或 timestamp。")
            try:
                return self._parse(repair, start, end), 1
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("semantic segmentation boundary protocol failed after one repair") from exc

    def _complete(self, prompt: str) -> Any:
        gateway = self.model_gateway
        try:
            return gateway.complete(
                prompt=prompt,
                system="You are a semantic boundary detector. Return JSON only.",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except TypeError:
            return gateway.complete(
                messages=[
                    {"role": "system", "content": "You are a semantic boundary detector. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

    def _prompt(self, items: list[Any], start: int, end: int) -> str:
        lines = [
            {"segment_index": item.segment_index, "text": item.raw_text or item.text}
            for item in items[start:end]
        ]
        return (
            "You are Stage 1 semantic segmentation. Return boundaries only, never claims, values, dates, "
            "timestamps, evidence, summaries, or copied transcript text. Produce full contiguous coverage: "
            "a new segment starts only at a real topic/subject/conclusion change. Keep one thesis, its evidence, "
            "risks and conditions together; separate advertisements, disclaimers, greetings and unrelated Q&A. "
            "For each boundary give the next topic and subject when known. Do not invent entities. No overlapping "
            "segments and no gaps are allowed; long blocks may overlap, so adjudicate a shared boundary using the "
            "strongest local evidence and confidence. Return exactly "
            '{"boundaries":[{"after_segment_index":int,"boundary_type":str,"next_topic":str|null,'
            '"next_subject":str|null,"confidence":number|null}]} and no prose.\n'
            + json.dumps(lines, ensure_ascii=False, separators=(",", ":"))
        )

    def _parse(self, response: Any, block_start: int, block_end: int) -> list[SemanticBoundary]:
        content = response.get("content", response) if isinstance(response, dict) else response
        if isinstance(content, str):
            content = json.loads(content)
        if (
            not isinstance(content, dict)
            or set(content) != {"boundaries"}
            or not isinstance(content["boundaries"], list)
        ):
            raise ValueError("boundary response must contain only boundaries")
        parsed = [SemanticBoundary(**item) for item in content["boundaries"]]
        for item in parsed:
            self._validate_metadata(item)
        # A block may not introduce a boundary outside its covered coordinates.
        validate_boundaries(parsed, block_end)
        if any(item.after_segment_index < block_start or item.after_segment_index >= block_end - 1 for item in parsed):
            raise ValueError("boundary outside requested block")
        return parsed


SemanticSegmentationStage = SemanticSegmenter

__all__ = ["SemanticSegmenter", "SemanticSegmentationStage", "SemanticSegmentationResult"]

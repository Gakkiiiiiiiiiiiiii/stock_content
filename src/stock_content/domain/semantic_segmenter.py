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
        require_model_identity: bool = False,
        require_initial_topic: bool = False,
        refine_segments: bool = False,
        refinement_min_tokens: int = 1,
        verify_brief_topic_labels: bool = False,
    ) -> None:
        if safe_tokens <= 0 or (block_tokens is not None and block_tokens <= 0):
            raise ValueError("token budgets must be positive")
        if segment_overlap < 0:
            raise ValueError("segment_overlap must be non-negative")
        if refinement_min_tokens <= 0:
            raise ValueError("refinement_min_tokens must be positive")
        self.model_gateway = model_gateway
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.safe_tokens = safe_tokens
        self.block_tokens = block_tokens or safe_tokens
        self.segment_overlap = segment_overlap
        self.allow_offline_fixture = allow_offline_fixture
        self.require_model_identity = require_model_identity
        self.require_initial_topic = require_initial_topic
        self.refine_segments = refine_segments
        self.refinement_min_tokens = refinement_min_tokens
        self.verify_brief_topic_labels = verify_brief_topic_labels
        self.last_metrics: dict[str, float] = {}

    def segment(
        self, transcript: TranscriptArtifact, *, offline_fixture: bool = False, identity_seed: str = ""
    ) -> SemanticSegmentationResult:
        items = list(transcript.segments)
        if not items:
            artifact = build_semantic_segment_artifact(
                transcript, (), model_id=self.model_id, prompt_version=self.prompt_version,
                schema_version=self.schema_version, identity_seed=identity_seed,
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
        if offline_fixture or not available:
            boundaries: list[SemanticBoundary] = []
            repair_count = 0
            refinement_call_count = 0
            refinement_added_count = 0
            initial_topic = initial_subject = None
        elif self._token_count(transcript) + self._prompt_overhead() <= self.safe_tokens:
            boundaries, initial_topic, initial_subject, repair_count = self._call_with_repair(items, 0, len(items))
            refinement_call_count = 0
            refinement_added_count = 0
        else:
            proposals: list[SemanticBoundary] = []
            repair_count = 0
            initial_topic = initial_subject = None
            for start, end in self._blocks(items):
                block, block_topic, block_subject, repairs = self._call_with_repair(items, start, end)
                proposals.extend(block)
                if start == 0:
                    initial_topic, initial_subject = block_topic, block_subject
                repair_count += repairs
            # Reconciliation is deterministic: coordinates are deduplicated,
            # sorted and validated globally; no proposal is clamped.  A model
            # can place one boundary at either side of an overlap, so nearby
            # conflicting coordinates are adjudicated as one candidate.
            # The block overlap supplies context, not a minimum topic length.
            # Reconciling across the entire context would erase genuinely
            # adjacent subject changes in a dense discussion.
            boundaries = self._reconcile(proposals, overlap=min(2, self.segment_overlap))
            refinement_call_count = 0
            refinement_added_count = 0

        if self.refine_segments and not offline_fixture and available:
            coarse_segments = materialize_semantic_segments(
                transcript, boundaries, initial_topic=initial_topic, initial_subject=initial_subject
            )
            refinements: list[SemanticBoundary] = []
            refined_labels: dict[int, tuple[str, str | None]] = {}
            adjudication_call_count = 0
            adjudication_rejected_count = 0
            for coarse in coarse_segments:
                start, end = coarse.start_segment_index, coarse.end_segment_index + 1
                span_cost = sum(self._item_token_cost(item) for item in items[start:end])
                if end - start < 2 or span_cost < self.refinement_min_tokens:
                    continue
                local_proposals: list[SemanticBoundary] = []
                if span_cost + self._prompt_overhead(refinement=True) <= self.block_tokens:
                    windows = [(start, end)]
                else:
                    windows = [(start + a, start + b) for a, b in self._blocks(items[start:end], refinement=True)]
                for window_start, window_end in windows:
                    proposed, topic, subject, repairs = self._call_with_repair(
                        items, window_start, window_end, refinement=True
                    )
                    repair_count += repairs
                    if proposed:
                        accepted, repairs = self._adjudicate_internal_boundaries(
                            items, window_start, window_end, proposed,
                            parent_topic=coarse.topic, parent_subject=coarse.subject,
                        )
                        adjudication_call_count += 1
                        adjudication_rejected_count += len(proposed) - len(accepted)
                        repair_count += repairs
                    else:
                        accepted = proposed
                    if window_start == start and topic and (not proposed or accepted):
                        refined_labels[start] = (topic, subject)
                    local_proposals.extend(accepted)
                    refinement_call_count += 1
                refinements.extend(
                    local_proposals if len(windows) == 1
                    else self._reconcile(local_proposals, overlap=min(2, self.segment_overlap))
                )
            refinement_added_count = len(refinements)
            boundaries = sorted([*boundaries, *refinements], key=lambda item: item.after_segment_index)
            if 0 in refined_labels:
                initial_topic, initial_subject = refined_labels[0]
            boundaries = [
                replace(
                    boundary,
                    next_topic=refined_labels[boundary.after_segment_index + 1][0],
                    next_subject=refined_labels[boundary.after_segment_index + 1][1],
                )
                if boundary.after_segment_index + 1 in refined_labels else boundary
                for boundary in boundaries
            ]
            topic_audit_call_count = 0
            topic_audit_added_count = 0
            audited_segments = materialize_semantic_segments(
                transcript, boundaries, initial_topic=initial_topic, initial_subject=initial_subject
            )
            audited_boundaries: list[SemanticBoundary] = []
            audit_labels: dict[int, tuple[str, str | None]] = {}
            for segment in audited_segments:
                start, end = segment.start_segment_index, segment.end_segment_index + 1
                span_cost = sum(self._item_token_cost(item) for item in items[start:end])
                if end - start < 4 or (
                    span_cost < 256 and not self._looks_composite_topic(segment.topic, segment.subject)
                ):
                    continue
                # A coverage audit needs a narrower view than the model's
                # maximum context: many short ASR rows can hide a late subject
                # shift even when the entire chapter technically fits.
                audit_overhead = self._prompt_overhead(refinement=True, topic_audit=True)
                largest_row = max(self._item_token_cost(item) for item in items[start:end])
                audit_tokens = min(self.block_tokens, max(1400, audit_overhead + largest_row))
                if span_cost + audit_overhead <= audit_tokens:
                    windows = [(start, end)]
                else:
                    windows = [
                        (start + a, start + b)
                        for a, b in self._blocks(
                            items[start:end], refinement=True, topic_audit=True,
                            limit_tokens=audit_tokens,
                        )
                    ]
                local_proposals: list[SemanticBoundary] = []
                for window_start, window_end in windows:
                    proposed, topic, subject, repairs = self._call_with_repair(
                        items, window_start, window_end, refinement=True, topic_audit=True
                    )
                    repair_count += repairs
                    if proposed:
                        accepted, repairs = self._adjudicate_internal_boundaries(
                            items, window_start, window_end, proposed,
                            parent_topic=segment.topic, parent_subject=segment.subject,
                        )
                        adjudication_call_count += 1
                        adjudication_rejected_count += len(proposed) - len(accepted)
                        repair_count += repairs
                    else:
                        accepted = proposed
                    if window_start == start and topic and (not proposed or accepted):
                        audit_labels[start] = (topic, subject)
                    local_proposals.extend(accepted)
                    topic_audit_call_count += 1
                audited_boundaries.extend(
                    local_proposals if len(windows) == 1
                    else self._reconcile(local_proposals, overlap=min(2, self.segment_overlap))
                )
            topic_audit_added_count = len(audited_boundaries)
            boundaries = sorted([*boundaries, *audited_boundaries], key=lambda item: item.after_segment_index)
            if 0 in audit_labels:
                initial_topic, initial_subject = audit_labels[0]
            boundaries = [
                replace(
                    boundary,
                    next_topic=audit_labels[boundary.after_segment_index + 1][0],
                    next_subject=audit_labels[boundary.after_segment_index + 1][1],
                )
                if boundary.after_segment_index + 1 in audit_labels else boundary
                for boundary in boundaries
            ]
        else:
            topic_audit_call_count = 0
            topic_audit_added_count = 0
            adjudication_call_count = 0
            adjudication_rejected_count = 0

        brief_label_call_count = 0
        brief_label_corrected_count = 0
        if self.verify_brief_topic_labels and not offline_fixture and available:
            provisional = materialize_semantic_segments(
                transcript, boundaries, initial_topic=initial_topic, initial_subject=initial_subject
            )
            for segment in provisional:
                start, end = segment.start_segment_index, segment.end_segment_index + 1
                if end - start > 6:
                    continue
                topic, subject, repairs = self._verify_brief_topic_label(
                    items[start:end], segment.topic, segment.subject
                )
                brief_label_call_count += 1
                repair_count += repairs
                if (topic, subject) == (segment.topic, segment.subject):
                    continue
                brief_label_corrected_count += 1
                if start == 0:
                    initial_topic, initial_subject = topic, subject
                else:
                    boundaries = [
                        replace(boundary, next_topic=topic, next_subject=subject)
                        if boundary.after_segment_index == start - 1 else boundary
                        for boundary in boundaries
                    ]

        segments = materialize_semantic_segments(
            transcript,
            boundaries,
            initial_topic=initial_topic,
            initial_subject=initial_subject,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            identity_seed=identity_seed,
        )
        validate_full_coverage(segments, len(items))
        artifact = build_semantic_segment_artifact(
            transcript,
            boundaries,
            initial_topic=initial_topic,
            initial_subject=initial_subject,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            identity_seed=identity_seed,
        )
        # Artifact parent linkage is part of the authoritative chain.
        artifact = replace(artifact, parent_artifact_ids=(transcript.artifact_id,), artifact_id="", content_hash="")
        artifact = replace(artifact, artifact_id=artifact_id_of(artifact))
        self.last_metrics = {
            "segment_count": float(len(segments)),
            "repair_count": float(repair_count),
            "failure_count": 0.0,
            "boundary_count": float(len(boundaries)),
            "refinement_call_count": float(refinement_call_count),
            "refinement_added_boundary_count": float(refinement_added_count),
            "topic_audit_call_count": float(topic_audit_call_count),
            "topic_audit_added_boundary_count": float(topic_audit_added_count),
            "adjudication_call_count": float(adjudication_call_count),
            "adjudication_rejected_boundary_count": float(adjudication_rejected_count),
            "brief_label_call_count": float(brief_label_call_count),
            "brief_label_corrected_count": float(brief_label_corrected_count),
        }
        return SemanticSegmentationResult(artifact, tuple(segments), dict(self.last_metrics))

    materialize = segment

    def _blocks(
        self, items: list[Any], *, refinement: bool = False, topic_audit: bool = False,
        limit_tokens: int | None = None,
    ) -> list[tuple[int, int]]:
        row_budget = (limit_tokens or self.block_tokens) - self._prompt_overhead(
            refinement=refinement, topic_audit=topic_audit
        )
        if row_budget <= 0:
            raise ValueError("block token budget cannot hold the segmentation instructions")
        blocks: list[tuple[int, int]] = []
        start = 0
        while start < len(items):
            budget = 0
            end = start
            while end < len(items):
                item = items[end]
                cost = self._item_token_cost(item)
                if cost > row_budget:
                    raise ValueError("single transcript segment exceeds the model block token budget")
                if end > start and budget + cost > row_budget:
                    break
                budget += cost
                end += 1
            end = max(start + 1, end)
            end = min(len(items), end)
            blocks.append((start, end))
            if end == len(items):
                break
            # A transcript with unusually long ASR rows must still advance by
            # meaningful new content instead of making near-duplicate calls.
            # A two-row block cannot afford overlap without doubling calls.
            overlap = min(self.segment_overlap, (end - start) // 3)
            start = end - overlap
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
        # Count the serialized rows actually sent to the model.  Counting only
        # transcript text misses JSON/index overhead and greatly underestimates
        # Chinese ASR, causing long videos to be sent as one oversized prompt.
        return sum(SemanticSegmenter._item_token_cost(item) for item in transcript.segments)

    @staticmethod
    def _item_token_cost(item: Any) -> int:
        row = json.dumps(
            {"segment_index": item.segment_index, "text": item.raw_text or item.text or ""},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return SemanticSegmenter._text_token_cost(row)

    @staticmethod
    def _text_token_cost(text: str) -> int:
        ascii_chars = sum(ord(char) < 128 for char in text)
        return max(1, (ascii_chars + 3) // 4 + len(text) - ascii_chars)

    def _prompt_overhead(self, *, refinement: bool = False, topic_audit: bool = False) -> int:
        # Reserve the instructions and a small allowance for larger coordinate
        # numbers.  This is independent of the source video and its duration.
        instruction = self._refinement_instruction() if refinement else ""
        if topic_audit:
            instruction += self._topic_audit_instruction()
        return self._text_token_cost(self._prompt([], 0, 1) + instruction) + 32

    @staticmethod
    def _refinement_instruction() -> str:
        return (
            "\nThis is a second look inside one coarse topic. Split only independent subject or thesis changes. "
            "A brief but substantive view about a named asset, sector or company is a real topic change. "
            "Keep each thesis with its evidence, risks, conclusion and closing recap. "
            "A brief aside or conversational interjection is not an independent topic. "
            "Do not split a summary or a related detail merely because it is shorter or later."
        )

    @staticmethod
    def _looks_composite_topic(topic: str | None, subject: str | None) -> bool:
        label = " ".join(value for value in (topic, subject) if value)
        return any(marker in label for marker in ("、", "以及", "及", "与", "和", " and "))

    @staticmethod
    def _topic_audit_instruction() -> str:
        return (
            "\nAudit topic coverage independently of the previous labels. Find every supported subject-specific "
            "view or recommendation about an asset, sector or company, even if brief or omitted from the label. "
            "Check each item in a recap list. Split independent views, not one causal thesis. "
            "Never turn conversational filler, a greeting or a host-arrival aside into a topic."
        )

    def _adjudicate_internal_boundaries(
        self, items: list[Any], start: int, end: int, proposals: list[SemanticBoundary],
        *, parent_topic: str, parent_subject: str | None,
    ) -> tuple[list[SemanticBoundary], int]:
        """Confirm that a second-pass split is an independent teachable theme."""
        candidates = [
            {
                "after_segment_index": item.after_segment_index,
                "next_topic": item.next_topic,
                "next_subject": item.next_subject,
            }
            for item in proposals
        ]
        rows = [
            {"segment_index": item.segment_index, "text": item.raw_text or item.text}
            for item in items[start:end]
        ]
        prompt = (
            "Adjudicate these proposed INTERNAL topic splits in the full context below. "
            "You are a CHAPTER editor, not an atomic-claim splitter. Reject a split that merely separates "
            "evidence, a causal mechanism, risk, condition or downstream implication from its parent thesis. "
            "A component supply constraint followed by a short supplier-profit qualification is normally "
            "one causal discussion; different entity labels alone are not enough. Keep a split for a real "
            "transition to another named asset or sector, even if its view is brief. Within one broad field, "
            "also keep a split when the central question changes and each side develops its OWN explanation "
            "or evidence rather than merely qualifying the prior question. An extended discussion of a new "
            "workflow, market structure, technology, data constraint or business model may be its own chapter "
            "even within the same industry. A new named policy, negotiation or market event with its OWN "
            "outcome and market implication is independent even when introduced as an alternative or with "
            "'because/therefore' after another geopolitical story. A shared backdrop or causal connection "
            "does not erase a separately developed conclusion; conversely a brief risk qualification remains "
            "with its thesis. The parent label may itself be incomplete, so do not use it as a veto. "
            "A brief substantive view in a recap can also be independent; a conversational aside is not. "
            "Use the words on both sides; never infer from duration or position. Return exactly "
            '{"accepted_after_segment_indices":[int,...]} with a subset of the proposed coordinates. '
            "Do not add coordinates, explanations or any other keys.\n"
            + json.dumps(
                {"parent_topic": parent_topic, "parent_subject": parent_subject,
                 "proposals": candidates, "transcript": rows},
                ensure_ascii=False, separators=(",", ":"),
            )
        )
        allowed = {item.after_segment_index for item in proposals}
        for attempt in range(2):
            response = self._complete(prompt)
            try:
                content = response.get("content", response) if isinstance(response, dict) else response
                if isinstance(content, str):
                    content = json.loads(content)
                if not isinstance(content, dict) or set(content) != {"accepted_after_segment_indices"}:
                    raise ValueError("boundary adjudication response has invalid shape")
                accepted = content["accepted_after_segment_indices"]
                if (
                    not isinstance(accepted, list)
                    or any(isinstance(index, bool) or not isinstance(index, int) for index in accepted)
                    or len(accepted) != len(set(accepted))
                    or not set(accepted) <= allowed
                ):
                    raise ValueError("boundary adjudication response must be a unique proposed-coordinate subset")
                return [item for item in proposals if item.after_segment_index in accepted], attempt
            except (TypeError, ValueError, json.JSONDecodeError):
                if attempt:
                    raise ValueError("boundary adjudication protocol failed after one repair") from None
                prompt += "\nPrevious output was invalid. Repair JSON schema only; select only proposed integers."
        raise AssertionError("unreachable")

    def _verify_brief_topic_label(
        self, items: list[Any], topic: str | None, subject: str | None
    ) -> tuple[str, str | None, int]:
        """Ground a brief chapter's label in exact source wording, without guessing entities."""
        rows = [item.raw_text or item.text or "" for item in items]
        prompt = (
            "Independently verify this brief chapter label against ONLY its transcript rows. "
            "Correct a mislabeled asset, company, instrument or sector; never substitute a nearby market "
            "term for the term actually spoken. Do not infer a full company name or ticker from an unclear "
            "abbreviation or ASR homophone. Search ALL rows for a concrete named asset, instrument, company "
            "or sector; if one is spoken, subject MUST copy that exact term and topic MUST include it. "
            "Do not evade an incorrect candidate by dropping a clearly spoken subject to null. "
            "Retain the chapter's objective thesis, not a third-person "
            "narration. Return exactly one JSON object with topic (non-empty string), subject (an exact "
            "substring copied from one source_quote, or null when no concrete subject is spoken), and "
            "source_quote (a non-empty exact substring copied from ONE transcript row). If subject is "
            "non-null, include its exact source spelling in topic. Do not add keys or prose.\n"
            + json.dumps(
                {"candidate_topic": topic, "candidate_subject": subject, "transcript_rows": rows},
                ensure_ascii=False, separators=(",", ":"),
            )
        )
        for attempt in range(2):
            response = self._complete(prompt)
            try:
                content = response.get("content", response) if isinstance(response, dict) else response
                if isinstance(content, str):
                    content = json.loads(content)
                if not isinstance(content, dict) or set(content) != {"topic", "subject", "source_quote"}:
                    raise ValueError("brief label response has invalid shape")
                revised_topic, revised_subject, quote = (
                    content["topic"], content["subject"], content["source_quote"]
                )
                if (
                    not isinstance(revised_topic, str) or not revised_topic.strip()
                    or not isinstance(quote, str) or not quote.strip()
                    or not any(quote in row for row in rows)
                    or subject is not None and revised_subject is None
                    or revised_subject is not None and (
                        not isinstance(revised_subject, str) or not revised_subject.strip()
                        or revised_subject not in quote or revised_subject not in revised_topic
                    )
                ):
                    raise ValueError("brief label is not grounded in its exact source quote")
                return revised_topic, revised_subject, attempt
            except (TypeError, ValueError, json.JSONDecodeError):
                if attempt:
                    raise ValueError("brief label grounding failed after one repair") from None
                prompt += (
                    "\nPrevious output failed exact-quote validation. Repair using only literal row text. "
                    "If the candidate had a subject, null is invalid: identify the actual spoken term in "
                    "the rows and copy it exactly, or the whole parse must fail closed."
                )
        raise AssertionError("unreachable")

    def _call_with_repair(
        self, items: list[Any], start: int, end: int, *, refinement: bool = False,
        topic_audit: bool = False,
    ) -> tuple[list[SemanticBoundary], str | None, str | None, int]:
        prompt = self._prompt(items, start, end)
        if refinement:
            prompt += self._refinement_instruction()
        if topic_audit:
            prompt += self._topic_audit_instruction()
        response = self._complete(prompt)
        try:
            boundaries, topic, subject = self._parse(response, start, end)
            return boundaries, topic, subject, 0
        except (TypeError, ValueError, json.JSONDecodeError):
            repair = self._complete(
                prompt
                + "\n上一次输出无效。仅修复 JSON schema，仍不得输出 claim 或 timestamp。"
                + " confidence 必须为 null 或 [0,1]（含端点）内的有限 JSON number；不得使用百分制（例如 100）。"
                + "\n"
                + self._boundary_coordinate_instruction(start, end)
            )
            try:
                boundaries, topic, subject = self._parse(repair, start, end)
                return boundaries, topic, subject, 1
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("semantic segmentation boundary protocol failed after one repair") from exc

    def _complete(self, prompt: str) -> Any:
        gateway = self.model_gateway
        try:
            response = gateway.complete(
                prompt=prompt,
                system="You are a semantic boundary detector. Return JSON only.",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except TypeError:
            response = gateway.complete(
                messages=[
                    {"role": "system", "content": "You are a semantic boundary detector. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        if self.require_model_identity and (
            not isinstance(response, dict) or response.get("model") != self.model_id
        ):
            raise RuntimeError("semantic segmentation runtime model identity mismatch")
        return response

    def _prompt(self, items: list[Any], start: int, end: int) -> str:
        lines = [
            {"segment_index": item.segment_index, "text": item.raw_text or item.text}
            for item in items[start:end]
        ]
        return (
            "You are Stage 1 semantic segmentation. Return topic labels and boundaries only; never claims, "
            "values, dates, "
            "timestamps, evidence, summaries, or copied transcript text. Produce full contiguous coverage: "
            "a new segment starts only at a substantive topic, subject or independent conclusion change. "
            "Keep one thesis, its evidence, risks, conditions and closing recap together. "
            "Attach brief asides, greetings, self-corrections and filler to adjacent substantive discussion; "
            "a brief subject-specific view about a named asset, sector or company is substantive, not filler. "
            "separate independent advertisements, disclaimers and unrelated substantive Q&A. "
            "Do not split at a fixed elapsed time, block edge, or target segment count; the subject change must "
            "be supported by the words on both sides of the boundary. "
            "Name the first topic and every next topic with the concrete subject and thesis from nearby words; "
            "avoid generic labels such as investment direction, market analysis, or next topic. "
            "For each boundary give the next topic and subject when known. Do not invent entities. No overlapping "
            "segments and no gaps are allowed; long blocks may overlap, so adjudicate a shared boundary using the "
            "strongest local evidence and confidence. Return exactly "
            '{"initial_topic":str,"initial_subject":str|null,"boundaries":'
            '[{"after_segment_index":int,"boundary_type":str,"next_topic":str|null,'
            '"next_subject":str|null,"confidence":number|null}]} and no prose. '
            "confidence 必须为 null 或 [0,1]（含端点）内的有限 JSON number；不得使用百分制（例如 100）。\n"
            + self._boundary_coordinate_instruction(start, end)
            + "\n"
            + json.dumps(lines, ensure_ascii=False, separators=(",", ":"))
        )

    @staticmethod
    def _boundary_coordinate_instruction(block_start: int, block_end: int) -> str:
        """Describe the block-local boundary coordinates without weakening validation."""
        final_provided_segment = block_end - 1
        last_legal_boundary = block_end - 2
        if last_legal_boundary < block_start:
            return (
                "This block has no legal after_segment_index: return an empty boundaries list. "
                f"Never emit {final_provided_segment}, the final provided segment."
            )
        return (
            "The only legal after_segment_index values for this block are integers in the inclusive range "
            f"[{block_start}, {last_legal_boundary}]. Never emit {final_provided_segment}: it is the final "
            "provided segment and cannot be a boundary."
        )

    def _parse(
        self, response: Any, block_start: int, block_end: int
    ) -> tuple[list[SemanticBoundary], str | None, str | None]:
        content = response.get("content", response) if isinstance(response, dict) else response
        if isinstance(content, str):
            content = json.loads(content)
        if (
            not isinstance(content, dict)
            or set(content) not in (
                {"boundaries"},
                {"initial_topic", "initial_subject", "boundaries"},
            )
            or not isinstance(content["boundaries"], list)
        ):
            raise ValueError("boundary response must contain only topic labels and boundaries")
        initial_topic = content.get("initial_topic")
        initial_subject = content.get("initial_subject")
        if self.require_initial_topic and (not isinstance(initial_topic, str) or not initial_topic.strip()):
            raise ValueError("semantic segmentation requires a concrete initial topic")
        if initial_topic is not None and (not isinstance(initial_topic, str) or not initial_topic.strip()):
            raise ValueError("initial_topic must be a non-empty string or null")
        if initial_subject is not None and (not isinstance(initial_subject, str) or not initial_subject.strip()):
            raise ValueError("initial_subject must be a non-empty string or null")
        parsed = [SemanticBoundary(**item) for item in content["boundaries"]]
        for item in parsed:
            self._validate_metadata(item)
        # A block may not introduce a boundary outside its covered coordinates.
        validate_boundaries(parsed, block_end)
        if any(item.after_segment_index < block_start or item.after_segment_index >= block_end - 1 for item in parsed):
            raise ValueError("boundary outside requested block")
        return parsed, initial_topic, initial_subject


SemanticSegmentationStage = SemanticSegmenter

__all__ = ["SemanticSegmenter", "SemanticSegmentationStage", "SemanticSegmentationResult"]

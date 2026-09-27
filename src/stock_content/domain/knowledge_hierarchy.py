"""Chapter-thesis planning and additive parent/child knowledge metadata.

The chapter thesis is deliberately identified before atomic claims.  It is a
navigation/interpretation layer over transcript evidence, not a replacement
for the validated atomic claim authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from .artifacts import TranscriptArtifact, canonical_json
from .claim_draft import ClaimOccurrenceDraft


class KnowledgeRole(StrEnum):
    THESIS = "THESIS"
    MECHANISM = "MECHANISM"
    EVIDENCE = "EVIDENCE"
    IMPLICATION = "IMPLICATION"


KNOWLEDGE_ROLES = frozenset(item.value for item in KnowledgeRole)


@dataclass(frozen=True, slots=True)
class ChapterThesis:
    thesis_id: str
    knowledge_title: str
    atomic_statement: str
    subject_type: str
    subject_key: str
    subject_name: str
    predicate: str
    claim_type: str
    sentiment: str
    attribution: str
    detailed_explanation: str
    proposal_segment_indices: tuple[int, ...]
    argument_segment_indices: tuple[int, ...]
    conclusion_segment_indices: tuple[int, ...]
    evidence_segment_indices: tuple[int, ...]
    semantic_segment_ids: tuple[str, ...]
    knowledge_role: str = KnowledgeRole.THESIS.value
    model_id: str = ""

    @property
    def start_segment_index(self) -> int:
        return self.evidence_segment_indices[0]

    @property
    def end_segment_index(self) -> int:
        return self.evidence_segment_indices[-1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "start_segment_index": self.start_segment_index,
            "end_segment_index": self.end_segment_index,
        }


class ChapterThesisIdentifier:
    """Identify a chapter's upper thesis before atomic extraction."""

    # Deterministic five-minute chapters can cut a coherent topic at the
    # boundary.  Thesis identification reads a bounded context overlap while
    # the returned evidence coordinates remain exact transcript indices.
    CHAPTER_CONTEXT_PADDING_MS = 120_000

    def __init__(
        self,
        model_gateway: Any | None = None,
        *,
        model_id: str = "",
        prompt_version: str = "chapter-thesis.v1",
    ) -> None:
        self.model_gateway = model_gateway
        self.model_id = model_id
        self.prompt_version = prompt_version

    def identify(
        self,
        transcript: TranscriptArtifact,
        semantic_segments: Iterable[Any],
        chapters: Iterable[Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        fixture_theses: Iterable[Mapping[str, Any]] | None = None,
        offline_fixture: bool = False,
    ) -> list[ChapterThesis]:
        semantic_items = list(semantic_segments)
        if fixture_theses is not None:
            if not offline_fixture:
                raise ValueError("chapter thesis fixtures are allowed only in offline tests")
            return [self._parse(item, transcript, semantic_items) for item in fixture_theses]
        if self.model_gateway is None or not bool(getattr(self.model_gateway, "available", lambda: True)()):
            if offline_fixture:
                return []
            raise RuntimeError("chapter thesis identification model gateway is unavailable")

        output: list[ChapterThesis] = []
        seen: set[str] = set()
        for chapter in chapters:
            selected = self._chapter_segments(chapter, transcript)
            if not selected:
                continue
            response = self._complete(self._prompt(selected, metadata or {}))
            content = response.get("content", response) if isinstance(response, Mapping) else response
            returned_model = str(response.get("model") or "") if isinstance(response, Mapping) else ""
            if self.model_id == "gpt-6-sol" and returned_model != self.model_id:
                raise ValueError("chapter thesis model identity does not match gpt-6-sol")
            payload = json.loads(content) if isinstance(content, str) else content
            if not isinstance(payload, Mapping) or set(payload) != {"theses"}:
                raise ValueError("chapter thesis response must contain only theses")
            for item in payload["theses"]:
                thesis = self._parse(
                    item, transcript, semantic_items, allowed=selected,
                    model_id=returned_model or self.model_id,
                )
                if thesis.thesis_id not in seen:
                    output.append(thesis)
                    seen.add(thesis.thesis_id)
        return output

    def _complete(self, prompt: str) -> Any:
        try:
            return self.model_gateway.complete(
                prompt=prompt,
                system="You identify chapter-level investment theses. Return JSON only.",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except TypeError:
            return self.model_gateway.complete(
                messages=[
                    {"role": "system", "content": "You identify chapter-level investment theses. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

    def _prompt(self, segments: list[Any], metadata: Mapping[str, Any]) -> str:
        source = [
            {
                "segment_index": item.segment_index,
                "text": item.text or item.normalized_text or item.raw_text,
            }
            for item in segments
        ]
        return (
            "在原子命题抽取之前，先识别本章的上位投资论点。一个论点必须完整覆盖观点提出、论证、"
            "收束三个修辞阶段；相邻且同主题的机制、预测和影响不得拆成多个顶层论点。"
            "knowledge_title 必须概括整段核心主题，不能用局部机制或内部 subject 充当标题。"
            "atomic_statement 必须是输入转录中连续、逐字可定位的原文；相邻 segment 的原文用一个空格连接，"
            "不能把总结标题改写成 atomic_statement。subject_key、subject_name 和 predicate 也必须逐字出现在"
            "atomic_statement 所覆盖的原文中。"
            "detailed_explanation 必须直接从内容结论开始，禁止以讲述者、讲者、视频、节目、课程、"
            "口播或转录等来源套话开头；来源只写 attribution。不得外推原文没有的数字或事实。"
            "每个 thesis 返回 knowledge_title, atomic_statement, subject_type, subject_key, subject_name, "
            "predicate, claim_type, sentiment, attribution, detailed_explanation, "
            "proposal_segment_indices, argument_segment_indices, conclusion_segment_indices。"
            "三个 indices 数组均不可为空，且只能使用输入 segment_index。只返回 {\"theses\":[...]}。\n"
            + json.dumps(
                {"metadata": dict(metadata), "transcript_segments": source},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    @staticmethod
    def _chapter_segments(chapter: Any, transcript: TranscriptArtifact) -> list[Any]:
        start_ms = max(
            0,
            int(float(getattr(chapter, "start_seconds", 0.0)) * 1000)
            - ChapterThesisIdentifier.CHAPTER_CONTEXT_PADDING_MS,
        )
        end_ms = (
            int(float(getattr(chapter, "end_seconds", 0.0)) * 1000)
            + ChapterThesisIdentifier.CHAPTER_CONTEXT_PADDING_MS
        )
        return [item for item in transcript.segments if item.end_ms >= start_ms and item.start_ms <= end_ms]

    @staticmethod
    def _parse(
        raw: Mapping[str, Any],
        transcript: TranscriptArtifact,
        semantic_segments: list[Any],
        *,
        allowed: list[Any] | None = None,
        model_id: str = "",
    ) -> ChapterThesis:
        if not isinstance(raw, Mapping):
            raise ValueError("chapter thesis must be an object")
        allowed_indices = {item.segment_index for item in (allowed or transcript.segments)}

        def indices(name: str) -> tuple[int, ...]:
            values = tuple(sorted({int(value) for value in raw.get(name) or ()}))
            if not values or not set(values).issubset(allowed_indices):
                raise ValueError(f"invalid {name}")
            return values

        proposal = indices("proposal_segment_indices")
        argument = indices("argument_segment_indices")
        conclusion = indices("conclusion_segment_indices")
        if min(argument) < min(proposal) or min(conclusion) < min(argument):
            raise ValueError("thesis rhetorical spans must be ordered")
        start, end = min(proposal), max(conclusion)
        evidence = tuple(index for index in sorted(allowed_indices) if start <= index <= end)
        if not evidence:
            raise ValueError("thesis evidence window is empty")

        required = {
            key: str(raw.get(key) or "").strip()
            for key in (
                "knowledge_title", "atomic_statement", "subject_type", "subject_key",
                "subject_name", "predicate", "claim_type", "sentiment", "attribution",
                "detailed_explanation",
            )
        }
        if any(not value for value in required.values()):
            raise ValueError("chapter thesis fields are incomplete")
        evidence_text = " ".join(
            str(item.raw_text or item.text or item.normalized_text or "")
            for item in transcript.segments
            if item.segment_index in evidence
        )
        if required["atomic_statement"].casefold() not in evidence_text.casefold():
            raise ValueError("chapter thesis atomic_statement must be verbatim transcript text")
        for field in ("subject_key", "subject_name", "predicate"):
            if required[field].casefold() not in evidence_text.casefold():
                raise ValueError(f"chapter thesis {field} must be transcript-local")
        # Imported lazily to keep the hierarchy module independent from the
        # semantic envelope's initialization order.
        from .knowledge_semantics import normalize_detail_text

        detail = normalize_detail_text(required["detailed_explanation"])
        semantic_ids = tuple(
            item.semantic_segment_id
            for item in semantic_segments
            if item.end_segment_index >= start and item.start_segment_index <= end
        )
        identity = {
            "statement": required["atomic_statement"],
            "start_segment_index": start,
            "end_segment_index": end,
            "semantic_segment_ids": semantic_ids,
        }
        thesis_id = "thesis_" + hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:56]
        return ChapterThesis(
            thesis_id=thesis_id,
            knowledge_title=required["knowledge_title"],
            atomic_statement=required["atomic_statement"],
            subject_type=required["subject_type"],
            subject_key=required["subject_key"],
            subject_name=required["subject_name"],
            predicate=required["predicate"],
            claim_type=required["claim_type"],
            sentiment=required["sentiment"],
            attribution=required["attribution"],
            detailed_explanation=detail,
            proposal_segment_indices=proposal,
            argument_segment_indices=argument,
            conclusion_segment_indices=conclusion,
            evidence_segment_indices=evidence,
            semantic_segment_ids=semantic_ids,
            model_id=model_id,
        )


def role_for_claim(draft: ClaimOccurrenceDraft) -> KnowledgeRole:
    explicit = str((draft.bundle_v2 or {}).get("knowledge_role") or "").upper()
    if explicit in KNOWLEDGE_ROLES - {KnowledgeRole.THESIS.value}:
        return KnowledgeRole(explicit)
    statement = str(draft.normalized_statement or draft.conclusion or "")
    nature = str((draft.bundle_v2 or {}).get("claim_nature") or draft.claim_type or "").upper()
    if (
        any(token in nature for token in ("CAUSAL", "MECHANISM", "WORKFLOW"))
        or any(token in statement for token in ("因为", "因此", "所以", "依赖", "关键条件", "卡住", "瓶颈", "约束"))
    ):
        return KnowledgeRole.MECHANISM
    if "FORECAST" in nature:
        if re.search(r"\d|价格|涨幅|预测值|图表", statement):
            return KnowledgeRole.EVIDENCE
        return KnowledgeRole.IMPLICATION
    if any(token in statement for token in ("意味着", "有望", "可能", "受益", "反弹", "业绩")):
        return KnowledgeRole.IMPLICATION
    return KnowledgeRole.EVIDENCE


def attach_thesis_hierarchy(
    drafts: Iterable[ClaimOccurrenceDraft], theses: Iterable[ChapterThesis]
) -> list[ClaimOccurrenceDraft]:
    """Attach additive role/parent metadata without changing atomic facts."""

    thesis_list = list(theses)
    output: list[ClaimOccurrenceDraft] = []
    for draft in drafts:
        evidence = set(int(value) for value in draft.evidence_segment_indices)
        matches = []
        if evidence:
            role = role_for_claim(draft)
            for thesis in thesis_list:
                contained = (
                    min(evidence) >= thesis.start_segment_index
                    and max(evidence) <= thesis.end_segment_index
                )
                # A prediction or implication often immediately follows the
                # thesis' closing sentence.  Keep the parent's rhetorical
                # evidence scope immutable, but allow a tightly adjacent
                # atomic child to point back to it.  The eight-segment guard
                # prevents a later topic from being absorbed by position
                # alone; mechanisms must still be inside the parent window.
                adjacent_after = (
                    role in {KnowledgeRole.EVIDENCE, KnowledgeRole.IMPLICATION}
                    and 0 < min(evidence) - thesis.end_segment_index <= 8
                )
                if contained or adjacent_after:
                    matches.append(thesis)
        if not matches:
            output.append(draft)
            continue
        parent = min(matches, key=lambda item: item.end_segment_index - item.start_segment_index)
        bundle = {
            **dict(draft.bundle_v2 or {}),
            "knowledge_role": role_for_claim(draft).value,
            "parent_knowledge_id": parent.thesis_id,
            "parent_knowledge_title": parent.knowledge_title,
            "thesis_evidence_scope": {
                "start_segment_index": parent.start_segment_index,
                "end_segment_index": parent.end_segment_index,
                "proposal_segment_indices": list(parent.proposal_segment_indices),
                "argument_segment_indices": list(parent.argument_segment_indices),
                "conclusion_segment_indices": list(parent.conclusion_segment_indices),
            },
        }
        output.append(draft.model_copy(update={"bundle_v2": bundle}))
    return output


def materialize_thesis_claim_drafts(theses: Iterable[ChapterThesis]) -> list[ClaimOccurrenceDraft]:
    """Materialize chapter theses as first-class, validator-bound claims.

    A hierarchy node that exists only as child metadata is not a knowledge
    record and cannot be read, cited, or replayed.  The parent therefore enters
    the same strict transcript-validation and persistence path as its children.
    ``hierarchy_node_id`` is internal projection metadata; the SQL Bundle
    projection resolves it to the parent's real occurrence/knowledge ID.
    """

    output: list[ClaimOccurrenceDraft] = []
    for thesis in theses:
        if not thesis.semantic_segment_ids:
            raise ValueError("chapter thesis must overlap a semantic segment")
        scope = {
            "start_segment_index": thesis.start_segment_index,
            "end_segment_index": thesis.end_segment_index,
            "proposal_segment_indices": list(thesis.proposal_segment_indices),
            "argument_segment_indices": list(thesis.argument_segment_indices),
            "conclusion_segment_indices": list(thesis.conclusion_segment_indices),
        }
        output.append(
            ClaimOccurrenceDraft(
                semantic_segment_id=thesis.semantic_segment_ids[0],
                knowledge_kind="THESIS",
                claim_type=thesis.claim_type,
                subject_type=thesis.subject_type,
                subject_key=thesis.subject_key,
                subject_name=thesis.subject_name,
                predicate_key=thesis.predicate,
                conclusion=thesis.atomic_statement,
                normalized_statement=thesis.atomic_statement,
                verbatim_quote=thesis.atomic_statement,
                evidence_segment_indices=list(thesis.evidence_segment_indices),
                sentiment=thesis.sentiment,
                extraction_confidence=1.0,
                extraction_model_id=thesis.model_id or "chapter-thesis-identifier",
                extraction_prompt_version="chapter-thesis.v2",
                bundle_v2={
                    "claim_nature": "CAUSAL_THESIS",
                    "attribution": {
                        "attributed": True,
                        "source_label": thesis.attribution,
                    },
                    "source_grade": "SOURCE_ASSERTION",
                    "external_truth_status": "NOT_CHECKED",
                    "detail": {"explanation": thesis.detailed_explanation},
                    "knowledge_title": thesis.knowledge_title,
                    "knowledge_role": KnowledgeRole.THESIS.value,
                    "hierarchy_node_id": thesis.thesis_id,
                    "thesis_evidence_scope": scope,
                },
            )
        )
    return output


def permitted_thesis_evidence_indices(
    draft: Any,
    semantic_segment: Any,
    available_indices: Iterable[int],
) -> set[int]:
    """Return the fail-closed coordinate authority for a draft.

    Ordinary atomic claims remain confined to one semantic segment.  A
    materialized THESIS may span adjacent semantic segments, but only when it
    carries the complete chapter scope produced by the earlier thesis stage.
    """

    semantic_range = set(
        range(int(semantic_segment.start_segment_index), int(semantic_segment.end_segment_index) + 1)
    )
    bundle = getattr(draft, "bundle_v2", {}) or {}
    if not isinstance(bundle, Mapping) or bundle.get("knowledge_role") != KnowledgeRole.THESIS.value:
        return semantic_range
    scope = bundle.get("thesis_evidence_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("thesis evidence scope is required")
    try:
        start = int(scope["start_segment_index"])
        end = int(scope["end_segment_index"])
        rhetorical = [
            tuple(int(value) for value in scope[name])
            for name in (
                "proposal_segment_indices",
                "argument_segment_indices",
                "conclusion_segment_indices",
            )
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("thesis evidence scope is malformed") from exc
    if end < start or any(not values for values in rhetorical):
        raise ValueError("thesis evidence scope is empty or inverted")
    if min(rhetorical[1]) < min(rhetorical[0]) or min(rhetorical[2]) < min(rhetorical[1]):
        raise ValueError("thesis rhetorical spans are not ordered")
    allowed = {int(index) for index in available_indices if start <= int(index) <= end}
    if not allowed or min(allowed) != start or max(allowed) != end:
        raise ValueError("thesis evidence scope is outside transcript authority")
    if not semantic_range.intersection(allowed):
        raise ValueError("thesis does not overlap its selected semantic segment")
    if any(index not in allowed for values in rhetorical for index in values):
        raise ValueError("thesis rhetorical coordinate is outside its evidence scope")
    if set(int(value) for value in getattr(draft, "evidence_segment_indices", ())) != allowed:
        raise ValueError("thesis must cite its complete evidence scope")
    return allowed


__all__ = [
    "KNOWLEDGE_ROLES",
    "ChapterThesis",
    "ChapterThesisIdentifier",
    "KnowledgeRole",
    "attach_thesis_hierarchy",
    "materialize_thesis_claim_drafts",
    "permitted_thesis_evidence_indices",
    "role_for_claim",
]

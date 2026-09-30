"""Pure policies for evidence-scoped local ambiguity review.

The model-facing operator scripts may discover both a narrow ambiguity and a
larger inherited window that contains it.  Review the narrow item once instead
of spending model/vision work on both representations of the same uncertainty.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Iterable

RESOLVED_REVIEW_STATUSES = {
    "RESOLVED_BY_AUDIO",
    "RESOLVED_BY_VISUAL",
    "RESOLVED_BY_CROSS_MODAL",
    "RESOLVED_BY_VIDEO_CONTEXT",
}

ACTIVE_AMBIGUITY_KINDS = {"ENTITY", "TERM", "NUMBER", "UNIT", "DATE", "EVENT"}
HIGH_RISK_AMBIGUITY_KINDS = {"ENTITY", "NUMBER", "UNIT", "EVENT"}
TRIAGE_ACTIONS = {
    "REMOVE_FALSE_POSITIVE",
    "AUDIO_REVIEW_REQUIRED",
    "VISUAL_REVIEW_REQUIRED",
    "KEEP_UNRESOLVED",
    "RESOLVED_BY_VIDEO_CONTEXT",
}
ENTITY_TYPES = {
    "EQUITY",
    "INDEX",
    "ORGANIZATION",
    "PERSON",
    "PRODUCT",
    "TECHNOLOGY",
    "PLACE",
    "GENERIC",
    "NOT_ENTITY",
}


def granular_unresolved_items(
    items: Iterable[dict], *, kinds: set[str] | None = None
) -> list[dict]:
    """Return valid unresolved items with duplicate/superset windows removed.

    A strict superset is redundant only when it has the same ambiguity kind and
    overlaps a narrower item.  This keeps independent entity and term reviews
    separate even when they share a transcript segment.
    """

    candidates: list[dict] = []
    for source in items:
        kind = source.get("kind")
        indices = source.get("segment_indices")
        if (
            source.get("status") != "UNRESOLVED"
            or not isinstance(kind, str)
            or (kinds is not None and kind not in kinds)
            or not isinstance(indices, list)
            or not indices
            or any(not isinstance(index, int) or index < 0 for index in indices)
        ):
            continue
        item = deepcopy(source)
        item["segment_indices"] = sorted(set(indices))
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            len(item["segment_indices"]),
            item["segment_indices"][0],
            item["kind"],
            str(item.get("raw_text") or ""),
        )
    )
    selected: list[dict] = []
    seen_exact: set[tuple[str, tuple[int, ...], str]] = set()
    for item in candidates:
        coordinates = tuple(item["segment_indices"])
        exact = (item["kind"], coordinates, str(item.get("raw_text") or ""))
        if exact in seen_exact:
            continue
        coordinate_set = set(coordinates)
        if any(
            existing["kind"] == item["kind"]
            and set(existing["segment_indices"]).issubset(coordinate_set)
            for existing in selected
        ):
            continue
        selected.append(item)
        seen_exact.add(exact)
    return sorted(selected, key=lambda item: (item["segment_indices"][0], item["kind"]))


def partition_review_items(items: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Split true pending items from resolutions retained for audit display."""

    unresolved: list[dict] = []
    resolved: list[dict] = []
    for item in items:
        (resolved if item.get("status") in RESOLVED_REVIEW_STATUSES else unresolved).append(item)
    return unresolved, resolved


def ambiguity_item_id(knowledge_id: str, item: dict) -> str:
    """Return a stable identity for one card-scoped ambiguity record."""

    coordinates = item.get("segment_indices") or []
    if (
        not isinstance(knowledge_id, str)
        or not knowledge_id
        or not isinstance(item.get("kind"), str)
        or not isinstance(coordinates, list)
        or not coordinates
        or any(not isinstance(index, int) or index < 0 for index in coordinates)
    ):
        raise ValueError("Ambiguity item cannot be assigned a stable identity")
    payload = json.dumps(
        {
            "knowledge_id": knowledge_id,
            "kind": item["kind"],
            "raw_text": str(item.get("raw_text") or ""),
            "segment_indices": sorted(set(coordinates)),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"ambiguity-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def card_has_high_risk_ambiguity(card: dict) -> bool:
    """Whether a card needs bounded model/audio/visual ambiguity review."""

    return any(
        item.get("status") == "UNRESOLVED"
        and item.get("kind") in HIGH_RISK_AMBIGUITY_KINDS
        for item in card.get("unresolved_items") or []
    )


def validate_triage_decision(decision: dict, source: dict) -> dict:
    """Validate and normalize one model triage decision against source evidence."""

    if decision.get("item_id") != source.get("item_id"):
        raise ValueError("Ambiguity triage item identity changed")
    action = decision.get("action")
    if action not in TRIAGE_ACTIONS:
        raise ValueError("Ambiguity triage action is invalid")
    corrected_kind = decision.get("corrected_kind")
    if corrected_kind is not None and corrected_kind not in ACTIVE_AMBIGUITY_KINDS:
        raise ValueError("Ambiguity triage corrected kind is invalid")
    entity_type = decision.get("entity_type")
    effective_kind = corrected_kind or source.get("kind")
    if effective_kind == "ENTITY":
        if action == "REMOVE_FALSE_POSITIVE" and entity_type is None:
            entity_type = "NOT_ENTITY"
        if entity_type not in ENTITY_TYPES:
            raise ValueError("Entity ambiguity triage is missing its entity type")
    elif entity_type is not None:
        raise ValueError("Non-entity ambiguity triage must not carry an entity type")
    resolution = decision.get("resolution")
    if action == "RESOLVED_BY_VIDEO_CONTEXT":
        if effective_kind != "DATE" or not isinstance(resolution, str) or not resolution.strip():
            raise ValueError("Only a date with an explicit resolution may use video context")
    elif resolution is not None:
        raise ValueError("Unresolved ambiguity triage must not invent a resolution")
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Ambiguity triage reason is missing")
    candidate_text = decision.get("candidate_text")
    if candidate_text is not None and (
        not isinstance(candidate_text, str) or not candidate_text.strip()
    ):
        raise ValueError("Ambiguity triage candidate text is invalid")
    review_text = decision.get("review_text") or source["raw_text"]
    if (
        not isinstance(review_text, str)
        or not review_text.strip()
        or review_text not in source["raw_text"]
    ):
        raise ValueError("Ambiguity triage review text must be a literal narrowed source span")
    return {
        "item_id": source["item_id"],
        "knowledge_id": source["knowledge_id"],
        "source_kind": source["kind"],
        "source_raw_text": source["raw_text"],
        "source_segment_indices": list(source["segment_indices"]),
        "action": action,
        "corrected_kind": corrected_kind,
        "entity_type": entity_type,
        "candidate_text": candidate_text,
        "review_text": review_text.strip(),
        "resolution": resolution,
        "reason": reason.strip(),
    }

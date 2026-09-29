"""Pure policies for evidence-scoped local ambiguity review.

The model-facing operator scripts may discover both a narrow ambiguity and a
larger inherited window that contains it.  Review the narrow item once instead
of spending model/vision work on both representations of the same uncertainty.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

RESOLVED_REVIEW_STATUSES = {
    "RESOLVED_BY_AUDIO",
    "RESOLVED_BY_VISUAL",
    "RESOLVED_BY_CROSS_MODAL",
    "RESOLVED_BY_VIDEO_CONTEXT",
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

from __future__ import annotations

from collections import defaultdict


class CrossVideoCorroboration:
    """Measures repeated market narratives; it never verifies a fact."""

    def annotate(self, units: list[dict], all_units: list[dict]) -> list[dict]:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for item in all_units:
            key = (str(item.get("subject_key") or ""), str(item.get("predicate_key") or ""))
            if key != ("", "") and item.get("support_status") in {
                "SOURCE_SUPPORTED",
                "CROSS_MODAL_SUPPORTED",
                "EXTERNALLY_VERIFIED",
                "VALIDATED",
            }:
                groups[key].append(item)
        results = []
        for unit in units:
            key = (str(unit.get("subject_key") or ""), str(unit.get("predicate_key") or ""))
            corroborators = groups.get(key, [])
            sources = sorted(
                {item.get("source_video_id") for item in corroborators if item.get("source_video_id") is not None}
            )
            sentiments = {str(item.get("sentiment") or "NEUTRAL") for item in corroborators}
            narrative = {
                "distinct_video_count": len(sources),
                "corroborating_unit_count": len(corroborators),
                "consensus": "MIXED" if len(sentiments) > 1 else (next(iter(sentiments)) if sentiments else "NONE"),
                "meaning": "market_narrative_strength_only",
                "does_not_verify_truth": True,
            }
            results.append(
                dict(unit) | {"attributes": (unit.get("attributes") or {}) | {"cross_video_corroboration": narrative}}
            )
        return results

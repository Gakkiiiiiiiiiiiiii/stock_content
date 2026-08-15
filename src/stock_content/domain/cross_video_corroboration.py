from __future__ import annotations

from collections import defaultdict

from stock_content.domain.knowledge_enums import SupportStatus, support_rank


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


class CrossVideoCorroborationService:
    """Pure domain calculation; persistence belongs to the repository."""

    def calculate(self, rows: list[dict]) -> dict[str, dict]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            key = (str(row.get("subject_key") or ""), str(row.get("predicate_key") or ""))
            if key[0] and row.get("lifecycle_status") == "ACTIVE":
                grouped[key].append(row)
        result: dict[str, dict] = {}
        for group in grouped.values():
            by_video: dict[str, list[dict]] = defaultdict(list)
            for row in group:
                if support_rank(row.get("support_status")) >= support_rank(SupportStatus.SOURCE_SUPPORTED.value):
                    by_video[str(row["video_id"])].append(row)
            sentiments = [str(row.get("sentiment") or "NEUTRAL") for items in by_video.values() for row in items]
            bullish, bearish = sentiments.count("BULLISH"), sentiments.count("BEARISH")
            total = max(len(by_video), 1)
            evidence = sorted(
                {
                    str(evidence)
                    for items in by_video.values()
                    for row in items
                    for evidence in row.get("evidence_ids", [])
                }
            )
            values = {
                "corroborating_video_count": max(bullish, bearish),
                "contradicting_video_count": min(bullish, bearish),
                "independent_source_count": len(by_video),
                "content_attention_score": len(group) / total,
                "consensus_score": (bullish - bearish) / max(len(sentiments), 1),
                "disagreement_score": min(bullish, bearish) / max(len(sentiments), 1),
                "evidence_ids": evidence,
            }
            result.update({str(row["knowledge_uid"]): values for rows in by_video.values() for row in rows})
        return result

from __future__ import annotations

import hashlib


class KnowledgeConflictResolver:
    NEGATIVE = {"BEARISH"}
    POSITIVE = {"BULLISH"}
    AUTO_SUPERSEDE_KINDS = {"STATE", "TECHNICAL_SIGNAL"}
    CONDITIONAL_SUPERSEDE_KINDS = {"ACTION", "RISK_CONDITION"}
    REVIEW_ONLY_KINDS = {"METHOD", "CONCEPT", "FORECAST", "FACT"}

    def resolve(self, units: list[dict]) -> tuple[list[dict], list[dict]]:
        by_key: dict[str, list[dict]] = {}
        for unit in units:
            key = str(unit.get("conflict_key") or "")
            if not key:
                continue
            by_key.setdefault(key, []).append(unit)

        relations: list[dict] = []
        for key, group in by_key.items():
            if len(group) <= 1:
                continue
            group_id = "kcg_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            for unit in group:
                unit["conflict_group_id"] = group_id
            ordered = sorted(
                group, key=lambda item: (item.get("as_of_time") is not None, item.get("as_of_time")), reverse=True
            )
            latest = ordered[0]
            for older in ordered[1:]:
                if self._contradicts(latest, older):
                    relation_type = self._relation_for_contradiction(latest, older)
                    if relation_type == "SUPERSEDES":
                        older["lifecycle_status"] = "SUPERSEDED"
                    else:
                        self._mark_review_if_needed(latest, older)
                    attributes = self._resolution_attributes(latest, older, relation_type=relation_type)
                    relations.append(
                        {
                            "source_uid": latest.get("knowledge_uid"),
                            "target_uid": older.get("knowledge_uid"),
                            "relation_type": relation_type,
                            "confidence_score": 0.72 if relation_type == "SUPERSEDES" else 0.64,
                            "attributes": attributes,
                        }
                    )
                else:
                    attributes = self._resolution_attributes(latest, older, relation_type="REINFORCES")
                    relations.append(
                        {
                            "source_uid": latest.get("knowledge_uid"),
                            "target_uid": older.get("knowledge_uid"),
                            "relation_type": "REINFORCES",
                            "confidence_score": 0.62,
                            "attributes": attributes,
                        }
                    )
        return units, relations

    @classmethod
    def _contradicts(cls, left: dict, right: dict) -> bool:
        left_sentiment = str(left.get("sentiment") or "")
        right_sentiment = str(right.get("sentiment") or "")
        return (left_sentiment in cls.POSITIVE and right_sentiment in cls.NEGATIVE) or (
            left_sentiment in cls.NEGATIVE and right_sentiment in cls.POSITIVE
        )

    @classmethod
    def _relation_for_contradiction(cls, latest: dict, older: dict) -> str:
        kind = str(latest.get("knowledge_kind") or older.get("knowledge_kind") or "STATE")
        if kind in cls.AUTO_SUPERSEDE_KINDS:
            return "SUPERSEDES"
        if kind in cls.CONDITIONAL_SUPERSEDE_KINDS:
            if latest.get("condition_text") or latest.get("invalidation_text"):
                return "SUPERSEDES"
            return "CONFLICTS_WITH"
        if kind in cls.REVIEW_ONLY_KINDS:
            return "CONFLICTS_WITH"
        return "CONFLICTS_WITH"

    @classmethod
    def _mark_review_if_needed(cls, latest: dict, older: dict) -> None:
        kind = str(latest.get("knowledge_kind") or older.get("knowledge_kind") or "STATE")
        if kind in {"METHOD", "CONCEPT", "FACT", "ACTION", "RISK_CONDITION"}:
            latest["verification_status"] = "NEEDS_REVIEW"
            older["verification_status"] = "NEEDS_REVIEW"

    @classmethod
    def _resolution_attributes(cls, latest: dict, older: dict, *, relation_type: str) -> dict:
        kind = str(latest.get("knowledge_kind") or older.get("knowledge_kind") or "STATE")
        reason = (
            "same_conflict_key_same_direction"
            if relation_type == "REINFORCES"
            else "same_conflict_key_newer_opposite_sentiment"
        )
        if kind == "STATE":
            recommended_action = "keep_latest_as_current"
        elif kind == "ACTION":
            recommended_action = (
                "retire_stale_action" if relation_type == "SUPERSEDES" else "review_or_retire_stale_action"
            )
        elif kind == "FORECAST":
            recommended_action = "keep_forecast_history"
        elif kind in {"METHOD", "CONCEPT"}:
            recommended_action = "manual_review_before_supersede"
        elif kind == "FACT":
            recommended_action = "require_evidence_verification"
        else:
            recommended_action = "review_conflict_group"
        return {
            "reason": reason,
            "conflict_resolution_reason": reason,
            "recommended_action": recommended_action,
            "conflict_scope": "same_video"
            if latest.get("source_video_id") == older.get("source_video_id")
            else "cross_video_or_unknown",
            "knowledge_kind": kind,
        }

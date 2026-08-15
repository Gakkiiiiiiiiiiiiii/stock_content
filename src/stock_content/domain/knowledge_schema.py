from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stock_content.domain.knowledge_enums import KnowledgeKind

REQUIRED_FIELDS = {
    "primary_domain",
    "knowledge_kind",
    "expression_type",
    "statement",
    "canonical_statement",
    "evidence",
}

DOMAIN_LEVEL_KINDS = {"METHOD", "CONCEPT", "CAUSAL_THESIS"}

# P0-1：SchemaValidator 不得再重建"缩减版 Evidence"，按白名单透传全部字段。
EVIDENCE_FIELDS = {
    "source_type",
    "source_ref",
    "evidence_text",
    "start_ms",
    "end_ms",
    "frame_id",
    "confidence_score",
    "is_primary",
    "raw_text",
    "normalized_text",
    "speaker_id",
    "speaker_attribution_confidence",
    "word_timestamps",
    "bbox",
    "asr_metrics",
    "ocr_metrics",
    "correction_trace",
    "semantic_support_score",
    "numeric_consistency_score",
    "entity_consistency_score",
}


@dataclass
class KnowledgeSchemaValidation:
    valid_units: list[dict] = field(default_factory=list)
    rejected_units: list[dict] = field(default_factory=list)
    repaired_units: list[dict] = field(default_factory=list)

    @property
    def metrics(self) -> dict:
        return {
            "accepted_count": len(self.valid_units),
            "rejected_count": len(self.rejected_units),
            "repaired_count": len(self.repaired_units),
            "rejection_reasons": [item["reason"] for item in self.rejected_units],
        }


class KnowledgeUnitSchemaValidator:
    def validate_many(self, units: list[dict], *, chapter: dict | None = None) -> KnowledgeSchemaValidation:
        result = KnowledgeSchemaValidation()
        for unit in units:
            validated = self.validate_one(unit, chapter=chapter)
            if validated["accepted"]:
                result.valid_units.append(validated["unit"])
                if validated["repaired"]:
                    result.repaired_units.append({"unit": validated["unit"], "repairs": validated["repairs"]})
            else:
                result.rejected_units.append({"unit": unit, "reason": validated["reason"]})
        return result

    def validate_one(self, unit: dict, *, chapter: dict | None = None) -> dict:
        item = dict(unit or {})
        repairs: list[str] = []
        for field_name in REQUIRED_FIELDS:
            if field_name == "evidence":
                continue
            if not str(item.get(field_name) or "").strip():
                fallback = self._fallback_value(field_name, item, chapter)
                if fallback:
                    item[field_name] = fallback
                    repairs.append(f"filled_{field_name}")
                else:
                    return self._reject(item, f"missing_{field_name}")
        # P1-7 / §61：normalize（含 fallback 回填）后 knowledge_kind 必须属于统一枚举。
        if str(item.get("knowledge_kind") or "") not in KnowledgeKind.values():
            return self._reject(item, "unknown_knowledge_kind")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            return self._reject(item, "missing_evidence")
        normalized_evidence = []
        for evidence_item in evidence:
            if not isinstance(evidence_item, dict):
                continue
            text = str(evidence_item.get("evidence_text") or evidence_item.get("text") or "").strip()
            if not text:
                continue
            # P0-2：禁止 `or` 取值，0.0 是合法置信度，必须原样保留。
            confidence = evidence_item.get("confidence_score")
            if confidence is None:
                confidence = evidence_item.get("confidence")
            normalized = {key: evidence_item.get(key) for key in EVIDENCE_FIELDS if key in evidence_item}
            normalized["source_type"] = str(
                evidence_item.get("source_type") or evidence_item.get("evidence_type") or "ASR"
            )
            normalized["evidence_text"] = text
            normalized["confidence_score"] = confidence
            normalized["is_primary"] = bool(evidence_item.get("is_primary", len(normalized_evidence) == 0))
            normalized_evidence.append(normalized)
        if not normalized_evidence:
            return self._reject(item, "empty_evidence_text")
        item["evidence"] = normalized_evidence
        if not item.get("entities") and (chapter or {}).get("entities"):
            item["entities"] = [
                {
                    "entity_type": "SECURITY"
                    if any(char.isdigit() for char in str(entity))
                    else str((chapter or {}).get("primary_domain") or "GENERAL"),
                    "entity_key": str(entity),
                    "entity_name": str(entity),
                    "relation_role": "SUBJECT",
                    # chapter 推导实体无测量置信度：保持 UNKNOWN（None），禁止伪造 0.65。
                    "confidence_score": None,
                }
                for entity in (chapter or {}).get("entities") or []
            ]
            repairs.append("copied_chapter_entities")
        if not self._has_subject(item, chapter):
            if self._allow_domain_level(item, chapter):
                domain = str((chapter or {}).get("primary_domain") or item.get("primary_domain") or "GENERAL")
                item["subject_type"] = "DOMAIN"
                item["subject_key"] = domain
                item["subject_name"] = domain
                attrs = dict(item.get("attributes") or {})
                attrs["domain_level"] = True
                item["attributes"] = attrs
                item["verification_status"] = "NEEDS_REVIEW"
                repairs.append("marked_domain_level_subject")
            else:
                return self._reject(item, "missing_subject")
        if not self._has_subject(item, chapter):
            return self._reject(item, "missing_subject")
        if not str(item.get("predicate_key") or "").strip():
            return self._reject(item, "missing_predicate_key")
        return {"accepted": True, "unit": item, "repaired": bool(repairs), "repairs": repairs, "reason": None}

    @staticmethod
    def _fallback_value(field_name: str, unit: dict, chapter: dict | None) -> Any:
        if field_name == "primary_domain":
            return (chapter or {}).get("primary_domain") or "GENERAL"
        if field_name == "knowledge_kind":
            return "STATE"
        if field_name == "expression_type":
            return "AUTHOR_EXPLICIT"
        if field_name == "canonical_statement":
            return unit.get("statement")
        return None

    @staticmethod
    def _has_subject(unit: dict, chapter: dict | None) -> bool:
        if unit.get("subject_key") or unit.get("subject_name"):
            return True
        if unit.get("entities"):
            return True
        if (chapter or {}).get("entities"):
            return True
        return False

    @staticmethod
    def _allow_domain_level(unit: dict, chapter: dict | None) -> bool:
        return str(unit.get("knowledge_kind") or "") in DOMAIN_LEVEL_KINDS and bool(
            (chapter or {}).get("primary_domain") or unit.get("primary_domain")
        )

    @staticmethod
    def _reject(unit: dict, reason: str) -> dict:
        return {"accepted": False, "unit": unit, "repaired": False, "repairs": [], "reason": reason}

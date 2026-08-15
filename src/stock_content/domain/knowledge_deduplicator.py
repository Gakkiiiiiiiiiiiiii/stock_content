from __future__ import annotations

import re
from difflib import SequenceMatcher


class KnowledgeDeduplicator:
    def deduplicate(self, units: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for unit in units:
            key = str(unit.get("content_hash") or unit.get("semantic_hash") or unit.get("statement"))
            if key not in seen:
                seen[key] = unit
                continue
            existing = seen[key]
            existing_evidence = existing.setdefault("evidence", [])
            for evidence in unit.get("evidence") or []:
                if evidence not in existing_evidence:
                    existing_evidence.append(evidence)
            existing["extraction_confidence"] = max(
                float(existing.get("extraction_confidence") or 0), float(unit.get("extraction_confidence") or 0)
            )
        selected: list[dict] = []
        for unit in seen.values():
            duplicate = next((existing for existing in selected if self._same_claim(existing, unit)), None)
            if duplicate is None:
                selected.append(unit)
                continue
            existing_evidence = duplicate.setdefault("evidence", [])
            for evidence in unit.get("evidence") or []:
                if evidence not in existing_evidence:
                    existing_evidence.append(evidence)
            duplicate["extraction_confidence"] = max(
                float(duplicate.get("extraction_confidence") or 0),
                float(unit.get("extraction_confidence") or 0),
            )
        return selected

    @staticmethod
    def _same_claim(left: dict, right: dict) -> bool:
        left_subject = str(left.get("subject_key") or "")
        right_subject = str(right.get("subject_key") or "")
        if left_subject and right_subject and left_subject != right_subject:
            return False
        left_text = re.sub(r"\s+", "", str(left.get("canonical_statement") or left.get("statement") or ""))
        right_text = re.sub(r"\s+", "", str(right.get("canonical_statement") or right.get("statement") or ""))
        if not left_text or not right_text:
            return False
        if left_text == right_text or left_text in right_text or right_text in left_text:
            return True
        same_kind = str(left.get("knowledge_kind") or "") == str(right.get("knowledge_kind") or "")
        return same_kind and SequenceMatcher(a=left_text, b=right_text).ratio() >= 0.84

"""Deterministic financial-event projection from verified knowledge units."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


class FinancialEventExtractor:
    _EVENT_TYPES = {
        "指引": "GUIDANCE",
        "业绩": "EARNINGS",
        "营收": "EARNINGS",
        "净利润": "EARNINGS",
        "订单": "ORDER_CONTRACT",
        "合同": "ORDER_CONTRACT",
        "扩产": "CAPACITY",
        "发布": "PRODUCT_LAUNCH",
        "政策": "POLICY_REGULATION",
        "监管": "POLICY_REGULATION",
        "收购": "M_AND_A",
        "回购": "BUYBACK",
        "减持": "SHAREHOLDER_CHANGE",
        "风险": "RISK",
    }

    def extract(self, units: Iterable[dict]) -> list[dict]:
        events: list[dict] = []
        for unit in units:
            statement = str(unit.get("statement") or "")
            event_type = next((kind for term, kind in self._EVENT_TYPES.items() if term in statement), None)
            if event_type is None:
                continue
            knowledge_uid = str(unit.get("knowledge_uid") or "")
            digest = hashlib.sha256(f"{knowledge_uid}:{event_type}".encode()).hexdigest()[:32]
            events.append(
                {
                    "event_id": f"evt_{digest}",
                    "knowledge_uid": knowledge_uid,
                    "event_type": event_type,
                    "subject_key": unit.get("subject_key") or unit.get("ticker"),
                    "event_time": unit.get("as_of"),
                    "effective_time": unit.get("valid_from"),
                    "available_from": unit.get("available_from"),
                    "direction": unit.get("sentiment"),
                    "strength": unit.get("confidence"),
                    "numeric_refs": list(unit.get("numeric_ids") or []),
                    "evidence_refs": list(unit.get("evidence_ids") or []),
                    "confidence": unit.get("confidence"),
                }
            )
        return events

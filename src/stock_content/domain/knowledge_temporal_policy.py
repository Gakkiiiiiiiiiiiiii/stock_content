from __future__ import annotations

from datetime import UTC, datetime, timedelta


class KnowledgeTemporalPolicy:
    DEFAULTS = {
        "METHOD": ("DURABLE", None, 730.0),
        "CONCEPT": ("DURABLE", None, 730.0),
        "CAUSAL_THESIS": ("CYCLICAL", 365, 90.0),
        "FACT": ("SNAPSHOT", 2, 1.0),
        "STATE": ("SNAPSHOT", 14, 5.0),
        "FORECAST": ("EVENT_BOUND", 90, 21.0),
        "TECHNICAL_SIGNAL": ("SNAPSHOT", 7, 2.0),
        "ACTION": ("EVENT_BOUND", 14, 5.0),
        "RISK_CONDITION": ("EVENT_BOUND", 30, 10.0),
        "MODEL_INFERENCE": ("SNAPSHOT", 14, 5.0),
    }

    DOMAIN_OVERRIDES = {
        ("MACRO", "STATE"): ("CYCLICAL", 45, 14.0),
        ("INDUSTRY", "STATE"): ("CYCLICAL", 90, 30.0),
        ("CAPITAL_FLOW", "STATE"): ("SNAPSHOT", 7, 3.0),
        ("TRADING", "ACTION"): ("EVENT_BOUND", 14, 5.0),
    }

    def apply(self, units: list[dict], source_date: datetime | None = None) -> list[dict]:
        as_of = source_date or datetime.now(UTC)
        results = []
        for unit in units:
            kind = str(unit.get("knowledge_kind") or "STATE")
            domain = str(unit.get("primary_domain") or "GENERAL")
            temporal_class, valid_days, half_life = self.DOMAIN_OVERRIDES.get(
                (domain, kind), self.DEFAULTS.get(kind, ("SNAPSHOT", 14, 5.0))
            )
            enriched = dict(unit)
            enriched["temporal_class"] = enriched.get("temporal_class") or temporal_class
            if kind in {"FACT", "STATE", "FORECAST", "TECHNICAL_SIGNAL", "ACTION", "RISK_CONDITION", "MODEL_INFERENCE"}:
                enriched["as_of_time"] = enriched.get("as_of_time") or as_of
                enriched["valid_from"] = enriched.get("valid_from") or as_of
            if enriched.get("valid_to") is None and valid_days is not None:
                enriched["valid_to"] = as_of + timedelta(days=valid_days)
            enriched["decay_half_life_days"] = enriched.get("decay_half_life_days") or half_life
            enriched["lifecycle_status"] = enriched.get("lifecycle_status") or "ACTIVE"
            if enriched.get("valid_to") and enriched["valid_to"] < datetime.now(UTC):
                enriched["lifecycle_status"] = "EXPIRED"
            results.append(enriched)
        return results

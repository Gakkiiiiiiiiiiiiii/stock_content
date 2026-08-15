from __future__ import annotations

import os
from typing import Any, Protocol


class ExternalFactProvider(Protocol):
    def verify(self, unit: dict[str, Any]) -> dict[str, Any]: ...


class ExternalFactVerifier:
    """Keeps external truth separate from source-evidence support."""

    ELIGIBLE_KINDS = {"FACT", "VALUATION", "FINANCIAL_METRIC", "PRICE_LEVEL", "POLICY_FACT"}

    def __init__(self, provider: ExternalFactProvider | None = None) -> None:
        self._provider = provider
        self._enabled = os.getenv("CONTENT_EXTERNAL_FACT_VERIFICATION", "false").lower() in {"1", "true", "yes"}

    def verify_many(self, units: list[dict]) -> list[dict]:
        results = []
        for unit in units:
            item = dict(unit)
            if str(item.get("knowledge_kind") or "").upper() not in self.ELIGIBLE_KINDS:
                results.append(item)
                continue
            if not self._enabled or self._provider is None:
                results.append(
                    item
                    | {
                        "external_verification_status": "NOT_RUN",
                        "truth_status": item.get("truth_status") or "NOT_CHECKED",
                    }
                )
                continue
            outcome = self._provider.verify(item)
            status = str(outcome.get("status") or "NOT_FOUND").upper()
            attributes = (item.get("attributes") or {}) | {"external_verification": outcome}
            if status == "MATCH":
                results.append(
                    item
                    | {
                        "external_verification_status": "EXTERNAL_MATCH",
                        "truth_status": "EXTERNALLY_VERIFIED",
                        "attributes": attributes,
                    }
                )
            elif status == "CONFLICT":
                results.append(
                    item
                    | {
                        "external_verification_status": "EXTERNAL_CONFLICT",
                        "truth_status": "EXTERNAL_CONFLICT",
                        "attributes": attributes,
                    }
                )
            else:
                results.append(
                    item
                    | {
                        "external_verification_status": "EXTERNAL_NOT_FOUND",
                        "truth_status": "NOT_FOUND",
                        "attributes": attributes,
                    }
                )
        return results

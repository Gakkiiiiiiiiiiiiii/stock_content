"""Compatibility response metadata for legacy signal endpoints."""

from __future__ import annotations

from typing import Any


def compatibility_response(contract: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "contract_version": contract,
        "authority": "COMPATIBILITY_READ_ONLY",
        "formal_eligible": False,
        "items": items,
    }


__all__ = ["compatibility_response"]

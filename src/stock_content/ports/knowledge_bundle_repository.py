from __future__ import annotations

from typing import Any, Protocol

from stock_content.domain.knowledge_bundle import KnowledgeBundleRequest


class KnowledgeBundleRepository(Protocol):
    def insert(self, bundle: dict[str, Any]) -> dict[str, Any]: ...
    def get(self, bundle_id: str) -> dict[str, Any] | None: ...


class KnowledgeBundleAuthority(Protocol):
    """SQL-only formal snapshot read; search/Qdrant is intentionally absent."""
    def read_bundle_source(self, request: KnowledgeBundleRequest) -> dict[str, Any] | None: ...


class InMemoryKnowledgeBundleRepository:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
    def insert(self, bundle: dict[str, Any]) -> dict[str, Any]:
        existing = self._items.get(bundle["bundle_id"])
        if existing is not None:
            if existing["bundle_hash"] != bundle["bundle_hash"] or existing["payload"] != bundle:
                raise ValueError("immutable bundle id collision")
            return dict(existing["payload"])
        self._items[bundle["bundle_id"]] = {"bundle_hash": bundle["bundle_hash"], "payload": dict(bundle)}
        return dict(bundle)
    def get(self, bundle_id: str) -> dict[str, Any] | None:
        item = self._items.get(bundle_id)
        return None if item is None else dict(item["payload"])

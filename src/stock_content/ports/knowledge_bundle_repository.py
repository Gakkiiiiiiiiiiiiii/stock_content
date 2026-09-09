from __future__ import annotations

from typing import Any, Protocol

from stock_content.domain.knowledge_bundle import KnowledgeBundleRequest


class KnowledgeBundleRepository(Protocol):
    def insert(
        self,
        bundle: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        idempotency_request_hash: str | None = None,
    ) -> dict[str, Any]: ...
    def get(self, bundle_id: str) -> dict[str, Any] | None: ...
    def get_idempotent(self, *, idempotency_key: str, idempotency_request_hash: str) -> dict[str, Any] | None: ...


class KnowledgeBundleAuthority(Protocol):
    """SQL-only formal snapshot read; search/Qdrant is intentionally absent."""
    def read_bundle_source(self, request: KnowledgeBundleRequest) -> dict[str, Any] | None: ...


class InMemoryKnowledgeBundleRepository:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}

    def insert(
        self,
        bundle: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        idempotency_request_hash: str | None = None,
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            if not idempotency_request_hash:
                raise ValueError("idempotency request hash is required")
            existing = self.get_idempotent(
                idempotency_key=idempotency_key,
                idempotency_request_hash=idempotency_request_hash,
            )
            if existing is not None:
                return existing
        existing = self._items.get(bundle["bundle_id"])
        if existing is not None:
            if existing["bundle_hash"] != bundle["bundle_hash"] or existing["payload"] != bundle:
                raise ValueError("immutable bundle id collision")
            result = dict(existing["payload"])
        else:
            self._items[bundle["bundle_id"]] = {"bundle_hash": bundle["bundle_hash"], "payload": dict(bundle)}
            result = dict(bundle)
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = (str(idempotency_request_hash), result["bundle_id"])
        return result

    def get(self, bundle_id: str) -> dict[str, Any] | None:
        item = self._items.get(bundle_id)
        return None if item is None else dict(item["payload"])

    def get_idempotent(self, *, idempotency_key: str, idempotency_request_hash: str) -> dict[str, Any] | None:
        from stock_content.ports.repositories import IdempotencyConflict

        existing = self._idempotency.get(idempotency_key)
        if existing is None:
            return None
        request_hash, bundle_id = existing
        if request_hash != idempotency_request_hash:
            raise IdempotencyConflict()
        return self.get(bundle_id)

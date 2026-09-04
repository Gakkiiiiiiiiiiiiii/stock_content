"""Application use case for strict, snapshot-bound formal signal queries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from stock_content.domain.bitemporal_query import FormalContentSignalQueryV2


class FormalQueryAuthority(Protocol):
    def query_formal(self, query: FormalContentSignalQueryV2) -> Iterable[dict[str, Any]]: ...


@dataclass(frozen=True)
class FormalSignalQueryResult:
    query_id: str
    content_snapshot_id: str
    clocks: dict[str, str]
    items: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "content_snapshot_id": self.content_snapshot_id,
            "business_as_of": self.clocks["business_as_of"],
            "knowledge_as_of": self.clocks["knowledge_as_of"],
            "availability_as_of": self.clocks["availability_as_of"],
            "items": list(self.items),
        }


class FormalSignalQueryService:
    """Validate and execute the sole formal query path.

    ``authority`` must read SQL/event authority.  The optional membership
    callback is intentionally fail-closed: no callback means no candidates.
    """

    def __init__(
        self, authority: FormalQueryAuthority | None = None, *, membership: Callable[[str, str], bool] | None = None
    ) -> None:
        self._authority = authority
        self._membership = membership

    def execute(self, query: FormalContentSignalQueryV2) -> FormalSignalQueryResult:
        if not isinstance(query, FormalContentSignalQueryV2):
            raise TypeError("formal query must be FormalContentSignalQueryV2")
        if self._authority is None:
            rows: Iterable[dict[str, Any]] = ()
        else:
            rows = self._authority.query_formal(query)
        items: list[dict[str, Any]] = []
        for raw in rows:
            item = dict(raw)
            claim_id = str(item.get("claim_id") or "")
            if not claim_id or self._membership is None or not self._membership(query.content_snapshot_id, claim_id):
                continue
            items.append(item)
        clocks = {
            name: getattr(query, name).isoformat().replace("+00:00", "Z")
            for name in ("business_as_of", "knowledge_as_of", "availability_as_of")
        }
        return FormalSignalQueryResult(query.query_id, query.content_snapshot_id, clocks, tuple(items))

    def query(self, value: FormalContentSignalQueryV2 | dict[str, Any]) -> FormalSignalQueryResult:
        query = (
            value if isinstance(value, FormalContentSignalQueryV2) else FormalContentSignalQueryV2.from_mapping(value)
        )
        return self.execute(query)


__all__ = ["FormalQueryAuthority", "FormalSignalQueryService", "FormalSignalQueryResult"]

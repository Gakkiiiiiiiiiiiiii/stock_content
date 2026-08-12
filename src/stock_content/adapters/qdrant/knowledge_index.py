from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Callable
from uuid import UUID

from stock_content.domain.models import KnowledgeUnit


class NullKnowledgeIndex:
    def index(self, units: list[KnowledgeUnit]) -> None:
        return None

    def search(self, query: str, limit: int) -> list[str]:
        return []


def _stable_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Local deterministic embedding keeps the service useful without a model provider."""
    vector = [0.0] * dimensions
    normalized = re.sub(r"\s+", "", text.casefold())
    tokens = [normalized[index : index + 2] for index in range(max(1, len(normalized) - 1))]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:2], "big") % dimensions] += 1.0 if digest[2] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class QdrantKnowledgeIndex:
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        embed: Callable[[str], list[float]] = _stable_embedding,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("install stock-content[search] to enable Qdrant") from exc
        self._client = QdrantClient(url=url or os.getenv("CONTENT_QDRANT_URL", "http://localhost:6333"))
        self._collection = collection or os.getenv("CONTENT_QDRANT_COLLECTION", "stock_content_knowledge_v1")
        self._embed = embed

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=256, distance=Distance.COSINE),
            )

    def index(self, units: list[KnowledgeUnit]) -> None:
        if not units:
            return
        from qdrant_client.models import PointStruct

        self._ensure_collection()
        self._client.upsert(
            self._collection,
            points=[
                PointStruct(
                    id=str(UUID(unit.knowledge_uid[:32])),
                    vector=self._embed(unit.statement),
                    payload={"knowledge_uid": unit.knowledge_uid, "ticker": unit.ticker, "kind": unit.kind},
                )
                for unit in units
            ],
        )

    def search(self, query: str, limit: int) -> list[str]:
        self._ensure_collection()
        response = self._client.query_points(
            collection_name=self._collection,
            query=self._embed(query),
            limit=limit,
            with_payload=True,
        )
        return [str(point.payload["knowledge_uid"]) for point in response.points]

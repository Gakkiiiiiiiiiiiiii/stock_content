from __future__ import annotations

import os
from collections.abc import Callable
from uuid import NAMESPACE_URL, uuid5

from stock_content.adapters.http.embedding_client import ContentEmbeddingClient
from stock_content.domain.models import KnowledgeUnit


class NullKnowledgeIndex:
    def index(self, units: list[KnowledgeUnit]) -> None:
        return None

    def search(self, query: str, limit: int) -> list[str]:
        return []


class QdrantKnowledgeIndex:
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        embed: Callable[[str], list[float]] | None = None,
        vector_size: int | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("install stock-content[search] to enable Qdrant") from exc
        self._client = QdrantClient(url=url or os.getenv("CONTENT_QDRANT_URL", "http://localhost:6333"))
        self._collection = collection or os.getenv("CONTENT_QDRANT_COLLECTION", "stock_content_knowledge_v1")
        self._embed = embed or ContentEmbeddingClient().embed
        self._vector_size = vector_size or int(os.getenv("CONTENT_EMBEDDING_DIMENSIONS", "1536"))

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
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
                    id=str(uuid5(NAMESPACE_URL, unit.knowledge_uid)),
                    vector=self._embed(unit.statement),
                    # Qdrant is only a search projection.  Keep the complete
                    # filtering/lineage envelope in the payload, while the
                    # relational row remains authoritative for hydration.
                    payload={
                        "knowledge_uid": unit.knowledge_uid,
                        "content_snapshot_id": (unit.attributes or {}).get("content_snapshot_id"),
                        "claim_ids": list((unit.provenance or {}).get("claim_ids") or []),
                        "ticker": unit.ticker,
                        "knowledge_kind": unit.knowledge_kind,
                        "kind": unit.kind,
                        "support_status": unit.support_status,
                        "verification_status": unit.truth_status,
                        "available_from": unit.available_from.isoformat(),
                    },
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

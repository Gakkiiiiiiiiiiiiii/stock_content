from __future__ import annotations

import os

import httpx


class ContentEmbeddingClient:
    """OpenAI-compatible embedding adapter; no deterministic pseudo vectors."""

    def __init__(self, url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self.url = (url or os.getenv("CONTENT_EMBEDDING_URL", "")).rstrip("/")
        self.model = model or os.getenv("CONTENT_EMBEDDING_MODEL", "")
        self.api_key = api_key or os.getenv("CONTENT_EMBEDDING_API_KEY", "")

    def embed(self, text: str) -> list[float]:
        if not self.url or not self.model:
            raise RuntimeError("CONTENT_EMBEDDING_URL and CONTENT_EMBEDDING_MODEL are required when Qdrant is enabled")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(
            self.url,
            json={"model": self.model, "input": text},
            headers=headers,
            timeout=httpx.Timeout(10.0, read=60.0),
        )
        response.raise_for_status()
        body = response.json()
        vector = ((body.get("data") or [{}])[0]).get("embedding") or body.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("embedding service returned no vector")
        return [float(value) for value in vector]

from __future__ import annotations

import os

import httpx


class HttpExternalFactProvider:
    """Configured authoritative verification API boundary for Content."""

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self._url = (url or os.getenv("CONTENT_EXTERNAL_FACT_API_URL", "")).rstrip("/")
        self._api_key = api_key or os.getenv("CONTENT_EXTERNAL_FACT_API_KEY", "")

    def configured(self) -> bool:
        return bool(self._url)

    def verify(self, unit: dict) -> dict:
        if not self._url:
            raise RuntimeError("CONTENT_EXTERNAL_FACT_API_URL is required when external fact verification is enabled")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = httpx.post(self._url, json={"unit": unit}, headers=headers, timeout=httpx.Timeout(10.0, read=45.0))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("external fact provider returned a non-object response")
        return result

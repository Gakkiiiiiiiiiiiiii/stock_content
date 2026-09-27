from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx


class ContentModelClient:
    """OpenAI-compatible model adapter owned by the Content bounded context."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("CONTENT_MODEL_URL", "")).rstrip("/")
        self.model = model or os.getenv("CONTENT_MODEL_NAME", "")
        self.api_key = api_key or os.getenv("CONTENT_MODEL_API_KEY", "")
        self.provider = "content-openai-compatible"

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def complete(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError("CONTENT_MODEL_URL and CONTENT_MODEL_NAME are required for production extraction")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        responses_api = urlparse(self.base_url).path.rstrip("/").endswith("/responses")
        if self.model == "gpt-6-sol" and not responses_api:
            raise RuntimeError("gpt-6-sol extraction requires a Responses API endpoint")
        if responses_api:
            payload: dict[str, Any] = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "reasoning": {"effort": os.getenv("CONTENT_MODEL_REASONING_EFFORT", "medium")},
            }
            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens
            if response_format is not None:
                payload["text"] = {"format": response_format}
            response = httpx.post(self.base_url, json=payload, headers=headers, timeout=httpx.Timeout(15.0, read=120.0))
            response.raise_for_status()
            body = response.json()
            if body.get("status") != "completed":
                raise ValueError("model response was not completed")
            returned_model = str(body.get("model") or "")
            if not returned_model or (self.model == "gpt-6-sol" and returned_model != self.model):
                raise ValueError("model response identity does not match gpt-6-sol")
            content = "".join(
                str(part.get("text") or "")
                for item in body.get("output") or ()
                if isinstance(item, dict) and item.get("type") == "message"
                for part in item.get("content") or ()
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
            if not content:
                raise ValueError("model response contained no output text")
            return {
                "content": content,
                "provider": self.provider,
                "model": returned_model,
                "finish_reason": body.get("status"),
                "raw_response": body,
            }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        response = httpx.post(self.base_url, json=payload, headers=headers, timeout=httpx.Timeout(15.0, read=120.0))
        response.raise_for_status()
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "content": message.get("content") or body.get("content") or body.get("output") or "",
            "provider": self.provider,
            "model": body.get("model") or self.model,
            "finish_reason": choice.get("finish_reason"),
            "raw_response": body,
        }

from __future__ import annotations

import os
import shutil
from typing import Any
from urllib.parse import urlparse

import httpx

from stock_content.adapters.codex_cli import CodexCliRunner


class ContentModelClient:
    """OpenAI-compatible model adapter owned by the Content bounded context."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self.backend = os.getenv("CONTENT_MODEL_BACKEND", "http").strip().lower()
        self.base_url = (base_url or os.getenv("CONTENT_MODEL_URL", "")).rstrip("/")
        self.model = model or os.getenv("CONTENT_MODEL_NAME", "") or (
            "gpt-6-sol" if self.backend == "codex_cli" else ""
        )
        self.api_key = api_key or os.getenv("CONTENT_MODEL_API_KEY", "")
        self.provider = "codex-cli-chatgpt-account" if self.backend == "codex_cli" else "content-openai-compatible"
        self._codex = CodexCliRunner(model=self.model) if self.backend == "codex_cli" else None

    def available(self) -> bool:
        if self._codex is not None:
            return shutil.which(self._codex.executable) is not None
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
        if self._codex is not None:
            if not self.available():
                raise RuntimeError("CONTENT_CODEX_CLI must resolve to an executable Codex CLI")
            return self._codex.run(system=system, prompt=prompt)
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
                "store": False,
            }
            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens
            if response_format is not None:
                payload["text"] = {"format": response_format}
        else:
            payload = {
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
        if responses_api:
            if body.get("status") != "completed":
                raise RuntimeError("model response did not complete")
            returned_model = str(body.get("model") or "")
            if not returned_model or (self.model == "gpt-6-sol" and returned_model != self.model):
                raise ValueError("model response identity does not match gpt-6-sol")
            output = "".join(
                str(item.get("text") or "")
                for message in body.get("output") or ()
                if isinstance(message, dict) and message.get("type") == "message"
                for item in message.get("content") or ()
                if isinstance(item, dict) and item.get("type") == "output_text"
            )
            if not output:
                raise ValueError("model response contained no output text")
            return {
                "content": output,
                "provider": self.provider,
                "model": returned_model,
                "finish_reason": body.get("status"),
                "raw_response": body,
            }
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "content": message.get("content") or body.get("content") or body.get("output") or "",
            "provider": self.provider,
            "model": body.get("model") or "",
            "finish_reason": choice.get("finish_reason"),
            "raw_response": body,
        }

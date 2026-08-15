from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx


class HttpVisionAnalyzer:
    """OpenAI-compatible vision endpoint for finance-screen interpretation."""

    def __init__(self, url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self._url = (url or os.getenv("CONTENT_VISION_URL", "")).rstrip("/")
        self._model = model or os.getenv("CONTENT_VISION_MODEL", "")
        self._key = api_key or os.getenv("CONTENT_VISION_API_KEY", "")

    def analyze(self, frame_path: str, transcript_context: str) -> dict:
        if not self._url or not self._model:
            raise RuntimeError("CONTENT_VISION_URL and CONTENT_VISION_MODEL are required for vision analysis")
        image = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        prompt = (
            "分析金融视频帧，只输出 JSON：visual_summary,themes,symbols,"
            "confidence_score,narration_aligned。口播上下文：" + transcript_context[:3000]
        )
        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        response = httpx.post(self._url, json=body, headers=headers, timeout=httpx.Timeout(10, read=90))
        response.raise_for_status()
        payload = response.json()
        import json

        return json.loads(((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "{}")

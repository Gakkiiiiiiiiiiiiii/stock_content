from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
from typing import Any

import httpx


def _finite_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("vision confidence_score must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("vision confidence_score must be between 0 and 1")
    return result


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"vision {field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"vision {field} must be a list of non-empty strings")
    if required and not value:
        raise ValueError(f"vision {field} must not be empty")
    return list(value)


class HttpVisionAnalyzer:
    """Strict OpenAI-compatible visual context adapter.

    Transcript evidence remains primary. The adapter returns only a validated,
    non-authoritative visual observation and never invents missing fields.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self._url = (url or os.getenv("CONTENT_VISION_URL", "")).rstrip("/")
        self._model = model or os.getenv("CONTENT_VISION_MODEL", "")
        self._key = api_key or os.getenv("CONTENT_VISION_API_KEY", "")
        self._model_version = model_version or os.getenv("CONTENT_VISION_MODEL_VERSION", "") or self._model

    def analyze(self, frame_path: str, transcript_context: str) -> dict:
        if not self._url or not self._model:
            raise RuntimeError("CONTENT_VISION_URL and CONTENT_VISION_MODEL are required for vision analysis")
        image = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        prompt = (
            "分析金融视频帧，只输出 JSON 对象，严格包含："
            "visual_summary(非空字符串),labels(至少一个非空标签),themes(字符串数组),"
            "symbols(字符串数组),confidence_score(0到1有限数字),narration_aligned(布尔值)。"
            "只描述画面，不得创造金融事实。口播上下文：" + transcript_context[:3000]
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
        try:
            payload = response.json()
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            result = json.loads(content or "{}")
        except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("vision endpoint returned invalid JSON") from exc
        return self._validate(result)

    def _validate(self, result: Any) -> dict:
        if not isinstance(result, dict):
            raise ValueError("vision endpoint response must be an object")
        narration_aligned = result.get("narration_aligned")
        if not isinstance(narration_aligned, bool):
            raise ValueError("vision narration_aligned must be boolean")
        return {
            "visual_summary": _string(result.get("visual_summary"), "visual_summary"),
            "labels": _string_list(result.get("labels"), "labels", required=True),
            "themes": _string_list(result.get("themes"), "themes"),
            "symbols": _string_list(result.get("symbols"), "symbols"),
            "confidence_score": _finite_confidence(result.get("confidence_score")),
            "narration_aligned": narration_aligned,
            "model": self._model,
            "model_version": self._model_version,
        }

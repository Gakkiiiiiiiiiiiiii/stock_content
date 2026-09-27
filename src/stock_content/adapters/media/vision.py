from __future__ import annotations

import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


def _ticker_list(value: Any) -> list[str]:
    tickers = _string_list(value, "observed_tickers")
    if any(not re.fullmatch(r"\d{6}", item) for item in tickers):
        raise ValueError("vision observed_tickers must contain six-digit tickers")
    return tickers


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
        responses_api = urlparse(self._url).path.rstrip("/").endswith("/responses")
        if self._model == "gpt-6-sol" and not responses_api:
            raise RuntimeError("gpt-6-sol vision requires a Responses API endpoint")
        image = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        prompt = (
            "分析金融视频帧，只输出 JSON 对象，严格包含："
            "visual_summary(非空字符串),labels(至少一个非空标签),themes(字符串数组),"
            "symbols(字符串数组),observed_entities(字符串数组),observed_tickers(六码股票代码字符串数组),"
            "confidence_score(0到1有限数字),narration_aligned(布尔值)。"
            "只描述画面，不得创造金融事实。口播上下文：" + transcript_context[:3000]
        )
        if responses_api:
            body = {
                "model": self._model,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + image},
                ]}],
                "reasoning": {"effort": os.getenv("CONTENT_VISION_REASONING_EFFORT", "medium")},
                "text": {"format": {"type": "json_object"}},
            }
        else:
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
        if responses_api:
            if payload.get("status") != "completed":
                raise ValueError("vision response was not completed")
            returned_model = str(payload.get("model") or "")
            if not returned_model or (self._model == "gpt-6-sol" and returned_model != self._model):
                raise ValueError("vision response identity does not match gpt-6-sol")
        else:
            returned_model = self._model
        try:
            if responses_api:
                content = "".join(
                    str(part.get("text") or "")
                    for item in payload.get("output") or ()
                    if isinstance(item, dict) and item.get("type") == "message"
                    for part in item.get("content") or ()
                    if isinstance(part, dict) and part.get("type") == "output_text"
                )
            else:
                content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            result = json.loads(content or "{}")
        except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("vision endpoint returned invalid JSON") from exc
        return self._validate(result, model=returned_model)

    def _validate(self, result: Any, *, model: str | None = None) -> dict:
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
            "observed_entities": _string_list(result.get("observed_entities"), "observed_entities"),
            "observed_tickers": _ticker_list(result.get("observed_tickers")),
            "confidence_score": _finite_confidence(result.get("confidence_score")),
            "narration_aligned": narration_aligned,
            "model": model or self._model,
            "model_version": self._model_version,
        }

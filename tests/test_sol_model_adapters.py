from __future__ import annotations

import json as jsonlib

import pytest

from stock_content.adapters.http.model_client import ContentModelClient
from stock_content.adapters.media.vision import HttpVisionAnalyzer


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _completed(text: str, *, model: str = "gpt-6-sol") -> dict:
    return {
        "status": "completed",
        "model": model,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
    }


def test_sol_text_extraction_uses_responses_without_temperature(monkeypatch):
    requests = []

    def post(url, *, json, headers, timeout):
        requests.append((url, json))
        return _Response(_completed('{"theses":[]}'))

    monkeypatch.setattr("stock_content.adapters.http.model_client.httpx.post", post)
    client = ContentModelClient("https://api.openai.com/v1/responses", "gpt-6-sol", "key")
    result = client.complete(
        prompt="转录片段 11", system="Return JSON only", temperature=0.0,
        response_format={"type": "json_object"},
    )

    assert result["model"] == "gpt-6-sol"
    assert result["content"] == '{"theses":[]}'
    body = requests[0][1]
    assert body["input"][1]["content"] == "转录片段 11"
    assert body["text"] == {"format": {"type": "json_object"}}
    assert body["reasoning"] == {"effort": "medium"}
    assert "temperature" not in body


def test_sol_text_extraction_rejects_replay_identity_and_chat_endpoint(monkeypatch):
    monkeypatch.setattr(
        "stock_content.adapters.http.model_client.httpx.post",
        lambda *args, **kwargs: _Response(_completed("{}", model="gpt-5-reviewed-replay")),
    )
    client = ContentModelClient("https://api.openai.com/v1/responses", "gpt-6-sol", "key")
    with pytest.raises(ValueError, match="identity"):
        client.complete(prompt="x", system="JSON", temperature=0.0)
    with pytest.raises(RuntimeError, match="Responses API"):
        ContentModelClient("https://api.openai.com/v1/chat/completions", "gpt-6-sol", "key").complete(
            prompt="x", system="JSON", temperature=0.0,
        )


def test_sol_vision_uses_frame_pixels_and_returns_runtime_identity(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test-image")
    requests = []
    observation = {
        "visual_summary": "图表显示需求",
        "labels": ["chart"],
        "themes": ["需求"],
        "symbols": [],
        "observed_entities": ["示例公司"],
        "observed_tickers": [],
        "confidence_score": 0.8,
        "narration_aligned": True,
    }

    def post(url, *, json, headers, timeout):
        requests.append(json)
        return _Response(_completed(jsonlib.dumps(observation, ensure_ascii=False)))

    monkeypatch.setattr("stock_content.adapters.media.vision.httpx.post", post)
    result = HttpVisionAnalyzer("https://api.openai.com/v1/responses", "gpt-6-sol", "key").analyze(
        str(frame), "当前转录窗口"
    )

    assert result["model"] == "gpt-6-sol"
    assert result["observed_entities"] == ["示例公司"]
    content = requests[0]["input"][0]["content"]
    assert content[0]["text"].endswith("当前转录窗口")
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "temperature" not in requests[0]


def test_sol_vision_rejects_wrong_runtime_model(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test-image")
    monkeypatch.setattr(
        "stock_content.adapters.media.vision.httpx.post",
        lambda *args, **kwargs: _Response(_completed("{}", model="gpt-5-reviewed-replay")),
    )
    with pytest.raises(ValueError, match="identity"):
        HttpVisionAnalyzer("https://api.openai.com/v1/responses", "gpt-6-sol", "key").analyze(
            str(frame), "窗口"
        )

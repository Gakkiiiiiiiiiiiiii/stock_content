"""The production model adapter preserves runtime identity on Responses API calls."""

from __future__ import annotations

import httpx
import pytest

from stock_content.adapters.http.model_client import ContentModelClient
from stock_content.api.dependencies import pipeline_config_from_env


def test_responses_api_uses_text_format_and_runtime_model(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "status": "completed",
                "model": "gpt-6-sol",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "content": [{"type": "output_text", "text": '{"boundaries":[]}' }]},
                ],
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    client = ContentModelClient(base_url="https://example.test/v1/responses", model="gpt-6-sol")
    result = client.complete(
        prompt="Locate topic changes", system="Return JSON", temperature=0.0,
        response_format={"type": "json_object"},
    )

    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["json"] == {
        "model": "gpt-6-sol",
        "input": [
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": "Locate topic changes"},
        ],
        "reasoning": {"effort": "medium"},
        "store": False,
        "text": {"format": {"type": "json_object"}},
    }
    assert result["content"] == '{"boundaries":[]}'
    assert result["model"] == "gpt-6-sol"


def test_responses_api_rejects_incomplete_result(monkeypatch):
    def post(url, **kwargs):
        return httpx.Response(
            200, request=httpx.Request("POST", url),
            json={"status": "incomplete", "model": "gpt-6-sol", "output": []},
        )

    monkeypatch.setattr(httpx, "post", post)
    client = ContentModelClient(base_url="https://example.test/v1/responses", model="gpt-6-sol")
    with pytest.raises(RuntimeError, match="did not complete"):
        client.complete(prompt="test", system="test", temperature=0.0)


def test_pipeline_model_identity_falls_back_to_shared_config(monkeypatch):
    monkeypatch.setenv("CONTENT_MODEL_NAME", "gpt-6-sol")
    monkeypatch.delenv("CONTENT_SEGMENTATION_MODEL", raising=False)
    monkeypatch.delenv("CONTENT_EXTRACTION_MODEL", raising=False)

    config = pipeline_config_from_env()
    assert config["segmentation_model"] == "gpt-6-sol"
    assert config["extraction_model"] == "gpt-6-sol"

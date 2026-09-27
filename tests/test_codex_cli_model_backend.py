from __future__ import annotations

import json
import subprocess

import pytest

from stock_content.adapters.codex_cli import CodexCliRunner
from stock_content.adapters.http.model_client import ContentModelClient
from stock_content.adapters.media.vision import CodexCliVisionAnalyzer
from stock_content.api.dependencies import pipeline_config_from_env


def _events(payload: dict, *, with_tool: bool = False) -> str:
    events = [{"type": "thread.started", "thread_id": "test"}, {"type": "turn.started"}]
    if with_tool:
        events.append({"type": "item.completed", "item": {"type": "command_execution"}})
    events.extend((
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(payload)}},
        {"type": "turn.completed"},
    ))
    return "\n".join(json.dumps(event) for event in events)


def test_codex_backend_uses_account_auth_and_never_needs_model_url(monkeypatch):
    monkeypatch.setenv("CONTENT_MODEL_BACKEND", "codex_cli")
    monkeypatch.setenv("CONTENT_CODEX_CLI", "C:/codex/codex.exe")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv("CODEX_API_KEY", "secret-codex-key")
    monkeypatch.delenv("CONTENT_MODEL_URL", raising=False)
    monkeypatch.setattr("stock_content.adapters.http.model_client.shutil.which", lambda path: path)
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, _events({"boundaries": []}), "")

    monkeypatch.setattr("stock_content.adapters.codex_cli.subprocess.run", fake_run)
    client = ContentModelClient()
    assert client.available()
    result = client.complete(prompt="两个主题", system="segment", temperature=0)
    assert result["model"] == "gpt-6-sol"
    assert result["model_identity_source"] == "accepted_codex_cli_invocation_not_response_metadata"
    assert result["provider"] == "codex-cli-chatgpt-account"
    assert "--model" in observed["command"] and "gpt-6-sol" in observed["command"]
    assert "--sandbox" in observed["command"] and "read-only" in observed["command"]
    assert "OPENAI_API_KEY" not in observed["env"]
    assert "CODEX_API_KEY" not in observed["env"]
    assert observed["encoding"] == "utf-8"
    assert "两个主题" in observed["input"]


@pytest.mark.parametrize("with_tool", [True, False])
def test_codex_backend_fails_closed_on_tool_use_or_cli_error(monkeypatch, with_tool):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0 if with_tool else 1, _events({}, with_tool=with_tool), "")

    monkeypatch.setattr("stock_content.adapters.codex_cli.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="attempted a tool|model call failed"):
        CodexCliRunner(executable="codex.exe").run(system="system", prompt="prompt")


def test_codex_vision_uses_same_model_and_validates_frame(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    observed = {}

    class FakeRunner:
        def run(self, **kwargs):
            observed.update(kwargs)
            return {
                "content": json.dumps({
                    "visual_summary": "图上出现公司名称",
                    "labels": ["chart"], "themes": ["earnings"], "symbols": ["600519"],
                    "confidence_score": 0.8, "narration_aligned": True,
                }),
                "model_identity_source": "accepted_codex_cli_invocation_not_response_metadata",
            }

    result = CodexCliVisionAnalyzer(FakeRunner()).analyze(str(frame), "公司盈利增长")
    assert observed["image_path"] == str(frame)
    assert "公司盈利增长" in observed["prompt"]
    assert result["model"] == "gpt-6-sol"
    assert result["symbols"] == ["600519"]


def test_codex_mode_defaults_all_interpretation_models_to_sol(monkeypatch):
    monkeypatch.setenv("CONTENT_MODEL_BACKEND", "codex_cli")
    for key in ("CONTENT_MODEL_NAME", "CONTENT_SEGMENTATION_MODEL", "CONTENT_EXTRACTION_MODEL", "CONTENT_VISION_MODEL"):
        monkeypatch.delenv(key, raising=False)
    config = pipeline_config_from_env()
    assert config["segmentation_model"] == "gpt-6-sol"
    assert config["segmentation_prompt_version"] == "semantic-segmentation.prompt.v4.codex-cli"
    assert config["extraction_model"] == "gpt-6-sol"
    assert config["extraction_prompt_version"] == "atomic-claim-extraction.prompt.v2.codex-cli"
    assert config["vision_model"] == "gpt-6-sol"
    assert config["vision_prompt_version"] == "vision-context.prompt.v2.codex-cli"
    assert config["vision_model_version"] == "unreported-by-codex-cli"
    assert config["model_backend"] == "codex_cli"
    assert config["model_identity_source"] == "accepted_codex_cli_invocation_not_response_metadata"


def test_codex_mode_rejects_any_other_configured_model(monkeypatch):
    monkeypatch.setenv("CONTENT_MODEL_BACKEND", "codex_cli")
    monkeypatch.setenv("CONTENT_VISION_MODEL", "different-model")
    with pytest.raises(ValueError, match="CONTENT_VISION_MODEL must be gpt-6-sol"):
        pipeline_config_from_env()

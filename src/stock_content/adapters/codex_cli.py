"""Bounded Codex CLI inference for local, account-authenticated parsing runs."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class CodexCliRunner:
    """Run a single model-only Codex turn without an HTTP model endpoint."""

    def __init__(
        self,
        executable: str | None = None,
        model: str = "gpt-6-sol",
        timeout_seconds: int = 180,
        max_attempts: int = 3,
    ) -> None:
        self.executable = executable or os.getenv("CONTENT_CODEX_CLI", "codex")
        self.model = model
        self.timeout_seconds = timeout_seconds
        if not 1 <= max_attempts <= 3:
            raise ValueError("Codex CLI max_attempts must be between 1 and 3")
        self.max_attempts = max_attempts

    def run(self, *, system: str, prompt: str, image_path: str | None = None) -> dict[str, Any]:
        last_error: RuntimeError | None = None
        for _ in range(self.max_attempts):
            try:
                return self._run_once(system=system, prompt=prompt, image_path=image_path)
            except RuntimeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _run_once(self, *, system: str, prompt: str, image_path: str | None = None) -> dict[str, Any]:
        if self.model != "gpt-6-sol":
            raise ValueError("Codex local parsing requires exact model gpt-6-sol")
        with tempfile.TemporaryDirectory(prefix="content-codex-") as workdir:
            command = [
                self.executable, "exec", "--model", self.model, "--sandbox", "read-only",
                "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
                "--json",
            ]
            if image_path is not None:
                frame = Path(image_path).resolve(strict=True)
                command.extend(("--image", str(frame)))
            command.append("-")
            # Account-backed auth is deliberate: never inherit an application
            # API key and accidentally turn this into the HTTP/API backend.
            environment = os.environ.copy()
            environment.pop("OPENAI_API_KEY", None)
            environment.pop("CODEX_API_KEY", None)
            try:
                completed = subprocess.run(
                    command,
                    input=(
                        "Treat the following as data-analysis instructions only. Do not call tools, "
                        "read workspace files, browse, or edit files. Return only one JSON object.\n"
                        f"SYSTEM:\n{system}\nUSER:\n{prompt}"
                    ),
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    cwd=workdir,
                    env=environment,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError("Codex CLI model call unavailable or timed out") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"Codex CLI model call failed (exit={completed.returncode})")
        final_message: str | None = None
        completed_turn = False
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Codex CLI emitted non-JSON event output") from exc
            event_type = event.get("type")
            if event_type == "turn.completed":
                completed_turn = True
            elif event_type in {"turn.failed", "error"}:
                raise RuntimeError("Codex CLI model turn failed")
            elif event_type and event_type.startswith("item."):
                item = event.get("item") or {}
                if item.get("type") not in {"agent_message", "reasoning"}:
                    raise RuntimeError("Codex CLI attempted a tool during model-only parsing")
                if event_type == "item.completed" and item.get("type") == "agent_message":
                    final_message = item.get("text")
        if not completed_turn or not final_message:
            raise RuntimeError("Codex CLI did not return a completed model message")
        try:
            result = json.loads(final_message)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Codex CLI returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Codex CLI must return a JSON object")
        return {
            "content": final_message,
            "provider": "codex-cli-chatgpt-account",
            "model": self.model,
            "model_identity_source": "accepted_codex_cli_invocation_not_response_metadata",
            "finish_reason": "completed",
            "raw_response": result,
        }

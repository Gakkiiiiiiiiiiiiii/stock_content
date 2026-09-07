"""Private filesystem deletion adapter with traversal and symlink protection."""
from __future__ import annotations

from pathlib import Path


class SafeFilesystemArtifactDeleter:
    """Delete only artifact-id mapped files under a configured private root."""

    def __init__(self, root: Path, artifact_paths: dict[str, str]) -> None:
        self._root = root.resolve()
        self._paths = dict(artifact_paths)

    def delete(self, artifact_id: str, *, idempotency_key: str) -> None:
        if not artifact_id or not idempotency_key.startswith("ts_"):
            raise ValueError("invalid retention deletion identity")
        relative = self._paths.get(artifact_id)
        if not relative:
            raise KeyError("artifact bytes are not configured for retention deletion")
        candidate = (self._root / relative).resolve()
        if candidate == self._root or self._root not in candidate.parents:
            raise ValueError("retention path escapes configured root")
        # Missing bytes is an idempotent completed deletion; no path is ever
        # exposed in persistent audit state.
        if candidate.exists():
            if candidate.is_dir():
                raise ValueError("retention target must be a file")
            candidate.unlink()


__all__ = ["SafeFilesystemArtifactDeleter"]

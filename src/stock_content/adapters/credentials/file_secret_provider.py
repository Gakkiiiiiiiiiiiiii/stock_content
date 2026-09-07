"""Allowlisted, read-only file credentials for media workers.

The provider deliberately returns a path, not the credential content.  It is
therefore safe to retain in adapter wiring and impossible to accidentally put
a cookie value into a task, artifact, or log record.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


class SecretUnavailable(RuntimeError):
    """A safe-to-log credential failure."""

    code = "SOURCE_SESSION_EXPIRED"

    def __init__(self) -> None:
        super().__init__(self.code)


class FileSecretProvider:
    """Resolve only configured references to a mode-0400 regular file."""

    def __init__(self, references: dict[str, str | os.PathLike[str]]) -> None:
        self._references = {name: Path(path).resolve() for name, path in references.items()}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(references=<redacted>)"

    def resolve(self, credential_ref: str) -> Path:
        path = self._references.get(credential_ref)
        if path is None or not path.is_file() or path.is_symlink():
            raise SecretUnavailable()
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as exc:
            raise SecretUnavailable() from exc
        # On POSIX, reject group/world-readable cookie files.  Windows does
        # not expose POSIX mode bits reliably, so its mounted-secret ACL is
        # the controlling permission boundary.
        if os.name != "nt" and mode != 0o400:
            raise SecretUnavailable()
        return path

    def resolve_hash(self, credential_ref_hash: str | None) -> Path:
        """Resolve a persisted one-way reference without recovering its name.

        The worker's allowlist is the only source of candidate references.
        This lets a queued Xiaoe task use its configured storage state while
        keeping the submitted reference out of durable task data.
        """
        if not credential_ref_hash:
            raise SecretUnavailable()
        for reference in self._references:
            digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()
            if digest == credential_ref_hash:
                return self.resolve(reference)
        raise SecretUnavailable()

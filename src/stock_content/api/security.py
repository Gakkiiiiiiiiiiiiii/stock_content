"""Small, fail-closed service authentication boundary for private APIs."""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Iterable
from pathlib import Path

_CALLER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class ServiceAuthError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ServiceAuthorizer:
    """Accept current and previous file-backed bearer tokens during rotation.

    The values are read at request time: secret mounts can rotate without an
    API restart, and no token is retained in application state or logs.
    """

    def __init__(self, token_files: Iterable[str | Path] = (), allowed_callers: Iterable[str] = ("stock_agent",)):
        self._token_files = tuple(Path(item) for item in token_files if str(item).strip())
        self._allowed_callers = frozenset(item for item in allowed_callers if _CALLER.fullmatch(item))

    @classmethod
    def from_environment(cls) -> "ServiceAuthorizer":
        callers = tuple(
            item.strip()
            for item in os.getenv("CONTENT_SERVICE_ALLOWED_CALLERS", "stock_agent").split(",")
            if item.strip()
        )
        return cls(
            (os.getenv("CONTENT_SERVICE_API_KEY_FILE", ""), os.getenv("CONTENT_SERVICE_API_KEY_PREVIOUS_FILE", "")),
            callers,
        )

    def configured(self) -> bool:
        return bool(self._tokens()) and bool(self._allowed_callers)

    def _tokens(self) -> tuple[str, ...]:
        tokens: list[str] = []
        for path in self._token_files:
            try:
                if not path.is_file() or path.is_symlink():
                    continue
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                tokens.append(value)
        return tuple(tokens)

    def authorize(self, authorization: str | None, caller: str | None, *, required_caller: str | None = None) -> str:
        tokens = self._tokens()
        if not tokens:
            raise ServiceAuthError(503, "AUTH_NOT_READY", "service authentication is not configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise ServiceAuthError(401, "AUTH_REQUIRED", "Bearer authentication is required")
        supplied = authorization[7:]
        if not supplied or not any(hmac.compare_digest(supplied, token) for token in tokens):
            raise ServiceAuthError(401, "AUTH_INVALID", "Bearer authentication failed")
        if not caller or not _CALLER.fullmatch(caller) or caller not in self._allowed_callers:
            raise ServiceAuthError(403, "CALLER_FORBIDDEN", "caller service is not allowed")
        if required_caller is not None and caller != required_caller:
            raise ServiceAuthError(403, "CALLER_FORBIDDEN", "caller service is not allowed")
        return caller


__all__ = ["ServiceAuthError", "ServiceAuthorizer"]

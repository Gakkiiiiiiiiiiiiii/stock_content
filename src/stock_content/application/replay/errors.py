"""Stable errors raised while validating a replay boundary."""
from __future__ import annotations

from typing import Any


class ReplayIntegrityError(ValueError):
    """Stable, structured failure at the replay integrity boundary."""

    def __init__(self, code: str, detail: str, **fields: Any) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.fields = fields

    def to_dict(self, *, content_snapshot_id: str | None = None) -> dict[str, Any]:
        payload = {"error": self.code, "detail": self.detail}
        payload.update(self.fields)
        if content_snapshot_id is not None:
            payload.setdefault("content_snapshot_id", content_snapshot_id)
        return payload


__all__ = ["ReplayIntegrityError"]

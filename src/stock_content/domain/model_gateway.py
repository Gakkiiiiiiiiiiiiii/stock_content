from __future__ import annotations

from typing import Any, Protocol


class StructuredModelGateway(Protocol):
    """Domain-facing port for the structured-knowledge model."""

    def available(self) -> bool: ...

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

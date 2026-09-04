from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Mapping


class MetricsRegistry:
    """Minimal in-process counter registry suitable for adapters/exporters."""

    def __init__(self) -> None:
        self._values: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._lock = Lock()

    def increment(self, name: str, value: float = 1, labels: Mapping[str, object] | None = None) -> float:
        labels = labels or {}
        forbidden = set(labels) - ALLOWED_LABELS
        if forbidden:
            raise ValueError(f"unbounded metric labels are not allowed: {sorted(forbidden)}")
        key = (name, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
        with self._lock:
            self._values[key] += value
            return self._values[key]

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                name if not labels else f"{name}{dict(labels)}": value
                for (name, labels), value in self._values.items()
            }


metrics = MetricsRegistry()

# Labels intentionally exclude claim/snapshot/source IDs (unbounded values).
ALLOWED_LABELS = frozenset({"source_type", "result", "reason_code", "state", "contract", "version", "event_type"})

__all__ = ["MetricsRegistry", "metrics"]

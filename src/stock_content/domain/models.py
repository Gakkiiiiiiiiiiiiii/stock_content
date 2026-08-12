from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentTask:
    task_id: str
    source_type: str
    source_ref: str
    status: str = "PENDING"
    stage: str = "queued"
    result: dict[str, Any] = field(default_factory=dict)

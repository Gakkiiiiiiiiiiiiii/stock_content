from __future__ import annotations

from typing import Protocol


class KnowledgeRepository(Protocol):
    def search(self, query: str, filters: dict, limit: int) -> list[dict]: ...


class ContentTaskRepository(Protocol):
    def get(self, task_id: str) -> dict | None: ...

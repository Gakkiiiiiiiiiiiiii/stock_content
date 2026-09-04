from __future__ import annotations

from typing import Protocol

from stock_content.domain.publication_run import ContentPublicationRun


class PublicationRepository(Protocol):
    def get_by_identity(self, snapshot_id: str, query_hash: str, policy: str) -> ContentPublicationRun | None: ...
    def save(self, run: ContentPublicationRun) -> ContentPublicationRun: ...

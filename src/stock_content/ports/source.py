"""Ports for deferred source resolution and materialization."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from stock_content.domain.source_materialization import ContentIngestionCommand, SourceMaterialization


class SourceResolver(Protocol):
    def resolve(self, request: ContentIngestionCommand) -> SourceMaterialization: ...


class SourceMaterializer(Protocol):
    def materialize(self, materialization: SourceMaterialization, target_dir: Path) -> object: ...


class CredentialProvider(Protocol):
    def resolve(self, credential_ref: str) -> object: ...


__all__ = ["CredentialProvider", "SourceMaterializer", "SourceResolver"]

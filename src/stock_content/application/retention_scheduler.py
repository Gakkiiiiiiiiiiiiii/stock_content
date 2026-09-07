"""Leased, idempotent retention sweep wired into the existing worker."""
from __future__ import annotations

from pathlib import Path

from stock_content.adapters.retention.safe_filesystem import SafeFilesystemArtifactDeleter
from stock_content.application.retention_service import RetentionService


class RetentionScheduler:
    def __init__(self, service: RetentionService, candidates, *, private_root: Path) -> None:
        self._service = service
        self._candidates = candidates
        self._root = private_root.resolve()

    def sweep(self, *, dry_run: bool = False) -> tuple[str, ...]:
        """Perform one bounded sweep; durable tombstones provide retry state."""
        if not self._root.is_dir():
            raise RuntimeError("RETENTION_PRIVATE_ROOT_NOT_READY")
        deleter = SafeFilesystemArtifactDeleter(self._root, self._candidates.private_locator_map())
        return tuple(
            self._service.execute(candidate, deleter, dry_run=dry_run).action
            for candidate in self._candidates.candidates()
        )


__all__ = ["RetentionScheduler"]

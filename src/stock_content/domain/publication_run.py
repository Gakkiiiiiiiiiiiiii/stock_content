"""Publication state machine and deterministic publication identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from stock_content.domain.artifacts import canonical_json


class PublicationState(StrEnum):
    ASSEMBLING = "ASSEMBLING"
    PROJECTING = "PROJECTING"
    SEALING = "SEALING"
    READY = "READY"
    PUBLISHING = "PUBLISHING"
    PUBLISHED = "PUBLISHED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"


_TRANSITIONS = {
    PublicationState.ASSEMBLING: {
        PublicationState.PROJECTING,
        PublicationState.FAILED_RETRYABLE,
        PublicationState.FAILED_TERMINAL,
    },
    PublicationState.PROJECTING: {
        PublicationState.SEALING,
        PublicationState.FAILED_RETRYABLE,
        PublicationState.FAILED_TERMINAL,
    },
    PublicationState.SEALING: {
        PublicationState.READY,
        PublicationState.FAILED_RETRYABLE,
        PublicationState.FAILED_TERMINAL,
    },
    PublicationState.READY: {PublicationState.PUBLISHING, PublicationState.FAILED_RETRYABLE},
    PublicationState.PUBLISHING: {PublicationState.PUBLISHED, PublicationState.FAILED_RETRYABLE},
    PublicationState.FAILED_RETRYABLE: {
        PublicationState.ASSEMBLING,
        PublicationState.PROJECTING,
        PublicationState.SEALING,
        PublicationState.PUBLISHING,
        PublicationState.FAILED_TERMINAL,
    },
    PublicationState.PUBLISHED: set(),
    PublicationState.FAILED_TERMINAL: set(),
}


@dataclass(frozen=True)
class ContentPublicationRun:
    content_snapshot_id: str
    query_hash: str
    signal_policy_version: str
    state: PublicationState = PublicationState.ASSEMBLING
    manifest_hash: str | None = None
    version: int = 1
    publication_run_id: str = ""

    def __post_init__(self) -> None:
        for name in ("content_snapshot_id", "query_hash", "signal_policy_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")
        identity = {
            "content_snapshot_id": self.content_snapshot_id,
            "query_hash": self.query_hash,
            "signal_policy_version": self.signal_policy_version,
        }
        expected = "pub_" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:32]
        if self.publication_run_id and self.publication_run_id != expected:
            raise ValueError("publication_run_id does not match identity")
        object.__setattr__(self, "publication_run_id", self.publication_run_id or expected)
        object.__setattr__(self, "state", PublicationState(self.state))
        if self.version < 1:
            raise ValueError("version must be positive")

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.content_snapshot_id, self.query_hash, self.signal_policy_version

    def transition(self, state: PublicationState | str, *, manifest_hash: str | None = None) -> "ContentPublicationRun":
        target = PublicationState(state)
        if target != self.state and target not in _TRANSITIONS[self.state]:
            raise ValueError(f"invalid publication transition {self.state}->{target}")
        if target == PublicationState.READY and not (manifest_hash or self.manifest_hash):
            raise ValueError("READY publication requires manifest_hash")
        return replace(
            self,
            state=target,
            manifest_hash=manifest_hash or self.manifest_hash,
            version=self.version + (target != self.state),
        )


def manifest_hash(manifest: Any) -> str:
    return hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()


__all__ = ["ContentPublicationRun", "PublicationState", "manifest_hash"]

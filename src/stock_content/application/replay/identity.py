"""Stable namespaces for immutable artifacts derived by migration replay."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from stock_content.domain.artifacts import canonical_json


def migration_derivation_namespace(source_snapshot_id: str, pipeline_version: str) -> str:
    """Return the deterministic namespace for one source/pipeline migration.

    Replays of the same migration must address the same derived immutable
    rows, while a different target pipeline must never overwrite their IDs.
    """
    if not source_snapshot_id or not pipeline_version:
        raise ValueError("migration derivation namespace requires snapshot and pipeline version")
    payload = {
        "kind": "migration-derived.v1",
        "source_snapshot_id": source_snapshot_id,
        "pipeline_version": pipeline_version,
    }
    return "migration-" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:32]


def canonical_migration_pipeline_version(pipeline_version: str | None) -> str:
    value = str(pipeline_version or "").strip()
    if not value:
        raise ValueError("MIGRATION_REPLAY requires pipeline_version")
    return value


def migration_replay_request_identity(
    source_snapshot_id: str,
    pipeline_version: str | None,
    overrides: Mapping[str, Any] | None,
    *,
    runtime_option_keys: Iterable[str] = (),
) -> str:
    """Hash only effective result inputs for one migration replay request."""
    normalized_pipeline = canonical_migration_pipeline_version(pipeline_version)
    runtime_keys = {str(key) for key in runtime_option_keys}
    effective_overrides = {
        str(key): value
        for key, value in (overrides or {}).items()
        if str(key) not in runtime_keys
    }
    payload = {
        "kind": "migration-replay-request.v1",
        "source_snapshot_id": source_snapshot_id,
        "mode": "MIGRATION_REPLAY",
        "pipeline_version": normalized_pipeline,
        "overrides": effective_overrides,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def migration_replay_idempotency_key(request_identity: str) -> str:
    return "replay-migration-" + request_identity


__all__ = [
    "canonical_migration_pipeline_version",
    "migration_derivation_namespace",
    "migration_replay_idempotency_key",
    "migration_replay_request_identity",
]

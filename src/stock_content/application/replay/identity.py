"""Stable namespaces for immutable artifacts derived by migration replay."""

from __future__ import annotations

import hashlib

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


__all__ = ["migration_derivation_namespace"]

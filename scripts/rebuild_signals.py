"""Rebuild the durable signal outbox from PostgreSQL snapshot truth.

The outbox is a projection, not the source of truth.  This command is safe to
run after an outbox purge: deterministic signal ids make repeated rebuilds
idempotent and conflicting payloads are rejected by the repository.
"""
from __future__ import annotations

import argparse

from sqlalchemy import select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ContentArtifactRow,
    ContentSnapshotRow,
    ContentSourceHeadRow,
)
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.adapters.postgres.repositories.signal_outbox_repository import SignalOutboxRepository
from stock_content.adapters.postgres.repositories.snapshot_repository import _row_to_snapshot
from stock_content.application.signal_service import SignalService
from stock_content.domain.artifacts import deserialize_artifact


def rebuild_signals(
    database_url: str | None = None,
    *,
    snapshot_policy: str = "latest-verified",
    snapshot_id: str | None = None,
) -> dict[str, int]:
    if snapshot_id:
        snapshot_policy = snapshot_id
    if snapshot_policy not in {"latest", "latest-verified"} and not snapshot_id:
        raise ValueError("snapshot_policy must be latest or latest-verified")
    database = Database(database_url)
    database.create_schema()
    claims_repo = SqlClaimRepository(database.session_factory)
    outbox = SignalOutboxRepository(database.session_factory)
    service = SignalService()
    snapshot_ids: list[str] = []
    with database.session_factory() as session:
        heads = session.scalars(select(ContentSourceHeadRow)).all() if not snapshot_id else []
        for head in heads:
            chosen = (
                head.latest_verified_snapshot_id
                if snapshot_policy == "latest-verified" and head.latest_verified_snapshot_id
                else head.latest_snapshot_id
            )
            if chosen and chosen not in snapshot_ids:
                snapshot_ids.append(chosen)
    rebuilt = 0
    suppressed = 0
    if snapshot_id:
        snapshot_ids = [snapshot_id]
    for snapshot_id in snapshot_ids:
        with database.session_factory() as session:
            row = session.get(ContentSnapshotRow, snapshot_id)
            if row is None:
                continue
            snapshot = _row_to_snapshot(row)
            claims_artifact_id = str((snapshot.artifact_ids or {}).get("claims") or "")
            verification_artifact_id = str((snapshot.artifact_ids or {}).get("verification") or "")
            claims_artifact_row = session.get(ContentArtifactRow, claims_artifact_id)
            verification_row = session.get(ContentArtifactRow, verification_artifact_id)
            if claims_artifact_row is None or verification_row is None:
                continue
            claims_artifact = deserialize_artifact(dict(claims_artifact_row.payload or {}))
            verification_artifact = deserialize_artifact(dict(verification_row.payload or {}))
            claim_ids = list(getattr(claims_artifact, "claims", ()) or ())
        claims = [claim for claim_id in claim_ids if (claim := claims_repo.get(str(claim_id))) is not None]
        signals = service.rebuild_signals(snapshot, claims, verification_artifact)
        for signal in signals:
            if signal.get("signal_status") == "SUPPRESSED":
                suppressed += 1
                continue
            outbox.enqueue(signal)
            rebuilt += 1
    return {"snapshots": len(snapshot_ids), "rebuilt": rebuilt, "suppressed": suppressed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-postgres", action="store_true", help="read PostgreSQL/SQL database truth")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--snapshot-policy", choices=("latest", "latest-verified"), default="latest-verified")
    parser.add_argument("--snapshot-id", default=None, help="rebuild one persisted snapshot")
    args = parser.parse_args()
    if not args.from_postgres:
        parser.error("--from-postgres is required")
    print(rebuild_signals(args.database_url, snapshot_policy=args.snapshot_policy, snapshot_id=args.snapshot_id))


if __name__ == "__main__":
    main()

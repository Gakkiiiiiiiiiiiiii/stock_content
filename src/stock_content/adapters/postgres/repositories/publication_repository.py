from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ContentPublicationManifestRow,
    ContentPublicationRunRow,
    ContentSealedSignalRow,
)
from stock_content.domain.publication_run import ContentPublicationRun, PublicationState


class PublicationRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def get_by_identity(self, snapshot_id: str, query_hash: str, policy: str) -> ContentPublicationRun | None:
        with self._sessions() as session:
            row = self.get_by_identity_in_session(session, snapshot_id, query_hash, policy)
        return _to_domain(row) if row else None

    def get_by_identity_in_session(self, session, snapshot_id: str, query_hash: str, policy: str):
        row = session.scalar(
            select(ContentPublicationRunRow)
            .where(
                ContentPublicationRunRow.content_snapshot_id == snapshot_id,
                ContentPublicationRunRow.query_hash == query_hash,
                ContentPublicationRunRow.signal_policy_version == policy,
            )
            .with_for_update()
        )
        return _to_domain(row) if row else None

    def is_ready(self, snapshot_id: str) -> bool:
        with self._sessions() as session:
            return session.scalar(select(ContentPublicationRunRow.publication_run_id).where(
                ContentPublicationRunRow.content_snapshot_id == snapshot_id,
                ContentPublicationRunRow.state.in_(("READY", "PUBLISHING", "PUBLISHED")),
            ).limit(1)) is not None

    def save(self, run: ContentPublicationRun) -> ContentPublicationRun:
        with self._sessions.begin() as session:
            self.save_in_session(session, run)
        return run

    def save_in_session(self, session, run: ContentPublicationRun) -> ContentPublicationRun:
        row = session.get(ContentPublicationRunRow, run.publication_run_id)
        if row is None:
            session.add(
                ContentPublicationRunRow(
                    publication_run_id=run.publication_run_id,
                    content_snapshot_id=run.content_snapshot_id,
                    query_hash=run.query_hash,
                    signal_policy_version=run.signal_policy_version,
                    state=run.state.value,
                    manifest_hash=run.manifest_hash,
                    version=run.version,
                )
            )
        else:
            if (
                row.content_snapshot_id != run.content_snapshot_id
                or row.query_hash != run.query_hash
                or row.signal_policy_version != run.signal_policy_version
            ):
                raise ValueError("publication run id conflicts with immutable publication identity")
            if (
                row.manifest_hash is not None
                and run.manifest_hash is not None
                and row.manifest_hash != run.manifest_hash
            ):
                raise ValueError("publication run already contains a different sealed manifest")
            row.state, row.manifest_hash, row.version = run.state.value, run.manifest_hash, run.version
        return run

    def save_manifest_in_session(
        self, session, run: ContentPublicationRun, manifest: dict[str, Any], sealed_hash: str
    ) -> None:
        row = session.get(ContentPublicationManifestRow, run.publication_run_id)
        if row is None:
            session.add(ContentPublicationManifestRow(
                publication_run_id=run.publication_run_id,
                content_snapshot_id=run.content_snapshot_id,
                query_hash=run.query_hash,
                signal_policy_version=run.signal_policy_version,
                manifest_hash=sealed_hash,
                manifest=dict(manifest),
                created_at=datetime.now(UTC),
            ))
            return
        if row.manifest_hash != sealed_hash or dict(row.manifest or {}) != dict(manifest):
            raise ValueError("publication manifest already contains different sealed content")

    def save_sealed_signals_in_session(
        self, session, run: ContentPublicationRun | str, signals: tuple[dict[str, Any], ...]
    ) -> None:
        if isinstance(run, str):
            run_row = session.get(ContentPublicationRunRow, run)
            if run_row is None:
                raise ValueError("publication run is required before sealing signals")
            run_id, snapshot_id = run_row.publication_run_id, run_row.content_snapshot_id
        else:
            run_id, snapshot_id = run.publication_run_id, run.content_snapshot_id
        for payload in signals:
            signal_id = str(payload.get("signal_id") or "")
            if not signal_id:
                raise ValueError("sealed signal requires signal_id")
            sealed_signal_id = "sealed-" + hashlib.sha256(
                f"{run_id}:{signal_id}".encode()
            ).hexdigest()[:32]
            row = session.get(ContentSealedSignalRow, sealed_signal_id)
            if row is None:
                session.add(ContentSealedSignalRow(
                    sealed_signal_id=sealed_signal_id,
                    publication_run_id=run_id,
                    signal_id=signal_id,
                    content_snapshot_id=snapshot_id,
                    claim_id=payload.get("claim_id"),
                    schema_version=str(payload.get("signal_schema_version") or payload.get("schema_version") or ""),
                    payload=dict(payload),
                    created_at=datetime.now(UTC),
                ))
                continue
            if dict(row.payload or {}) != dict(payload):
                raise ValueError("sealed signal already contains different payload")

    def get_manifest_in_session(self, session, publication_run_id: str) -> dict[str, Any] | None:
        row = session.get(ContentPublicationManifestRow, publication_run_id)
        return dict(row.manifest or {}) if row else None

    def list_sealed_signals_in_session(self, session, publication_run_id: str) -> list[dict[str, Any]]:
        rows = session.scalars(
            select(ContentSealedSignalRow)
            .where(ContentSealedSignalRow.publication_run_id == publication_run_id)
            .order_by(ContentSealedSignalRow.signal_id)
        ).all()
        return [dict(row.payload or {}) for row in rows]

    def read_sealed(self, snapshot_id: str, query_hash: str, policy: str) -> dict[str, Any] | None:
        """Read the immutable retry source, including its manifest and signals."""
        with self._sessions() as session:
            row = session.scalar(select(ContentPublicationRunRow).where(
                ContentPublicationRunRow.content_snapshot_id == snapshot_id,
                ContentPublicationRunRow.query_hash == query_hash,
                ContentPublicationRunRow.signal_policy_version == policy,
            ))
            if row is None:
                return None
            return {
                "publication_run": _to_domain(row),
                "manifest": self.get_manifest_in_session(session, row.publication_run_id),
                "signals": self.list_sealed_signals_in_session(session, row.publication_run_id),
            }

    # Explicit aliases keep the read side discoverable for retry workers while
    # preserving the session-scoped methods used by the publication UoW.
    def get_sealed_manifest(self, publication_run_id: str) -> dict[str, Any] | None:
        with self._sessions() as session:
            return self.get_manifest_in_session(session, publication_run_id)

    def list_sealed_signals(self, publication_run_id: str) -> list[dict[str, Any]]:
        with self._sessions() as session:
            return self.list_sealed_signals_in_session(session, publication_run_id)

    def verify_sealed_in_session(
        self,
        session,
        run: ContentPublicationRun,
        manifest: dict[str, Any],
        signals: tuple[dict[str, Any], ...],
        sealed_hash: str,
    ) -> None:
        stored_manifest = self.get_manifest_in_session(session, run.publication_run_id)
        stored_signals = self.list_sealed_signals_in_session(session, run.publication_run_id)
        if stored_manifest is None:
            raise ValueError("publication sealed manifest is unavailable for retry")
        if stored_manifest != dict(manifest) or _signal_keyed_set(stored_signals) != _signal_keyed_set(signals):
            raise ValueError("publication sealed projection differs from retry payload")
        if run.manifest_hash != sealed_hash:
            raise ValueError("publication sealed hash differs from retry payload")

    @contextmanager
    def transaction(self):
        with self._sessions.begin() as session:
            yield session

    def transition(
        self, run: ContentPublicationRun, state: PublicationState | str, *, manifest_hash: str | None = None
    ) -> ContentPublicationRun:
        updated = run.transition(state, manifest_hash=manifest_hash)
        return self.save(updated)


def _to_domain(row: ContentPublicationRunRow) -> ContentPublicationRun:
    return ContentPublicationRun(
        content_snapshot_id=row.content_snapshot_id,
        query_hash=row.query_hash,
        signal_policy_version=row.signal_policy_version,
        state=PublicationState(row.state),
        manifest_hash=row.manifest_hash,
        version=row.version,
        publication_run_id=row.publication_run_id,
    )


def _signal_keyed_set(rows: Any) -> dict[str, dict[str, Any]]:
    """Compare signal collections independent of caller/database ordering."""
    keyed: dict[str, dict[str, Any]] = {}
    for payload in rows:
        signal_id = str(payload.get("signal_id") or "")
        if not signal_id:
            raise ValueError("sealed signal requires signal_id")
        if signal_id in keyed:
            raise ValueError(f"duplicate sealed signal_id: {signal_id}")
        keyed[signal_id] = dict(payload)
    return keyed


__all__ = ["PublicationRepository"]

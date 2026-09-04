"""Atomic SQL publication orchestration with Qdrant kept outside the commit."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Any, Callable, Iterable

from sqlalchemy.exc import IntegrityError

from stock_content.domain.publication_run import ContentPublicationRun, PublicationState, manifest_hash


class PublicationConflictError(ValueError):
    """A publication identity was reused with different immutable content."""


class InMemoryPublicationRepository:
    def __init__(self) -> None:
        self.runs: dict[tuple[str, str, str], ContentPublicationRun] = {}

    def get_by_identity(self, snapshot_id: str, query_hash: str, policy: str):
        return self.runs.get((snapshot_id, query_hash, policy))

    def get_by_identity_in_session(self, session, snapshot_id: str, query_hash: str, policy: str):
        return self.get_by_identity(snapshot_id, query_hash, policy)

    def save(self, run):
        self.runs[run.identity] = run
        return run

    def is_ready(self, snapshot_id: str) -> bool:
        return any(run.content_snapshot_id == snapshot_id and run.state in {
            PublicationState.READY, PublicationState.PUBLISHING, PublicationState.PUBLISHED
        } for run in self.runs.values())


class PublicationUnitOfWork:
    def __init__(
        self,
        repository: Any,
        *,
        snapshot_writer: Callable[..., Any] | None = None,
        signal_writer: Callable[..., Any] | None = None,
        outbox_writer: Callable[..., Any] | None = None,
    ) -> None:
        self.repository = repository
        self.snapshot_writer, self.signal_writer, self.outbox_writer = snapshot_writer, signal_writer, outbox_writer

    def publish(
        self,
        *,
        content_snapshot_id: str,
        query_hash: str,
        signal_policy_version: str,
        manifest: dict[str, Any],
        signals: Iterable[dict[str, Any]] = (),
        outbox_events: Iterable[dict[str, Any]] = (),
        failure_hook: Callable[[str], None] | None = None,
        session: Any | None = None,
        snapshot: Any | None = None,
        snapshot_bundle: dict[str, Any] | None = None,
    ) -> ContentPublicationRun:
        identity = (content_snapshot_id, query_hash, signal_policy_version)
        signal_rows = tuple(signals)
        outbox_rows = tuple(outbox_events)
        mh = manifest_hash({"manifest": manifest, "signals": signal_rows, "outbox": outbox_rows})
        run = ContentPublicationRun(*identity)
        transaction = (
            nullcontext(session)
            if session is not None
            else self.repository.transaction()
            if hasattr(self.repository, "transaction")
            else nullcontext(None)
        )
        created_run = False
        try:
            with transaction as session:
                if hasattr(self.repository, "get_by_identity_in_session"):
                    existing = self.repository.get_by_identity_in_session(session, *identity)
                else:
                    existing = self.repository.get_by_identity(*identity)
                if existing is not None:
                    if existing.manifest_hash != mh:
                        raise PublicationConflictError(
                            "publication identity already contains a different sealed snapshot/signals payload"
                        )
                    verifier = getattr(self.repository, "verify_sealed_in_session", None)
                    if verifier is not None:
                        verifier(session, existing, manifest, signal_rows, mh)
                    return existing
                created_run = True
                run = self._save(run, PublicationState.PROJECTING, failure_hook, session=session)
                if self.snapshot_writer:
                    self._write_snapshot(
                        self.snapshot_writer,
                        session,
                        content_snapshot_id,
                        manifest,
                        snapshot=snapshot,
                        snapshot_bundle=snapshot_bundle,
                    )
                manifest_writer = getattr(self.repository, "save_manifest_in_session", None)
                if manifest_writer is not None:
                    manifest_writer(session, run, manifest, mh)
                if failure_hook:
                    failure_hook("snapshot")
                run = self._save(run, PublicationState.SEALING, failure_hook, session=session)
                if self.signal_writer:
                    self._write_signal(self.signal_writer, session, signal_rows, run.publication_run_id)
                if failure_hook:
                    failure_hook("signals")
                if self.outbox_writer:
                    self._write(self.outbox_writer, session, outbox_rows)
                if failure_hook:
                    failure_hook("outbox")
                return self._save(run, PublicationState.READY, failure_hook, manifest_hash=mh, session=session)
        except IntegrityError:
            # A concurrent publisher may win the unique identity race.  Once
            # its transaction is visible, only an identical sealed payload is
            # idempotent; conflicting content must still fail closed.
            existing = self.repository.get_by_identity(*identity)
            if existing is not None and existing.manifest_hash == mh:
                return existing
            if existing is not None:
                raise PublicationConflictError(
                    "publication identity was concurrently sealed with different content"
                )
            raise
        except Exception:
            # Repository implementations using a transaction roll back all
            # rows. In-memory repositories remove the unfinished run.
            if created_run and hasattr(self.repository, "runs"):
                self.repository.runs.pop(identity, None)
            raise

    def _save(
        self,
        run: ContentPublicationRun,
        state: PublicationState,
        failure_hook: Callable[[str], None] | None,
        *,
        manifest_hash: str | None = None,
        session=None,
    ):
        updated = run.transition(state, manifest_hash=manifest_hash)
        if failure_hook:
            failure_hook(state.value.lower())
        if session is not None:
            return self.repository.save_in_session(session, updated)
        return self.repository.save(updated)

    @staticmethod
    def _write(writer, session, *args):
        if session is not None:
            writer(session, *args)
        else:
            writer(*args)

    @staticmethod
    def _write_signal(writer, session, rows, publication_run_id: str) -> None:
        """Pass publication identity to new signal writers, retain old arity."""
        try:
            parameters = inspect.signature(writer).parameters
        except (TypeError, ValueError):
            parameters = {}
        supports_identity = "publication_run_id" in parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        )
        if supports_identity:
            if session is not None:
                writer(session, rows, publication_run_id=publication_run_id)
            else:
                writer(rows, publication_run_id=publication_run_id)
            return
        PublicationUnitOfWork._write(writer, session, rows)

    @staticmethod
    def _write_snapshot(
        writer,
        session,
        content_snapshot_id: str,
        manifest: dict[str, Any],
        *,
        snapshot: Any | None,
        snapshot_bundle: dict[str, Any] | None,
    ) -> None:
        """Write a candidate snapshot while preserving legacy writer arity.

        The four-argument form is used by the production pipeline so the
        snapshot and its dependent membership/ledger rows are written through
        the caller-owned SQL session.  Existing integrations that only know
        the historical ``(session, snapshot_id, manifest)`` callback continue
        to work when no candidate object is supplied.
        """
        if snapshot is not None:
            if session is not None:
                writer(session, snapshot, manifest, snapshot_bundle or {})
            else:
                writer(snapshot, manifest, snapshot_bundle or {})
            return
        PublicationUnitOfWork._write(writer, session, content_snapshot_id, manifest)


__all__ = ["InMemoryPublicationRepository", "PublicationConflictError", "PublicationUnitOfWork"]

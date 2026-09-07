"""Immutable content artifact repository (SQLite and PostgreSQL)."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ClaimArtifactMemberRow,
    ContentArtifactEdgeRow,
    ContentArtifactRow,
    ContentStageCheckpointRow,
    ContentTaskEffectRow,
    ContentTaskRow,
    RetentionArtifactLocatorRow,
    SourceArtifactMetadataRow,
)
from stock_content.domain.artifacts import (
    ArtifactBase,
    SourceArtifact,
    artifact_identity_payload,
    canonical_json,
    deserialize_artifact,
    serialize_artifact,
)
from stock_content.domain.checkpoint import CheckpointRecord, CheckpointValidationError, checkpoint_state_checksum
from stock_content.ports.repositories import StaleTaskLease


class ArtifactIntegrityError(ValueError):
    """Artifact id/hash/payload immutability violation."""


def _fenced_checkpoint_payload_hash(checkpoint: Any, artifacts: list[ArtifactBase]) -> str:
    """Return a lease- and clock-independent stage-effect receipt hash.

    Checkpoint audit fields retain the worker/fence and timestamps, but those
    fields are not part of the business effect.  This permits a valid resumed
    worker to recognize an already committed stage without weakening the
    immutable input/output identity check.
    """
    stable_checksum = checkpoint_state_checksum(checkpoint)
    if checkpoint.state_checksum and checkpoint.state_checksum != stable_checksum:
        raise ArtifactIntegrityError("fenced checkpoint state checksum is invalid")
    payload = {
        "checkpoint_state_checksum": stable_checksum,
        "schema_version": str(checkpoint.schema_version),
        "output_artifact_ids": list(checkpoint.output_artifact_ids),
        "output_artifact_hashes": list(checkpoint.output_hashes),
        "artifacts": [artifact.content_hash for artifact in artifacts],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _json_payload(artifact: ArtifactBase) -> dict:
    return json.loads(canonical_json(serialize_artifact(artifact)))


def _stored_identity(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key not in {"artifact_id", "created_at", "content_hash"}}


def _expected_id(artifact: ArtifactBase) -> str:
    digest = hashlib.sha256(canonical_json(artifact_identity_payload(artifact)).encode("utf-8")).hexdigest()[:32]
    return f"{artifact.artifact_type}-{digest}"


def _expected_hash(artifact: ArtifactBase) -> str:
    return hashlib.sha256(canonical_json(artifact_identity_payload(artifact)).encode("utf-8")).hexdigest()


def _validate_object(artifact: ArtifactBase) -> str:
    """Independently validate an object before it crosses the persistence boundary."""
    expected_hash = _expected_hash(artifact)
    if artifact.content_hash != expected_hash:
        raise ArtifactIntegrityError(
            f"artifact content_hash does not match canonical identity: {artifact.artifact_id}"
        )
    _validate_id(artifact)
    return expected_hash


def _validate_row_payload(row: ContentArtifactRow, payload: dict, artifact: ArtifactBase) -> None:
    """Reject disagreement between indexed columns, JSON payload and object."""
    expected_hash = _validate_object(artifact)
    expected_payload = _json_payload(artifact)
    if (
        row.artifact_id != artifact.artifact_id
        or payload.get("artifact_id") != row.artifact_id
        or row.artifact_type != artifact.artifact_type
        or payload.get("artifact_type") != row.artifact_type
        or row.schema_version != artifact.schema_version
        or payload.get("schema_version") != row.schema_version
        or row.producer_stage != artifact.producer_stage
        or payload.get("producer_stage") != row.producer_stage
        or row.producer_version != artifact.producer_version
        or payload.get("producer_version") != row.producer_version
        or list(row.parent_artifact_ids or []) != list(artifact.parent_artifact_ids)
        or list(payload.get("parent_artifact_ids") or []) != list(row.parent_artifact_ids or [])
        or row.content_hash != expected_hash
        or payload.get("content_hash") != expected_hash
        or _stored_identity(payload) != _stored_identity(expected_payload)
    ):
        raise ArtifactIntegrityError(f"artifact row/payload mismatch: {row.artifact_id}")


def _validate_id(artifact: ArtifactBase) -> None:
    """Validate content-addressed ids while retaining legacy ids for replay."""
    if (
        len(artifact.artifact_id.rsplit("-", 1)[-1]) == 32
        and artifact.artifact_id != _expected_id(artifact)
    ):
        raise ArtifactIntegrityError("artifact_id does not match canonical identity")


class SqlArtifactRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def put(self, artifact: ArtifactBase) -> ArtifactBase:
        with self._sessions.begin() as session:
            return _put_artifact_in_session(session, artifact)

    def put_in_session(self, session, artifact: ArtifactBase) -> ArtifactBase:
        """Persist an artifact in a caller-owned publication transaction."""
        return _put_artifact_in_session(session, artifact)

    def put_with_checkpoint(self, artifacts: Iterable[ArtifactBase], task_id: str, checkpoint: Any) -> None:
        """Persist stage artifacts and checkpoint in one database transaction."""
        items = list(artifacts)
        with self._sessions.begin() as session:
            self._put_checkpoint_in_session(session, items, task_id, checkpoint)

    def put_with_fenced_checkpoint(
        self,
        artifacts: Iterable[ArtifactBase],
        task_id: str,
        checkpoint: Any,
        worker_id: str,
        fencing_token: int,
    ) -> None:
        """Atomically persist a stage receipt only for the current task lease.

        A prior worker may still finish a model/download call after takeover;
        its immutable artifact is harmless, but its checkpoint must not make
        that output resumable or advance the task's durable recovery point.
        """
        items = list(artifacts)
        with self._sessions.begin() as session:
            row = session.scalar(
                select(ContentTaskRow).where(ContentTaskRow.task_id == task_id).with_for_update()
            )
            expires_at = row.lease_expires_at if row is not None else None
            if expires_at is not None and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            now = datetime.now(UTC)
            if (
                row is None
                or row.status != "RUNNING"
                or row.lease_owner != worker_id
                or row.fencing_token != fencing_token
                or expires_at is None
                or expires_at <= now
            ):
                raise StaleTaskLease(f"content task lease is stale: {task_id}")
            # A FAILED checkpoint is diagnostic/recovery state, not a
            # completed business effect.  It must remain retryable by a later
            # current lease; only a successful receipt consumes the stable
            # stage effect identity.
            if checkpoint.status != "SUCCEEDED":
                self._put_checkpoint_in_session(session, items, task_id, checkpoint)
                return
            effect_key = f"stage:{checkpoint.stage}:{checkpoint.stage_version}"
            # A task can resume in another process with a different worker,
            # lease token and wall clock.  The durable receipt must bind only
            # the frozen recovery facts, not those runtime audit fields.
            payload_hash = _fenced_checkpoint_payload_hash(checkpoint, items)
            effect = session.scalar(
                select(ContentTaskEffectRow)
                .where(ContentTaskEffectRow.task_id == task_id, ContentTaskEffectRow.effect_key == effect_key)
                .with_for_update()
            )
            if effect is not None and (
                effect.payload_hash != payload_hash or effect.effect_kind != "STAGE_ARTIFACT_CHECKPOINT"
            ):
                raise ArtifactIntegrityError("fenced stage effect identity already stores different payload")
            if effect is not None:
                # Preserve the original immutable receipt; a retry must not
                # overwrite it with a new process/lease audit envelope.
                return
            self._put_checkpoint_in_session(session, items, task_id, checkpoint)
            session.add(
                ContentTaskEffectRow(
                    effect_id="eff_" + hashlib.sha256(f"{task_id}|{effect_key}".encode()).hexdigest(),
                    task_id=task_id,
                    effect_key=effect_key,
                    effect_kind="STAGE_ARTIFACT_CHECKPOINT",
                    payload_hash=payload_hash,
                    fencing_token=fencing_token,
                    state="COMPLETED",
                    attempt_count=1,
                    completed_at=now,
                )
            )

    @staticmethod
    def _put_checkpoint_in_session(session, artifacts: list[ArtifactBase], task_id: str, checkpoint: Any) -> None:
        for artifact in artifacts:
            _put_artifact_in_session(session, artifact)
        checkpoint_id = f"{task_id}:{checkpoint.stage}:{checkpoint.stage_version}"
        existing_checkpoint = session.get(ContentStageCheckpointRow, checkpoint_id)
        if existing_checkpoint is None:
            session.add(
                ContentStageCheckpointRow(
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    stage=checkpoint.stage,
                    stage_version=checkpoint.stage_version,
                    status=checkpoint.status,
                    artifact_ids=list(checkpoint.output_artifact_ids),
                    artifact_hashes=list(checkpoint.output_hashes),
                    payload=checkpoint.to_dict(),
                )
            )
            return
        history = list((existing_checkpoint.payload or {}).get("attempt_history") or [])
        history.append(dict(existing_checkpoint.payload or {}))
        payload = checkpoint.to_dict()
        payload["attempt_history"] = history
        existing_checkpoint.status = checkpoint.status
        existing_checkpoint.artifact_ids = list(checkpoint.output_artifact_ids)
        existing_checkpoint.artifact_hashes = list(checkpoint.output_hashes)
        existing_checkpoint.payload = payload

    def list_checkpoints(self, task_id: str) -> list[CheckpointRecord]:
        """Return the durable stage history in execution order."""
        with self._sessions() as session:
            rows = session.scalars(
                select(ContentStageCheckpointRow)
                .where(ContentStageCheckpointRow.task_id == task_id)
                .order_by(ContentStageCheckpointRow.created_at, ContentStageCheckpointRow.checkpoint_id)
            ).all()
            records: list[CheckpointRecord] = []
            for row in rows:
                payload = dict(row.payload or {})
                # The JSON payload is a denormalized copy of the indexed
                # checkpoint columns.  A tampered copy must never be treated
                # as an empty/valid checkpoint (which would cause a restart).
                if payload:
                    if str(payload.get("stage") or row.stage) != row.stage:
                        raise CheckpointValidationError(
                            f"CHECKPOINT_ERROR: stage payload mismatch for {row.checkpoint_id}"
                        )
                    if str(payload.get("stage_version") or row.stage_version) != row.stage_version:
                        raise CheckpointValidationError(
                            f"CHECKPOINT_ERROR: stage version payload mismatch for {row.checkpoint_id}"
                        )
                    if str(payload.get("status") or row.status) != row.status:
                        raise CheckpointValidationError(
                            f"CHECKPOINT_ERROR: status payload mismatch for {row.checkpoint_id}"
                        )
                    payload_ids = list(payload.get("output_artifact_ids") or [])
                    payload_hashes = list(payload.get("output_hashes") or [])
                    if payload_ids != list(row.artifact_ids or []) or payload_hashes != list(row.artifact_hashes or []):
                        raise CheckpointValidationError(
                            f"CHECKPOINT_ERROR: checkpoint payload mismatch for {row.checkpoint_id}"
                        )
                payload.setdefault("stage", row.stage)
                payload.setdefault("stage_version", row.stage_version)
                payload.setdefault("status", row.status)
                payload.setdefault("output_artifact_ids", list(row.artifact_ids or []))
                payload.setdefault("output_hashes", list(row.artifact_hashes or []))
                records.append(CheckpointRecord.from_dict(payload))
            return records

    def load_checkpoints(
        self, task_id: str, stage_versions: dict[str, str] | None = None
    ) -> tuple[list[CheckpointRecord], dict[str, ArtifactBase]]:
        """Load and strictly validate durable checkpoints and their artifacts.

        This is intentionally fail-closed: a missing row, changed payload or
        stage version is an integrity error rather than a signal to download
        the source again.
        """
        records = self.list_checkpoints(task_id)
        with self._sessions() as session:
            referenced = {
                artifact_id
                for record in records
                for artifact_id in (*record.input_artifact_ids, *record.output_artifact_ids)
                if artifact_id
            }
            rows = {
                row.artifact_id: row
                for row in session.scalars(
                    select(ContentArtifactRow).where(ContentArtifactRow.artifact_id.in_(referenced))
                ).all()
            }
            artifacts: dict[str, ArtifactBase] = {}
            for artifact_id in referenced:
                row = rows.get(artifact_id)
                if row is None:
                    raise ArtifactIntegrityError(f"ARTIFACT_INTEGRITY_ERROR: missing artifact {artifact_id}")
                payload = dict(row.payload or {})
                try:
                    artifact = deserialize_artifact(payload)
                    expected_hash = _expected_hash(artifact)
                    _validate_row_payload(row, payload, artifact)
                except Exception as exc:  # noqa: BLE001 - stable integrity boundary
                    raise ArtifactIntegrityError(
                        f"ARTIFACT_INTEGRITY_ERROR: invalid artifact {artifact_id}: {exc}"
                    ) from exc
                if (
                    row.content_hash != expected_hash
                    or payload.get("content_hash") != expected_hash
                    or artifact.content_hash != expected_hash
                ):
                    raise ArtifactIntegrityError(
                        f"ARTIFACT_INTEGRITY_ERROR: hash mismatch for artifact {artifact_id}"
                    )
                artifacts[artifact_id] = artifact

            for record in records:
                expected_version = (stage_versions or {}).get(record.stage)
                if expected_version and record.stage_version != expected_version:
                    raise CheckpointValidationError(
                        f"CHECKPOINT_ERROR: checkpoint stage version incompatible for {record.stage}: "
                        f"checkpoint={record.stage_version} current={expected_version}"
                    )
                if len(record.output_artifact_ids) != len(record.output_hashes):
                    raise CheckpointValidationError(
                        f"checkpoint output hash count mismatch for stage {record.stage}"
                    )
                for artifact_id, expected_hash in zip(
                    record.output_artifact_ids, record.output_hashes
                ):
                    artifact = artifacts.get(artifact_id)
                    if artifact is None:
                        raise ArtifactIntegrityError(
                            f"ARTIFACT_INTEGRITY_ERROR: checkpoint artifact missing {artifact_id}"
                        )
                    if expected_hash and artifact.content_hash != expected_hash:
                        raise ArtifactIntegrityError(
                            f"ARTIFACT_INTEGRITY_ERROR: checkpoint hash mismatch {artifact_id}"
                        )
        return records, artifacts

    # Descriptive alias for callers that want to make the strict behavior
    # explicit.
    load_checkpoint_state = load_checkpoints

    def get(self, artifact_id: str) -> ArtifactBase | None:
        with self._sessions() as session:
            row = session.get(ContentArtifactRow, artifact_id)
            if row is None:
                return None
            payload = dict(row.payload or {})
            artifact = deserialize_artifact(payload)
            _validate_row_payload(row, payload, artifact)
            return artifact

    def find_task_options_for_snapshot(self, artifact_ids: dict[str, str]) -> dict[str, Any] | None:
        """Recover immutable source options through checkpoint/task lineage.

        Task options intentionally do not live in Snapshot identity.  A
        snapshot's artifact set is recorded by stage checkpoints, so finding
        the task whose checkpoint history contains that set gives Replay the
        original fixture/model/input manifest without refetching the source.
        """
        wanted = {str(value) for value in artifact_ids.values() if value}
        if not wanted:
            return None
        with self._sessions() as session:
            rows = session.scalars(select(ContentStageCheckpointRow)).all()
            artifacts_by_task: dict[str, set[str]] = {}
            for row in rows:
                payload = dict(row.payload or {})
                recorded = set(str(item) for item in (row.artifact_ids or []))
                recorded.update(str(item) for item in (payload.get("input_artifact_ids") or []))
                recorded.update(str(item) for item in (payload.get("output_artifact_ids") or []))
                artifacts_by_task.setdefault(str(row.task_id), set()).update(recorded)
            candidates = {
                task_id for task_id, recorded in artifacts_by_task.items() if wanted.issubset(recorded)
            }
            if not candidates:
                return None
            # A content-addressed artifact set can be produced by retries; all
            # matching tasks have equivalent immutable options.  Prefer the
            # earliest task for deterministic recovery.
            task_rows = session.scalars(
                select(ContentTaskRow).where(ContentTaskRow.task_id.in_(candidates))
            ).all()
            if not task_rows:
                return None
            task_rows.sort(key=lambda item: (item.created_at, item.task_id))
            return dict(task_rows[0].options or {})

    def put_claim_members(self, artifact: ArtifactBase) -> None:
        """Persist the reverse membership for a canonical ClaimArtifact."""
        with self._sessions.begin() as session:
            self.put_claim_members_in_session(session, artifact)

    def put_claim_members_in_session(self, session, artifact: ArtifactBase) -> None:
        claims = list(getattr(artifact, "claims", ()) or ())
        for claim_id in claims:
            claim_id = str(getattr(claim_id, "claim_id", claim_id))
            member_id = hashlib.sha256(f"{artifact.artifact_id}:{claim_id}".encode()).hexdigest()
            _insert_ignore(
                session,
                ClaimArtifactMemberRow,
                {
                    "member_id": member_id,
                    "artifact_id": artifact.artifact_id,
                    "claim_id": str(claim_id),
                },
                [ClaimArtifactMemberRow.member_id],
            )

    def get_many(self, artifact_ids: Iterable[str]) -> list[ArtifactBase]:
        ids = list(artifact_ids)
        if not ids:
            return []
        with self._sessions() as session:
            rows = session.scalars(select(ContentArtifactRow).where(ContentArtifactRow.artifact_id.in_(ids))).all()
            by_id: dict[str, ArtifactBase] = {}
            for row in rows:
                payload = dict(row.payload or {})
                artifact = deserialize_artifact(payload)
                _validate_row_payload(row, payload, artifact)
                by_id[row.artifact_id] = artifact
        return [by_id[item] for item in ids if item in by_id]

    def verify(self, artifact_id: str) -> bool:
        with self._sessions() as session:
            row = session.get(ContentArtifactRow, artifact_id)
            if row is None:
                raise KeyError(artifact_id)
            payload = dict(row.payload or {})
            artifact = deserialize_artifact(payload)
            _validate_row_payload(row, payload, artifact)
            return True

    def lineage(self, artifact_id: str) -> dict:
        """Return a complete, deterministic artifact DAG rooted at ``artifact_id``.

        ``artifact`` and the immediate ``parents`` list retain the historical
        response shape.  Each parent additionally carries its own nested
        ``parents`` list, and ``lineage`` exposes the same recursive root for
        callers that need an unambiguous tree.  Missing parents and cycles are
        reported as an incomplete graph: callers must reject
        ``lineage_complete=False`` rather than treating a partial graph as
        source-backed.  The legacy root artifact remains available for
        diagnostics.
        """
        root = self.get(artifact_id)
        if root is None:
            return {}

        errors: list[str] = []

        def walk(current_id: str, path: tuple[str, ...]) -> dict | None:
            if current_id in path:
                cycle = " -> ".join((*path, current_id))
                errors.append(f"artifact lineage cycle detected: {cycle}")
                return None
            current = self.get(current_id)
            if current is None:
                parent = path[-1] if path else artifact_id
                errors.append(f"artifact lineage parent missing: {parent} -> {current_id}")
                return None
            payload = serialize_artifact(current)
            payload["parents"] = [
                parent
                for parent_id in sorted(str(item) for item in (current.parent_artifact_ids or ()))
                if (parent := walk(str(parent_id), (*path, current_id))) is not None
            ]
            return payload

        tree = walk(str(artifact_id), ())
        if errors:
            # Do not expose a potentially misleading partial tree. Keep the
            # legacy root response shape and add machine-readable completeness
            # and error fields for fail-closed consumers.
            return {
                "artifact": serialize_artifact(root),
                "parents": [],
                "lineage": None,
                "lineage_complete": False,
                "lineage_errors": sorted(set(errors)),
            }
        assert tree is not None  # root was already loaded above
        # Keep the old top-level artifact serialization (without recursive
        # fields) and immediate parent objects, while returning a recursive
        # tree under the additive ``lineage`` key.
        return {
            "artifact": serialize_artifact(root),
            "parents": tree["parents"],
            "lineage": tree,
            "lineage_complete": True,
            "lineage_errors": [],
        }


def _put_artifact_in_session(session, artifact: ArtifactBase) -> ArtifactBase:
    """Insert an immutable artifact atomically, then validate the winner.

    A read-then-add sequence turns two legitimate retries into a raw unique
    violation.  Native ``ON CONFLICT DO NOTHING`` makes the loser re-read the
    committed row; the normal integrity checks then distinguish an idempotent
    retry from an ID collision with different content.
    """
    expected_hash = _validate_object(artifact)
    payload = _json_payload(artifact)
    values = {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "schema_version": artifact.schema_version,
        "producer_stage": artifact.producer_stage,
        "producer_version": artifact.producer_version,
        "content_hash": expected_hash,
        "parent_artifact_ids": list(artifact.parent_artifact_ids),
        "payload": payload,
        "created_at": artifact.created_at,
    }
    # Conflict on either the immutable id or the content-addressed
    # (artifact_type, content_hash) key is handled as a read-after-write.
    # The no-target form is important here: targeting only artifact_id would
    # allow a second legacy id for the same canonical content to look like a
    # successful insert.
    _insert_ignore(session, ContentArtifactRow, values, None)
    row = session.get(ContentArtifactRow, artifact.artifact_id)
    if row is None:
        row = session.scalar(
            select(ContentArtifactRow).where(
                ContentArtifactRow.artifact_type == artifact.artifact_type,
                ContentArtifactRow.content_hash == expected_hash,
            )
        )
        if row is None:
            raise RuntimeError("artifact disappeared after a unique-key conflict")
    stored_payload = dict(row.payload or {})
    stored = deserialize_artifact(stored_payload)
    _validate_row_payload(row, stored_payload, stored)
    if _stored_identity(stored_payload) != _stored_identity(payload):
        raise ArtifactIntegrityError(f"artifact id {artifact.artifact_id} already stores a different payload")

    for parent_id in stored.parent_artifact_ids:
        edge_id = hashlib.sha256(f"{stored.artifact_id}:{parent_id}".encode()).hexdigest()
        _insert_ignore(
            session,
            ContentArtifactEdgeRow,
            {
                "edge_id": edge_id,
                "artifact_id": stored.artifact_id,
                "parent_artifact_id": parent_id,
                "relation": "PARENT",
            },
            [ContentArtifactEdgeRow.edge_id],
        )
    if isinstance(stored, SourceArtifact):
        _persist_source_provenance(session, stored)
    return stored


def _persist_source_provenance(session, artifact: SourceArtifact) -> None:
    """Write closed public provenance in the same artifact transaction."""
    metadata = dict(artifact.source_metadata or {})
    required = {
        "source_policy_version": metadata.get("source_policy_version"),
        "access_classification": metadata.get("access_classification"),
        "mime_type": metadata.get("mime_type") or "application/octet-stream",
    }
    if not all(str(value or "").strip() for value in required.values()):
        # Legacy/replayed artifacts are not upgraded by guessing provenance.
        return
    values = {
        "artifact_id": artifact.artifact_id,
        "source_policy_version": str(required["source_policy_version"]),
        # Candidate enumeration operates on byte classes, not a broad policy
        # name such as "standard".
        "retention_class": "raw_media",
        "access_classification": str(required["access_classification"]),
        "source_content_hash": str(artifact.source_content_hash or artifact.raw_content_hash),
        "content_size": int(artifact.raw_content_length or 0),
        "mime_type": str(required["mime_type"]),
        "encryption_key_id": _safe_identifier(metadata.get("encryption_key_id")),
        "canonical_url": _public_url(metadata.get("canonical_url")),
        "source_type": _safe_identifier(artifact.source_type),
        "source_id": _safe_identifier(metadata.get("source_id")),
        "source_part": _safe_identifier(metadata.get("source_part")),
        "source_identity_hash": _safe_hash(artifact.source_identity_hash),
        "source_version_id": _safe_identifier(artifact.source_version_id),
        "author": _safe_identifier(metadata.get("author")),
        "published_at": _instant(metadata.get("published_at")),
        "business_as_of": _instant(metadata.get("business_as_of")),
        "source_available_from": _instant(metadata.get("source_available_from")),
        "pipeline_version": _safe_identifier(metadata.get("pipeline_version")),
        "service_version": _safe_identifier(os.getenv("CONTENT_SERVICE_VERSION")),
    }
    existing = session.get(SourceArtifactMetadataRow, artifact.artifact_id)
    if existing is None:
        session.add(SourceArtifactMetadataRow(**values))
    else:
        for key, value in values.items():
            stored = getattr(existing, key)
            # SQLite strips timezone metadata from DateTime(timezone=True).
            # Compare normalized instants so a retry of the exact immutable
            # artifact cannot be rejected merely by the test backend's
            # representation change.
            if key in {"published_at", "business_as_of", "source_available_from"}:
                stored = _instant(stored)
            if stored != value:
                raise ArtifactIntegrityError("source artifact provenance is immutable")
    _register_private_locator(session, artifact)


def _register_private_locator(session, artifact: SourceArtifact) -> None:
    root_text = os.getenv("CONTENT_RETENTION_PRIVATE_ROOT", "").strip()
    root_id = os.getenv("CONTENT_RETENTION_PRIVATE_ROOT_ID", "").strip()
    if not root_text or not root_id or not artifact.raw_storage_uri:
        return
    root = Path(root_text)
    if not root.is_dir():
        return
    raw = str(artifact.raw_storage_uri)
    # ``urlsplit`` mistakes a Windows drive (``D:\\...``) for a URI scheme.
    # Treat it as a local path before applying the external-locator rejection.
    local_path = Path(raw)
    parsed = urlsplit(raw) if not local_path.drive else None
    # No external/signed locator can become a deletion target.
    if parsed is not None and parsed.scheme and parsed.scheme != "file":
        return
    candidate = Path(parsed.path if parsed is not None and parsed.scheme == "file" else raw)
    try:
        resolved_root, resolved_candidate = root.resolve(strict=True), candidate.resolve(strict=True)
    except OSError:
        return
    if (
        resolved_candidate == resolved_root
        or resolved_root not in resolved_candidate.parents
        or not resolved_candidate.is_file()
    ):
        return
    relative = resolved_candidate.relative_to(resolved_root).as_posix()
    existing = session.get(RetentionArtifactLocatorRow, artifact.artifact_id)
    if existing is None:
        session.add(RetentionArtifactLocatorRow(
            artifact_id=artifact.artifact_id, private_root_id=root_id, relative_locator=relative,
        ))
    elif (existing.private_root_id, existing.relative_locator) != (root_id, relative):
        raise ArtifactIntegrityError("retention locator is immutable")


def _public_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        return None
    return f"https://{parsed.netloc}{parsed.path}"


def _safe_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or any(token in candidate.lower() for token in ("cookie", "secret", "token", "signed", "://")):
        return None
    return candidate


def _safe_hash(value: Any) -> str | None:
    value = _safe_identifier(value)
    return value if value and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower()) else None


def _instant(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _insert_ignore(session, model, values: dict, conflict_columns: list | None) -> bool:
    """Insert a membership/immutable row without poisoning the transaction."""
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        statement = postgres_insert(model).values(**values)
    elif dialect == "sqlite":
        statement = sqlite_insert(model).values(**values)
    else:
        try:
            with session.begin_nested():
                session.add(model(**values))
                session.flush()
            return True
        except IntegrityError:
            return False
    if conflict_columns is None:
        statement = statement.on_conflict_do_nothing()
    else:
        statement = statement.on_conflict_do_nothing(index_elements=conflict_columns)
    result = session.execute(statement)
    return result.rowcount == 1


ArtifactRepository = SqlArtifactRepository
PostgresArtifactRepository = SqlArtifactRepository

__all__ = ["ArtifactIntegrityError", "ArtifactRepository", "PostgresArtifactRepository", "SqlArtifactRepository"]

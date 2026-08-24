"""ContentSnapshot 持久化（详细修改方案 §4 P0-2）。

PostgreSQL 保持权威状态；identity 列保存完整身份 payload 以便原样还原。
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import and_, case, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ContentArtifactEdgeRow,
    ContentArtifactRow,
    ContentSnapshotArtifactRow,
    ContentSnapshotRow,
    ContentSourceHeadRow,
)
from stock_content.adapters.postgres.repositories.artifact_repository import (
    ArtifactIntegrityError,
    _validate_row_payload,
)
from stock_content.domain.artifacts import canonical_json, deserialize_artifact
from stock_content.domain.lineage import ContentSnapshot, compute_artifact_root_hash


class SnapshotIntegrityError(RuntimeError):
    """A content snapshot id was reused for a different immutable payload."""


_ROW_IDENTITY_FIELDS = {
    "source_type": "source_type",
    "source_ref": "source_ref",
    "source_content_hash": "source_content_hash",
    "artifact_ids": "artifact_ids",
    "quant_market_snapshot_ids": "quant_market_snapshot_ids",
    "pipeline_version": "pipeline_version",
    "schema_version": "schema_version",
    "code_sha": "code_sha",
    "config_hash": "config_hash",
    "source_artifact_id": "source_artifact_id",
    "artifact_root_hash": "artifact_root_hash",
    "snapshot_kind": "snapshot_kind",
    "parent_snapshot_id": "parent_snapshot_id",
    "supersedes_snapshot_id": "supersedes_snapshot_id",
    "producer_manifest": "producer_manifest",
}

# These keys were added after the v1 content_snapshot identity.  A migrated
# v1 row is allowed to omit them, but if it contains one it is authoritative
# and must agree with every redundant/indexed column.
_ADDITIVE_IDENTITY_FIELDS = frozenset(
    {
        "source_artifact_id",
        "artifact_root_hash",
        "snapshot_kind",
        "parent_snapshot_id",
        "supersedes_snapshot_id",
        "producer_manifest",
        "model_versions",
        "prompt_versions",
        "configuration",
        "external_snapshots",
        "policy_versions",
    }
)
_V1_SCHEMA_VERSION = "content.snapshot.v1"
_V2_SCHEMA_VERSION = "content.snapshot.v2"
_CORE_IDENTITY_FIELDS = frozenset(
    {
        "content_snapshot_id",
        "source_type",
        "source_ref",
        "source_content_hash",
        "artifact_ids",
        "quant_market_snapshot_ids",
        "pipeline_version",
        "schema_version",
        "code_sha",
        "config_hash",
        "parser_version",
        "asr_model",
        "asr_model_version",
        "vision_model",
        "llm_model",
        "prompt_bundle_version",
        "entity_alias_version",
        "verification_policy_version",
    }
)
_V2_REQUIRED_IDENTITY_FIELDS = _CORE_IDENTITY_FIELDS | _ADDITIVE_IDENTITY_FIELDS


def _canonical_equal(left, right) -> bool:
    return canonical_json(left) == canonical_json(right)


def _validate_provenance_alignment(
    *,
    code_sha,
    config_hash,
    producer_manifest,
    schema_version: str,
    snapshot_id: str,
) -> None:
    """Reject disagreement between indexed and nested release provenance.

    ``producer_manifest`` was not part of the v1 normalized snapshot row.  A
    legacy row with no manifest therefore remains readable, while any
    manifest values that are present are still checked.  Modern rows and
    directly constructed dataclasses must carry the same effective values in
    both locations; identity/hash equality alone is insufficient because a
    caller can forge a self-consistent identity payload.
    """

    if producer_manifest is None:
        manifest = {}
    elif isinstance(producer_manifest, dict):
        manifest = producer_manifest
    else:
        raise SnapshotIntegrityError(
            f"snapshot {snapshot_id} producer_manifest must be an object"
        )
    raw_configs = manifest.get("configs")
    if raw_configs is None:
        configs = {}
    elif isinstance(raw_configs, dict):
        configs = raw_configs
    else:
        raise SnapshotIntegrityError(
            f"snapshot {snapshot_id} producer_manifest.configs must be an object"
        )

    top_code = "" if code_sha is None else str(code_sha)
    nested_code = "" if manifest.get("code_sha") is None else str(manifest.get("code_sha"))
    top_config = "" if config_hash is None else str(config_hash)
    nested_config = "" if configs.get("config_hash") is None else str(configs.get("config_hash"))

    # A v1 row without any manifest is the pre-provenance shape.  Do not
    # reinterpret its empty/default columns as a contradiction.
    if schema_version == _V1_SCHEMA_VERSION and not manifest:
        return
    if top_code or nested_code:
        if not top_code or not nested_code or top_code != nested_code:
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} code_sha disagrees with producer_manifest"
            )
    if top_config or nested_config:
        if not top_config or not nested_config or top_config != nested_config:
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} config_hash disagrees with producer_manifest"
            )


def _row_redundant_values(row: ContentSnapshotRow) -> dict:
    return {key: getattr(row, attribute) for key, attribute in _ROW_IDENTITY_FIELDS.items()}


def _validate_identity_against_values(
    identity: dict,
    values: dict,
    *,
    snapshot_id: str,
) -> None:
    if not isinstance(identity, dict):
        raise SnapshotIntegrityError(f"snapshot {snapshot_id} has a non-object identity")
    row_schema_version = values.get("schema_version")
    identity_schema_version = identity.get("schema_version")
    if identity_schema_version != row_schema_version:
        raise SnapshotIntegrityError(f"snapshot {snapshot_id} identity schema does not match the row")
    if row_schema_version == _V1_SCHEMA_VERSION:
        required_fields = _CORE_IDENTITY_FIELDS
    elif row_schema_version == _V2_SCHEMA_VERSION:
        required_fields = _V2_REQUIRED_IDENTITY_FIELDS
    else:
        raise SnapshotIntegrityError(f"snapshot {snapshot_id} has unsupported schema {row_schema_version!r}")
    missing_fields = required_fields - identity.keys()
    if missing_fields:
        raise SnapshotIntegrityError(
            f"snapshot {snapshot_id} identity is missing required fields: {sorted(missing_fields)}"
        )
    identity_id = identity.get("content_snapshot_id")
    if identity_id is not None and str(identity_id) != snapshot_id:
        raise SnapshotIntegrityError(f"snapshot {snapshot_id} identity id does not match the row id")
    for key, value in values.items():
        # Missing additive keys are the only permitted legacy omission.  For
        # all other keys, an identity which omits the field is still a valid
        # old row, so there is no value to compare.
        if key not in identity:
            continue
        if not _canonical_equal(identity[key], value):
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} identity field {key!r} disagrees with its redundant column"
            )

    artifact_ids = values.get("artifact_ids") or {}
    if not isinstance(artifact_ids, dict):
        raise SnapshotIntegrityError(f"snapshot {snapshot_id} artifact_ids must be an object")
    source_artifact_id = values.get("source_artifact_id")
    source_slot = artifact_ids.get("source")
    if source_artifact_id and source_slot and source_artifact_id != source_slot:
        raise SnapshotIntegrityError(f"snapshot {snapshot_id} source artifact does not match artifact_ids")
    artifact_root_hash = values.get("artifact_root_hash")
    if artifact_root_hash and artifact_root_hash != compute_artifact_root_hash(artifact_ids):
        raise SnapshotIntegrityError(f"snapshot {snapshot_id} artifact root does not match artifact_ids")
    _validate_provenance_alignment(
        code_sha=values.get("code_sha"),
        config_hash=values.get("config_hash"),
        producer_manifest=values.get("producer_manifest"),
        schema_version=str(row_schema_version),
        snapshot_id=snapshot_id,
    )


def _is_legacy_identity(identity: dict, row: ContentSnapshotRow) -> bool:
    return row.schema_version == _V1_SCHEMA_VERSION and identity.get("schema_version") == _V1_SCHEMA_VERSION


def _validate_snapshot_members(session, row: ContentSnapshotRow, identity: dict) -> None:
    members = session.scalars(
        select(ContentSnapshotArtifactRow).where(
            ContentSnapshotArtifactRow.content_snapshot_id == row.content_snapshot_id
        )
    ).all()
    artifact_ids = dict(row.artifact_ids or {})
    expected = {
        (
            hashlib.sha256(
                f"{row.content_snapshot_id}:{slot}:{artifact_id}".encode()
            ).hexdigest(),
            str(artifact_id),
            str(slot),
        )
        for slot, artifact_id in artifact_ids.items()
    }
    actual = {(member.member_id, str(member.artifact_id), str(member.slot)) for member in members}
    if not members and _is_legacy_identity(identity, row):
        # v1 rows predate the normalized membership table.  They remain
        # readable when no contradictory membership rows were persisted.
        return
    if len(actual) != len(members) or actual != expected:
        raise SnapshotIntegrityError(
            f"snapshot {row.content_snapshot_id} artifact membership is inconsistent"
        )


def _validate_snapshot_artifacts(
    session,
    *,
    artifact_ids: dict[str, str],
    schema_version: str,
    snapshot_id: str,
) -> None:
    """Validate every v2 artifact reference and its recursive parent chain.

    ArtifactRepository owns the canonical hash/payload/type validation.  This
    boundary only supplies the snapshot's references and converts its domain
    error into the snapshot integrity error used by all snapshot entrypoints.
    """
    legacy = schema_version == _V1_SCHEMA_VERSION
    visiting: set[str] = set()
    validated: set[str] = set()

    def visit(artifact_id: str) -> None:
        artifact_id = str(artifact_id)
        if artifact_id in validated:
            return
        if artifact_id in visiting:
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} artifact parent cycle includes {artifact_id}"
            )
        visiting.add(artifact_id)
        row = session.get(ContentArtifactRow, artifact_id)
        if row is None:
            visiting.remove(artifact_id)
            if legacy:
                return
            cause = ArtifactIntegrityError(f"missing artifact {artifact_id}")
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} references missing artifact {artifact_id}"
            ) from cause
        payload = dict(row.payload or {})
        try:
            artifact = deserialize_artifact(payload)
            _validate_row_payload(row, payload, artifact)
        except Exception as exc:  # noqa: BLE001 - stable snapshot integrity boundary
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} references invalid artifact {artifact_id}"
            ) from exc
        expected_edges = {
            (
                hashlib.sha256(f"{artifact_id}:{parent_id}".encode()).hexdigest(),
                artifact_id,
                str(parent_id),
                "PARENT",
            )
            for parent_id in (row.parent_artifact_ids or ())
        }
        edges = session.scalars(
            select(ContentArtifactEdgeRow).where(ContentArtifactEdgeRow.artifact_id == artifact_id)
        ).all()
        actual_edges = {
            (edge.edge_id, edge.artifact_id, str(edge.parent_artifact_id), str(edge.relation))
            for edge in edges
        }
        if edges and actual_edges != expected_edges:
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} artifact parent edges are inconsistent for {artifact_id}"
            )
        if not edges and not legacy and expected_edges:
            raise SnapshotIntegrityError(
                f"snapshot {snapshot_id} artifact parent edges are missing for {artifact_id}"
            )
        for parent_id in row.parent_artifact_ids or ():
            visit(str(parent_id))
        visiting.remove(artifact_id)
        validated.add(artifact_id)

    for artifact_id in artifact_ids.values():
        visit(str(artifact_id))


def _validate_snapshot_row(session, row: ContentSnapshotRow) -> None:
    identity = dict(row.identity or {})
    _validate_identity_against_values(
        identity,
        _row_redundant_values(row),
        snapshot_id=row.content_snapshot_id,
    )
    _validate_snapshot_members(session, row, identity)
    _validate_snapshot_artifacts(
        session,
        artifact_ids=dict(row.artifact_ids or {}),
        schema_version=row.schema_version,
        snapshot_id=row.content_snapshot_id,
    )


def _validate_snapshot_candidate(snapshot: ContentSnapshot) -> dict:
    payload = snapshot.to_dict()
    if isinstance(payload.get("created_at"), datetime):
        payload["created_at"] = payload["created_at"].isoformat()
    values = {
        key: getattr(snapshot, attribute)
        for key, attribute in _ROW_IDENTITY_FIELDS.items()
    }
    _validate_identity_against_values(payload, values, snapshot_id=snapshot.content_snapshot_id)
    return payload


def _compare_existing_identity(
    existing_identity: dict,
    candidate_identity: dict,
    *,
    snapshot_id: str,
) -> None:
    existing_identity = dict(existing_identity or {})
    existing_identity.pop("created_at", None)
    candidate_identity = dict(candidate_identity or {})
    candidate_identity.pop("created_at", None)
    is_legacy = existing_identity.get("schema_version") == _V1_SCHEMA_VERSION
    # A v1 row may omit only additive fields.  Every field it did persist is
    # immutable and must compare canonically (JSON tuple/list boundaries are
    # intentionally ignored by canonical_json).
    for key, value in existing_identity.items():
        if key not in candidate_identity:
            if is_legacy and key in _ADDITIVE_IDENTITY_FIELDS:
                continue
            raise SnapshotIntegrityError(
                f"snapshot id {snapshot_id} already stores a different payload"
            )
        if not _canonical_equal(value, candidate_identity[key]):
            raise SnapshotIntegrityError(
                f"snapshot id {snapshot_id} already stores a different payload"
            )


def _compare_candidate_to_existing_row(
    snapshot: ContentSnapshot,
    row: ContentSnapshotRow,
    existing_identity: dict,
) -> None:
    candidate_values = {
        key: getattr(snapshot, attribute)
        for key, attribute in _ROW_IDENTITY_FIELDS.items()
    }
    existing_values = _row_redundant_values(row)
    is_legacy = existing_identity.get("schema_version") == _V1_SCHEMA_VERSION
    for key in _ROW_IDENTITY_FIELDS:
        # For a legacy omission, the old identity is the compatibility
        # boundary; new additive columns may contain migration defaults.
        if key not in existing_identity:
            if is_legacy and key in _ADDITIVE_IDENTITY_FIELDS:
                continue
            raise SnapshotIntegrityError(
                f"snapshot id {snapshot.content_snapshot_id} identity omitted redundant column {key!r}"
            )
        if not _canonical_equal(candidate_values[key], existing_values[key]):
            raise SnapshotIntegrityError(
                f"snapshot id {snapshot.content_snapshot_id} redundant column {key!r} changed"
            )


def _row_to_snapshot(row: ContentSnapshotRow, session=None) -> ContentSnapshot:
    identity = dict(row.identity or {})
    _validate_identity_against_values(
        identity,
        _row_redundant_values(row),
        snapshot_id=row.content_snapshot_id,
    )
    if session is not None:
        _validate_snapshot_members(session, row, identity)
        _validate_snapshot_artifacts(
            session,
            artifact_ids=dict(row.artifact_ids or {}),
            schema_version=row.schema_version,
            snapshot_id=row.content_snapshot_id,
        )
    created_at = identity.get("created_at")
    if isinstance(created_at, str):
        parsed = datetime.fromisoformat(created_at)
        created_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    if not isinstance(created_at, datetime):
        created_at = row.created_at
    return ContentSnapshot(
        content_snapshot_id=row.content_snapshot_id,
        source_type=row.source_type,
        source_ref=row.source_ref,
        source_content_hash=row.source_content_hash,
        parser_version=identity.get("parser_version"),
        asr_model=identity.get("asr_model"),
        asr_model_version=identity.get("asr_model_version"),
        vision_model=identity.get("vision_model"),
        llm_model=identity.get("llm_model"),
        prompt_bundle_version=str(identity.get("prompt_bundle_version") or "prompt_bundle.v1"),
        entity_alias_version=str(identity.get("entity_alias_version") or "entity_alias.v1"),
        verification_policy_version=str(identity.get("verification_policy_version") or "verification_policy.v1"),
        quant_market_snapshot_ids=tuple(identity.get("quant_market_snapshot_ids") or []),
        code_sha=identity.get("code_sha") or "",
        config_hash=identity.get("config_hash") or "",
        pipeline_version=identity.get("pipeline_version") or row.pipeline_version,
        schema_version=identity.get("schema_version") or row.schema_version,
        artifact_ids=dict(identity.get("artifact_ids") or {}),
        source_artifact_id=str(identity.get("source_artifact_id") or row.source_artifact_id or ""),
        artifact_root_hash=str(identity.get("artifact_root_hash") or row.artifact_root_hash or ""),
        snapshot_kind=str(identity.get("snapshot_kind") or row.snapshot_kind or "INITIAL"),
        parent_snapshot_id=identity.get("parent_snapshot_id") or row.parent_snapshot_id,
        supersedes_snapshot_id=identity.get("supersedes_snapshot_id") or row.supersedes_snapshot_id,
        producer_manifest=dict(identity.get("producer_manifest") or row.producer_manifest or {}),
        model_versions=dict(identity.get("model_versions") or {}),
        prompt_versions=dict(identity.get("prompt_versions") or {}),
        configuration=dict(identity.get("configuration") or {}),
        external_snapshots=tuple(identity.get("external_snapshots") or ()),
        policy_versions=dict(identity.get("policy_versions") or {}),
        created_at=created_at or row.created_at,
    )


class SqlSnapshotStore:
    """SnapshotStore 的 SQL 实现（SQLite / PostgreSQL 通用）。"""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def save(self, snapshot: ContentSnapshot) -> ContentSnapshot:
        # Validate the candidate before opening a transaction so a malformed
        # direct dataclass cannot become an authoritative row.
        payload = _validate_snapshot_candidate(snapshot)
        with self._sessions.begin() as session:
            values = {
                "content_snapshot_id": snapshot.content_snapshot_id,
                "source_type": snapshot.source_type,
                "source_ref": snapshot.source_ref,
                "source_content_hash": snapshot.source_content_hash,
                "identity": payload,
                "artifact_ids": dict(snapshot.artifact_ids),
                "quant_market_snapshot_ids": list(snapshot.quant_market_snapshot_ids),
                "pipeline_version": snapshot.pipeline_version,
                "schema_version": snapshot.schema_version,
                "code_sha": snapshot.code_sha,
                "config_hash": snapshot.config_hash,
                "source_artifact_id": snapshot.source_artifact_id,
                "artifact_root_hash": snapshot.artifact_root_hash,
                "snapshot_kind": snapshot.snapshot_kind,
                "parent_snapshot_id": snapshot.parent_snapshot_id,
                "supersedes_snapshot_id": snapshot.supersedes_snapshot_id,
                "producer_manifest": dict(snapshot.producer_manifest),
                "created_at": snapshot.created_at,
            }
            _validate_snapshot_artifacts(
                session,
                artifact_ids=dict(snapshot.artifact_ids),
                schema_version=snapshot.schema_version,
                snapshot_id=snapshot.content_snapshot_id,
            )
            inserted = _insert_ignore(session, ContentSnapshotRow, values, [ContentSnapshotRow.content_snapshot_id])
            row = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
            if row is None:
                raise RuntimeError("snapshot disappeared after a unique-key conflict")
            if not inserted:
                # Snapshot rows are append-only.  Replaying the same identity
                # is idempotent; reusing the id with altered payload is data
                # corruption and must never silently overwrite history.
                _validate_snapshot_row(session, row)
                existing = dict(row.identity or {})
                _compare_existing_identity(existing, payload, snapshot_id=snapshot.content_snapshot_id)
                _compare_candidate_to_existing_row(snapshot, row, existing)
            for slot, artifact_id in snapshot.artifact_ids.items():
                member_id = hashlib.sha256(f"{snapshot.content_snapshot_id}:{slot}:{artifact_id}".encode()).hexdigest()
                _insert_ignore(
                    session,
                    ContentSnapshotArtifactRow,
                    {
                        "member_id": member_id,
                        "content_snapshot_id": snapshot.content_snapshot_id,
                        "artifact_id": artifact_id,
                        "slot": slot,
                    },
                    [ContentSnapshotArtifactRow.member_id],
                )
            if inserted:
                session.flush()
                _validate_snapshot_row(session, row)
            source_identity_hash = hashlib.sha256(f"{snapshot.source_type}:{snapshot.source_ref}".encode()).hexdigest()
            if inserted or session.get(ContentSourceHeadRow, source_identity_hash) is None:
                _upsert_source_head(
                    session,
                    source_identity_hash=source_identity_hash,
                    snapshot_id=snapshot.content_snapshot_id,
                    created_at=snapshot.created_at,
                )
        return snapshot

    def get(self, content_snapshot_id: str) -> ContentSnapshot | None:
        with self._sessions() as session:
            row = session.get(ContentSnapshotRow, content_snapshot_id)
            return _row_to_snapshot(row, session) if row else None

    def list_for_source(self, source_type: str, source_ref: str) -> list[ContentSnapshot]:
        with self._sessions() as session:
            rows = session.scalars(
                select(ContentSnapshotRow)
                .where(
                    ContentSnapshotRow.source_type == source_type,
                    ContentSnapshotRow.source_ref == source_ref,
                )
                .order_by(ContentSnapshotRow.created_at)
            ).all()
            return [_row_to_snapshot(row, session) for row in rows]


__all__ = ["SnapshotIntegrityError", "SqlSnapshotStore"]


def _insert_ignore(session, model, values: dict, conflict_columns: list) -> bool:
    """Insert immutable/member rows without surfacing a race as a 5xx."""
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
    result = session.execute(statement.on_conflict_do_nothing(index_elements=conflict_columns))
    return result.rowcount == 1


def _upsert_source_head(
    session,
    *,
    source_identity_hash: str,
    snapshot_id: str,
    created_at: datetime,
) -> None:
    """Atomically advance a source head without allowing an older write to win.

    The read-then-insert implementation is vulnerable when two first snapshots
    for one source commit concurrently: both transactions observe no head and
    one loses on the unique key.  PostgreSQL and SQLite both serialize their
    native ``ON CONFLICT`` update for us.  The conditional update is deliberately
    monotonic by ``created_at`` (with a deterministic id tie-breaker), so the
    transaction that commits last cannot roll the head back to an older
    snapshot.
    """
    values = {
        "source_identity_hash": source_identity_hash,
        "latest_snapshot_id": snapshot_id,
        "updated_at": created_at,
    }
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        statement = postgres_insert(ContentSourceHeadRow).values(**values)
    elif dialect == "sqlite":
        statement = sqlite_insert(ContentSourceHeadRow).values(**values)
    else:
        # Keep the repository usable for SQLAlchemy-compatible test engines
        # which do not expose a dialect-specific upsert construct.
        try:
            with session.begin_nested():
                session.add(ContentSourceHeadRow(**values))
                session.flush()
            return
        except IntegrityError:
            pass
        head = session.get(ContentSourceHeadRow, source_identity_hash, with_for_update=True)
        if head is None:
            raise RuntimeError("source head disappeared after a unique-key conflict")
        if (created_at, snapshot_id) > (head.updated_at, head.latest_snapshot_id):
            head.latest_snapshot_id = snapshot_id
            head.updated_at = created_at
        return

    excluded = statement.excluded
    is_newer = or_(
        excluded.updated_at > ContentSourceHeadRow.updated_at,
        and_(
            excluded.updated_at == ContentSourceHeadRow.updated_at,
            excluded.latest_snapshot_id > ContentSourceHeadRow.latest_snapshot_id,
        ),
    )
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[ContentSourceHeadRow.source_identity_hash],
            set_={
                "latest_snapshot_id": case(
                    (is_newer, excluded.latest_snapshot_id),
                    else_=ContentSourceHeadRow.latest_snapshot_id,
                ),
                "updated_at": case(
                    (is_newer, excluded.updated_at),
                    else_=ContentSourceHeadRow.updated_at,
                ),
            },
        )
    )

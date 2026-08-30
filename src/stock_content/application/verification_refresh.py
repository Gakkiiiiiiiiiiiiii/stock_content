"""Atomic verification completion -> refresh snapshot -> signal outbox UoW."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import and_, case, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    ClaimArtifactMemberRow,
    ClaimVerificationJobRow,
    ClaimVerificationResultRow,
    ContentArtifactEdgeRow,
    ContentArtifactRow,
    ContentSnapshotArtifactRow,
    ContentSnapshotRow,
    ContentSourceHeadRow,
    FinancialClaimRow,
    SignalOutboxRow,
)
from stock_content.adapters.postgres.repositories.snapshot_repository import (
    _compare_candidate_to_existing_row,
    _compare_existing_identity,
    _validate_snapshot_artifacts,
    _validate_snapshot_candidate,
    _validate_snapshot_row,
)
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    LEASED,
    PostgresVerificationJobRepository,
    VerificationJobIntegrityError,
    persist_verification_result,
    verification_id_of,
)
from stock_content.application.signal_service import SignalService
from stock_content.domain.artifacts import (
    VerificationArtifact,
    artifact_id_of,
    canonical_json,
    deserialize_artifact,
    serialize_artifact,
)
from stock_content.domain.claims import FinancialClaim, VerificationArtifactEntry, VerificationResult
from stock_content.domain.initial_verification import verify_initial_verification_closure
from stock_content.domain.lineage import build_content_snapshot
from stock_content.domain.signal_contract import validate_signal_v4


class VerificationRefreshService:
    def __init__(
        self,
        session_factory: sessionmaker,
        jobs: PostgresVerificationJobRepository,
        claims,
        signal_service: SignalService | None = None,
    ) -> None:
        self._sessions = session_factory
        self._jobs = jobs
        self._claims = claims
        self._signals = signal_service or SignalService()

    @contextmanager
    def _transaction_lock(self, job_id: str):
        """Serialize refresh completion with initial planning/worker writes."""
        with self._sessions.begin() as session:
            hint = session.get(ClaimVerificationJobRow, job_id)
            if hint is None:
                raise KeyError("verification job not found")
            with self._jobs.coordination_lock(hint.claim_id, hint.provider, session):
                yield session

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result: VerificationResult,
        *,
        parent_snapshot_id: str | None = None,
        now: datetime | None = None,
        failure_hook: Callable[[str], None] | None = None,
    ) -> dict[str, str | bool | None]:
        if result.status in {"VERIFICATION_PENDING", "NOT_REQUIRED"}:
            raise VerificationJobIntegrityError(
                "verification refresh requires a terminal external result"
            )
        current = now or datetime.now(UTC)
        with self._transaction_lock(job_id) as session:
            job = session.get(ClaimVerificationJobRow, job_id)
            if job is None:
                raise KeyError("verification job not found")
            if result.claim_id != job.claim_id:
                raise VerificationJobIntegrityError("result claim does not match leased job")
            effective_available_at = max(
                _as_utc(result.available_at or current),
                _as_utc(current),
                _as_utc(result.verification_timestamp)
                if result.verification_timestamp is not None else _as_utc(current),
            )
            effective_result = result.model_copy(update={"available_at": effective_available_at})
            result_id = verification_id_of(job.claim_id, job.provider, effective_result)
            payload = effective_result.model_dump(mode="json")
            existing_result = session.get(ClaimVerificationResultRow, result_id)
            result_values = {
                "verification_id": result_id,
                "claim_id": job.claim_id,
                "provider": job.provider,
                "status": result.status,
                "market_snapshot_id": effective_result.market_snapshot_id,
                "market_data_version": effective_result.market_data_version,
                "result_payload": payload,
                "trace_id": job.trace_id,
                "fact_date": effective_result.fact_date,
                "adjustment": effective_result.adjustment,
                "verification_timestamp": effective_result.verification_timestamp,
                "verification_rule_version": effective_result.verification_rule_version,
                "verified_at": effective_result.verification_timestamp,
                "available_at": effective_available_at,
                "created_at": current,
            }
            if existing_result is not None and existing_result.available_at is not None:
                # A retry may arrive with a later wall-clock ``now``.  The
                # immutable result envelope is owned by the first completion;
                # reuse it so idempotent completion cannot look like a payload
                # conflict merely because the retry happened later.
                effective_available_at = _as_utc(existing_result.available_at)
                effective_result = result.model_copy(update={"available_at": effective_available_at})
                result_values["result_payload"] = effective_result.model_dump(mode="json")
                result_values["available_at"] = effective_available_at
            if existing_result is not None:
                # Validate the row even when another completion path won the
                # deterministic insert race.  If it already has a snapshot,
                # this refresh is fully idempotent; otherwise continue below
                # and materialize the missing snapshot in this UoW.
                persist_verification_result(session, result_values)
                prior = self._find_snapshot_for_result(session, result_id, job.claim_id)
                if prior:
                    job.status = effective_result.status
                    job.lease_owner = None
                    job.lease_expires_at = None
                    job.next_retry_at = None
                    return {
                        "idempotent": True,
                        "verification_id": result_id,
                        "snapshot_id": prior[0],
                        "signal_id": prior[1],
                    }
            if existing_result is None and (job.lease_owner != worker_id or job.status != LEASED):
                raise VerificationJobIntegrityError("verification job lease owner mismatch")
            self._hook(failure_hook, "result")
            persist_verification_result(session, result_values)
            claim_row = session.get(FinancialClaimRow, job.claim_id)
            if claim_row is None:
                raise KeyError(f"claim not found: {job.claim_id}")
            claim = FinancialClaim.model_validate(dict(claim_row.payload or {}))
            # Parent selection and source-head advancement must happen in this
            # transaction.  The worker's preflight resolution is useful for
            # diagnostics, but it is inherently stale by the time a provider
            # returns.  Locking the head makes two workers that started from
            # the same snapshot serialize and lets the later worker merge its
            # result into the first worker's refreshed snapshot.
            requested_parent = self._load_parent_snapshot(session, parent_snapshot_id)
            source_hash = hashlib.sha256(
                f"{requested_parent.source_type}:{requested_parent.source_ref}".encode()
            ).hexdigest()
            head = session.scalar(
                select(ContentSourceHeadRow)
                .where(ContentSourceHeadRow.source_identity_hash == source_hash)
                .with_for_update()
            )
            parent = requested_parent
            parent_is_locked_head = False
            if head is not None and head.latest_snapshot_id != requested_parent.content_snapshot_id:
                current_parent_row = session.get(ContentSnapshotRow, head.latest_snapshot_id)
                if current_parent_row is None:
                    raise VerificationJobIntegrityError("source head points to missing snapshot")
                current_parent = self._row_to_snapshot(current_parent_row, session)
                if self._snapshot_contains_claim(session, current_parent, job.claim_id):
                    parent = current_parent
                    parent_is_locked_head = True
            elif head is not None:
                parent_is_locked_head = True
            if not self._snapshot_contains_claim(session, parent, job.claim_id):
                raise VerificationJobIntegrityError(
                    "parent snapshot does not contain the leased claim"
                )
            refresh_commit_at = max(_as_utc(current), _as_utc(parent.created_at))
            old_verification_id = str((parent.artifact_ids or {}).get("verification") or "")
            parent_artifact = None
            if old_verification_id:
                old_row = session.get(ContentArtifactRow, old_verification_id)
                if old_row is not None:
                    parent_artifact = deserialize_artifact(dict(old_row.payload or {}))
            prior_results = list(getattr(parent_artifact, "results", ()) or ())
            # A refresh is a new immutable view of the whole batch.  Replace
            # only this claim's result; all sibling claims remain represented.
            replaced = False
            merged_results = []
            for prior_result in prior_results:
                if getattr(prior_result, "claim_id", None) == result.claim_id:
                    if not replaced:
                        merged_results.append(VerificationArtifactEntry.from_result(
                            effective_result, provider=job.provider, verification_id=result_id
                        ))
                        replaced = True
                else:
                    merged_results.append(prior_result)
            if not replaced:
                merged_results.append(VerificationArtifactEntry.from_result(
                    effective_result, provider=job.provider, verification_id=result_id
                ))
            verification = VerificationArtifact(
                artifact_id="verification-refresh-pending",
                artifact_type="verification",
                producer_stage="verification_refresh",
                claim_artifact_id=str(getattr(parent_artifact, "claim_artifact_id", "")),
                results=merged_results,
                parent_artifact_ids=(old_verification_id,) if old_verification_id else (),
            )
            verification = VerificationArtifact(
                **{**verification.__dict__, "artifact_id": artifact_id_of(verification)}
            )
            self._persist_artifact(session, verification)
            self._hook(failure_hook, "verification_artifact")
            artifact_ids = dict(parent.artifact_ids or {})
            artifact_ids["verification"] = verification.artifact_id
            external = list(parent.external_snapshots or parent.quant_market_snapshot_ids or ())
            if effective_result.market_snapshot_id and effective_result.market_snapshot_id not in external:
                external.append(effective_result.market_snapshot_id)
            refreshed = build_content_snapshot(
                source_type=parent.source_type,
                source_ref=parent.source_ref,
                source_content_hash=parent.source_content_hash,
                source_artifact_id=parent.source_artifact_id,
                artifact_ids=artifact_ids,
                parser_version=parent.parser_version,
                asr_model=parent.asr_model,
                asr_model_version=parent.asr_model_version,
                vision_model=parent.vision_model,
                llm_model=parent.llm_model,
                prompt_bundle_version=parent.prompt_bundle_version,
                entity_alias_version=parent.entity_alias_version,
                verification_policy_version=parent.verification_policy_version,
                quant_market_snapshot_ids=tuple(external),
                code_sha=parent.code_sha,
                config_hash=parent.config_hash,
                pipeline_version=parent.pipeline_version,
                snapshot_kind="VERIFICATION_REFRESH",
                parent_snapshot_id=parent.content_snapshot_id,
                supersedes_snapshot_id=parent.content_snapshot_id,
                producer_manifest=parent.producer_manifest,
                policy_versions=parent.policy_versions,
                model_versions=parent.model_versions,
                prompt_versions=parent.prompt_versions,
                configuration=parent.configuration,
                external_snapshots=tuple(external),
                created_at=refresh_commit_at,
            )
            self._hook(failure_hook, "snapshot")
            self._persist_snapshot(session, refreshed)
            self._hook(failure_hook, "snapshot_mapping")
            verification_view = effective_result.model_dump(mode="json") | {"provider": job.provider}
            signal = self._signals.build_signal(
                refreshed,
                claim,
                verification_view,
                verification_artifact_id=verification.artifact_id,
                trace_id=job.trace_id,
            )
            self._hook(failure_hook, "signal")
            signal_id = None
            if self._signals.policy.evaluate(claim, verification_view, snapshot=refreshed).allowed:
                # Refresh bypasses ``SignalService.enqueue_initial`` because
                # its outbox write participates in this larger transaction.
                # Validate the exact payload before adding the row so a
                # contract violation aborts the complete refresh UoW.
                validate_signal_v4(signal)
                signal_id = str(signal["signal_id"])
                self._persist_outbox(session, signal, current)
            self._hook(failure_hook, "outbox")
            job.status = effective_result.status
            job.lease_owner = None
            job.lease_expires_at = None
            job.next_retry_at = None
            # ``head`` was locked before parent validation and remains locked
            # until commit.  Never let an older completion move it backwards.
            if head is None or parent_is_locked_head:
                _upsert_source_head(
                    session,
                    source_identity_hash=source_hash,
                    snapshot_id=refreshed.content_snapshot_id,
                    verified_snapshot_id=(
                        refreshed.content_snapshot_id
                        if effective_result.status in {"VERIFIED", "PARTIALLY_VERIFIED"}
                        else None
                    ),
                    updated_at=refreshed.created_at,
                )
            return {
                "idempotent": False,
                "verification_id": result_id,
                "snapshot_id": refreshed.content_snapshot_id,
                "signal_id": signal_id,
            }

    @staticmethod
    def _hook(hook: Callable[[str], None] | None, name: str) -> None:
        if hook:
            hook(name)

    @staticmethod
    def _load_parent_snapshot(session, snapshot_id: str | None):
        if snapshot_id:
            row = session.get(ContentSnapshotRow, snapshot_id)
            if row is None:
                raise KeyError("parent content snapshot not found")
            return VerificationRefreshService._row_to_snapshot(row, session)
        raise ValueError("parent_snapshot_id is required for verification refresh")

    @staticmethod
    def _row_to_snapshot(row: ContentSnapshotRow, session=None):
        from stock_content.adapters.postgres.repositories.snapshot_repository import _row_to_snapshot

        return _row_to_snapshot(row, session)

    def resolve_parent_snapshot_id(self, claim_id: str) -> str:
        """Resolve the latest snapshot containing this claim's persisted artifact.

        This deliberately joins claim-artifact membership to snapshot-artifact
        membership; a globally newest snapshot from another source is never a
        valid parent.
        """
        with self._sessions() as session:
            rows = session.scalars(
                select(ContentSnapshotRow)
                .join(
                    ContentSnapshotArtifactRow,
                    ContentSnapshotArtifactRow.content_snapshot_id == ContentSnapshotRow.content_snapshot_id,
                )
                .join(
                    ClaimArtifactMemberRow,
                    ClaimArtifactMemberRow.artifact_id == ContentSnapshotArtifactRow.artifact_id,
                )
                .where(ClaimArtifactMemberRow.claim_id == claim_id)
                .order_by(ContentSnapshotRow.created_at.desc(), ContentSnapshotRow.content_snapshot_id.desc())
            ).all()
            unique = {row.content_snapshot_id: row for row in rows}
            if not unique:
                raise KeyError(f"no snapshot lineage for claim: {claim_id}")
            return next(iter(unique.values())).content_snapshot_id

    @staticmethod
    def _snapshot_contains_claim(session, snapshot, claim_id: str) -> bool:
        claim_artifact_id = str((snapshot.artifact_ids or {}).get("claims") or "")
        if not claim_artifact_id:
            return False
        member = session.scalar(
            select(ClaimArtifactMemberRow).where(
                ClaimArtifactMemberRow.artifact_id == claim_artifact_id,
                ClaimArtifactMemberRow.claim_id == claim_id,
            )
        )
        return member is not None

    @staticmethod
    def _find_snapshot_for_result(session, verification_id: str, claim_id: str):
        """Return (snapshot_id, signal_id) for an already-applied result."""
        from stock_content.adapters.postgres.repositories.snapshot_repository import _row_to_snapshot

        result_row = session.get(ClaimVerificationResultRow, verification_id)
        if result_row is None or result_row.claim_id != claim_id:
            return None
        expected_result = dict(result_row.result_payload or {})
        rows = session.scalars(select(ContentSnapshotRow).order_by(ContentSnapshotRow.created_at.desc())).all()
        for row in rows:
            snapshot = _row_to_snapshot(row, session)
            if not VerificationRefreshService._snapshot_contains_claim(session, snapshot, claim_id):
                continue
            verification_id_in_snapshot = str((snapshot.artifact_ids or {}).get("verification") or "")
            if not verification_id_in_snapshot:
                continue
            artifact_row = session.get(ContentArtifactRow, verification_id_in_snapshot)
            if artifact_row is None:
                continue
            payload = dict(artifact_row.payload or {})
            results = payload.get("results") or []
            if not any(
                isinstance(item, dict)
                and str(item.get("claim_id")) == claim_id
                and canonical_json({
                    key: value for key, value in item.items()
                    if key not in {"provider", "verification_id", "verification_job_id"}
                }) == canonical_json(expected_result)
                for item in results
            ):
                continue
            outbox = session.scalar(
                select(SignalOutboxRow).where(
                    SignalOutboxRow.content_snapshot_id == snapshot.content_snapshot_id,
                    SignalOutboxRow.claim_id == claim_id,
                )
            )
            return snapshot.content_snapshot_id, outbox.signal_id if outbox else None
        return None

    @staticmethod
    def _persist_artifact(session, artifact: VerificationArtifact) -> None:
        payload = json.loads(canonical_json(serialize_artifact(artifact)))
        row = session.get(ContentArtifactRow, artifact.artifact_id)
        if row is not None:
            if row.content_hash != artifact.content_hash:
                raise ValueError("verification artifact id conflict")
            return
        session.add(
            ContentArtifactRow(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                schema_version=artifact.schema_version,
                producer_stage=artifact.producer_stage,
                producer_version=artifact.producer_version,
                content_hash=artifact.content_hash,
                parent_artifact_ids=list(artifact.parent_artifact_ids),
                payload=payload,
                created_at=artifact.created_at,
            )
        )
        for parent_id in artifact.parent_artifact_ids:
            session.add(
                ContentArtifactEdgeRow(
                    edge_id=hashlib.sha256(f"{artifact.artifact_id}:{parent_id}".encode()).hexdigest(),
                    artifact_id=artifact.artifact_id,
                    parent_artifact_id=parent_id,
                )
            )

    @staticmethod
    def _persist_snapshot(session, snapshot) -> None:
        payload = _validate_snapshot_candidate(snapshot)
        _validate_snapshot_artifacts(
            session,
            artifact_ids=dict(snapshot.artifact_ids),
            schema_version=snapshot.schema_version,
            snapshot_id=snapshot.content_snapshot_id,
        )
        existing = session.get(ContentSnapshotRow, snapshot.content_snapshot_id)
        if existing is not None:
            _validate_snapshot_row(session, existing)
            existing_identity = dict(existing.identity or {})
            _compare_existing_identity(existing_identity, payload, snapshot_id=snapshot.content_snapshot_id)
            _compare_candidate_to_existing_row(snapshot, existing, existing_identity)
            return
        session.add(
            ContentSnapshotRow(
                content_snapshot_id=snapshot.content_snapshot_id,
                source_type=snapshot.source_type,
                source_ref=snapshot.source_ref,
                source_content_hash=snapshot.source_content_hash,
                identity=payload,
                artifact_ids=dict(snapshot.artifact_ids),
                quant_market_snapshot_ids=list(snapshot.quant_market_snapshot_ids),
                pipeline_version=snapshot.pipeline_version,
                schema_version=snapshot.schema_version,
                code_sha=snapshot.code_sha,
                config_hash=snapshot.config_hash,
                source_artifact_id=snapshot.source_artifact_id,
                artifact_root_hash=snapshot.artifact_root_hash,
                snapshot_kind=snapshot.snapshot_kind,
                parent_snapshot_id=snapshot.parent_snapshot_id,
                supersedes_snapshot_id=snapshot.supersedes_snapshot_id,
                producer_manifest=dict(snapshot.producer_manifest),
                created_at=snapshot.created_at,
            )
        )
        for slot, artifact_id in snapshot.artifact_ids.items():
            session.add(
                ContentSnapshotArtifactRow(
                    member_id=hashlib.sha256(
                        f"{snapshot.content_snapshot_id}:{slot}:{artifact_id}".encode()
                    ).hexdigest(),
                    content_snapshot_id=snapshot.content_snapshot_id,
                    artifact_id=artifact_id,
                    slot=slot,
                )
            )
        session.flush()
        _validate_refresh_verification_closure(session, snapshot)

    @staticmethod
    def _persist_outbox(session, payload: dict, now: datetime) -> None:
        signal_id = str(payload["signal_id"])
        existing = session.scalar(select(SignalOutboxRow).where(SignalOutboxRow.signal_id == signal_id))
        if existing is not None:
            if dict(existing.payload or {}) != payload:
                raise ValueError("signal outbox payload conflict")
            return
        session.add(
            SignalOutboxRow(
                outbox_id="outbox-" + hashlib.sha256(signal_id.encode()).hexdigest()[:32],
                signal_id=signal_id,
                content_snapshot_id=payload.get("content_snapshot_id"),
                claim_id=payload.get("claim_id"),
                schema_version=payload.get("signal_schema_version", "content-factor-signal.v4"),
                payload=payload,
                status="PENDING",
                next_attempt_at=now,
                created_at=now,
            )
        )


def _validate_refresh_verification_closure(session, snapshot) -> None:
    """Validate every exact job/result reference in a refresh artifact."""
    verification_id = str((snapshot.artifact_ids or {}).get("verification") or "")
    if not verification_id:
        return
    row = session.get(ContentArtifactRow, verification_id)
    if row is None:
        return
    artifact = deserialize_artifact(dict(row.payload or {}))
    entries = [
        item for item in (getattr(artifact, "results", ()) or ())
        if isinstance(item, VerificationArtifactEntry)
    ]
    if not entries:
        return
    job_ids = {str(item.verification_job_id) for item in entries if item.verification_job_id}
    result_ids = {str(item.verification_id) for item in entries if item.verification_id}
    jobs = {
        item.job_id: item
        for item in session.scalars(
            select(ClaimVerificationJobRow).where(ClaimVerificationJobRow.job_id.in_(job_ids))
        ).all()
    } if job_ids else {}
    results = {
        item.verification_id: item
        for item in session.scalars(
            select(ClaimVerificationResultRow).where(
                ClaimVerificationResultRow.verification_id.in_(result_ids)
            )
        ).all()
    } if result_ids else {}
    verify_initial_verification_closure(
        artifact_results=entries, jobs=jobs, results=results,
        snapshot_committed_at=snapshot.created_at,
    )


def _upsert_source_head(
    session,
    *,
    source_identity_hash: str,
    snapshot_id: str,
    verified_snapshot_id: str | None,
    updated_at: datetime,
) -> None:
    """Create/advance a source head without a first-writer race."""
    values = {
        "source_identity_hash": source_identity_hash,
        "latest_snapshot_id": snapshot_id,
        "latest_verified_snapshot_id": verified_snapshot_id,
        "updated_at": updated_at,
    }
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        statement = postgres_insert(ContentSourceHeadRow).values(**values)
    elif dialect == "sqlite":
        statement = sqlite_insert(ContentSourceHeadRow).values(**values)
    else:
        try:
            with session.begin_nested():
                session.add(ContentSourceHeadRow(**values))
                session.flush()
            return
        except IntegrityError:
            head = session.get(ContentSourceHeadRow, source_identity_hash)
            if head is None:
                raise RuntimeError("source head disappeared after a unique-key conflict")
            if _head_is_newer(updated_at, snapshot_id, head.updated_at, head.latest_snapshot_id):
                head.latest_snapshot_id = snapshot_id
                head.updated_at = updated_at
                if verified_snapshot_id is not None:
                    head.latest_verified_snapshot_id = verified_snapshot_id
            return

    excluded = statement.excluded
    is_newer = or_(
        excluded.updated_at > ContentSourceHeadRow.updated_at,
        and_(
            excluded.updated_at == ContentSourceHeadRow.updated_at,
            excluded.latest_snapshot_id > ContentSourceHeadRow.latest_snapshot_id,
        ),
    )


    verified_is_newer = and_(
        excluded.latest_verified_snapshot_id.is_not(None),
        or_(ContentSourceHeadRow.latest_verified_snapshot_id.is_(None), is_newer),
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
                "latest_verified_snapshot_id": case(
                    (verified_is_newer, excluded.latest_verified_snapshot_id),
                    else_=ContentSourceHeadRow.latest_verified_snapshot_id,
                ),
            },
        )
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _head_is_newer(
    updated_at: datetime,
    snapshot_id: str,
    existing_updated_at: datetime,
    existing_snapshot_id: str,
) -> bool:
    def normalized(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    return (normalized(updated_at), snapshot_id) > (
        normalized(existing_updated_at),
        existing_snapshot_id,
    )


__all__ = ["VerificationRefreshService"]

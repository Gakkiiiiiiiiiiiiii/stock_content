"""Verification refresh transaction orchestration."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import select

from stock_content.adapters.postgres.models import (
    ClaimVerificationJobRow,
    ClaimVerificationResultRow,
    ContentArtifactRow,
    ContentSnapshotRow,
    ContentSourceHeadRow,
    FinancialClaimRow,
)
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    LEASED,
    VerificationJobIntegrityError,
    persist_verification_result,
    verification_id_of,
)
from stock_content.application.verification.closure import _as_utc, _upsert_source_head
from stock_content.domain.artifacts import VerificationArtifact, artifact_id_of, deserialize_artifact
from stock_content.domain.claim_state_event import ClaimStateEvent
from stock_content.domain.claims import FinancialClaim, VerificationArtifactEntry, VerificationResult
from stock_content.domain.lineage import build_content_snapshot
from stock_content.domain.signal_contract import validate_signal_v4


class VerificationTransactionMixin:
    @contextmanager
    def _transaction_lock(self, job_id: str):
        """Serialize refresh completion with initial planning/worker writes."""
        with self._sessions.begin() as session:
            hint = session.get(ClaimVerificationJobRow, job_id)
            if hint is None:
                raise KeyError("verification job not found")
            with self._jobs.coordination_lock(hint.claim_id, hint.provider, session):
                yield session

    @staticmethod
    def _hook(hook: Callable[[str], None] | None, name: str) -> None:
        """Invoke an optional failure-injection hook at a UoW boundary."""
        if hook:
            hook(name)

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
            if self._claim_events is not None:
                prior_events = list(self._claim_events.list_for_claim(job.claim_id))
                previous = prior_events[-1].event_hash if prior_events else None
                self._claim_events.append_in_session(session, ClaimStateEvent(
                    claim_id=job.claim_id,
                    event_type="VERIFICATION_REFRESH",
                    payload={"status": result.status, "verification_id": result_id, "provider": job.provider},
                    known_from=effective_available_at,
                    source_available_from=effective_available_at,
                    previous_event_hash=previous,
                ))
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



__all__ = ["VerificationTransactionMixin"]

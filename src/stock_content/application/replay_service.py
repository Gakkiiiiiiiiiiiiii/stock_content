"""ContentSnapshot Replay V2 and immutable lineage verification."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from stock_content.application.pipeline import PipelineContext
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.artifacts import ArtifactRegistry
from stock_content.domain.initial_verification import TERMINAL_STATUSES
from stock_content.domain.lineage import (
    compute_artifact_root_hash,
    compute_content_snapshot_id,
    lineage_of,
    snapshot_identity_payload,
)
from stock_content.domain.models import ContentTask
from stock_content.ports.temporal_reference import (
    ExchangeCalendarRef,
    FiscalCalendarRef,
    ResolvedPeriod,
    TemporalReferenceProviderUnavailableError,
)
from stock_content.ports.temporal_reference_snapshot import (
    PinnedTemporalReferenceProvider,
    TemporalReferenceSnapshotMismatchError,
    TemporalReferenceSnapshotNotFoundError,
)


class ReplayIntegrityError(ValueError):
    """Stable, structured failure at the replay integrity boundary."""

    def __init__(self, code: str, detail: str, **fields: Any) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.fields = fields

    def to_dict(self, *, content_snapshot_id: str | None = None) -> dict[str, Any]:
        payload = {"error": self.code, "detail": self.detail}
        payload.update(self.fields)
        if content_snapshot_id is not None:
            payload.setdefault("content_snapshot_id", content_snapshot_id)
        return payload


class ReplayService:
    """Run VERIFY_LINEAGE, REPROCESS, or MIGRATION_REPLAY.

    Replay deliberately does not use task-specific values from Snapshot
    identity. The immutable task options are recovered through the durable
    checkpoint/artifact association supplied by ``SqlArtifactRepository``.
    """

    _MODES = ("VERIFY_LINEAGE", "REPROCESS", "MIGRATION_REPLAY")
    _RUNTIME_OPTIONS = frozenset(
        {
            "idempotency_key", "trace_id", "decision_id", "replay_raw_storage_uri",
            "replay_snapshot_kind", "replay_parent_snapshot_id", "replay_supersedes_snapshot_id",
            "replay_pipeline_version", "replay_lifecycle_timestamp",
            "temporal_reference_provider",
        }
    )

    def __init__(self, snapshots: SnapshotService, *, artifact_repository: Any | None = None,
                 signal_outbox: Any | None = None, task_repository: Any | None = None,
                 pipeline: Any | None = None, claim_repository: Any | None = None,
                 occurrence_repository: Any | None = None,
                 lifecycle_repository: Any | None = None,
                 verification_repository: Any | None = None,
                 temporal_reference_snapshot_provider: Any | None = None) -> None:
        self._snapshots = snapshots
        self._artifacts = artifact_repository
        self._signal_outbox = signal_outbox
        self._tasks = task_repository
        self._pipeline = pipeline
        self._claims = claim_repository
        self._occurrences = occurrence_repository
        self._lifecycle = lifecycle_repository
        self._verification = verification_repository
        self._reference_snapshots = temporal_reference_snapshot_provider

    def replay(self, content_snapshot_id: str, *, mode: str | None = None,
               pipeline_version: str | None = None,
               overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        snapshot = self._snapshots.get(content_snapshot_id)
        if snapshot is None:
            return {"error": "SNAPSHOT_NOT_FOUND", "content_snapshot_id": content_snapshot_id}
        requested = str(mode or "VERIFY_LINEAGE").upper()
        if requested == "EXACT":
            requested = "VERIFY_LINEAGE"
        if requested not in self._MODES:
            return {"error": "INVALID_REPLAY_MODE", "content_snapshot_id": content_snapshot_id,
                    "supported_modes": list(self._MODES)}
        try:
            lineage = self._verify_lineage(snapshot)
            if requested == "VERIFY_LINEAGE":
                result = {
                    "content_snapshot_id": content_snapshot_id, "mode": "VERIFY_LINEAGE",
                    "replay_mode": "EXACT" if mode is None else "VERIFY_LINEAGE",
                    "identity_match": lineage["identity_match"],
                    "recomputed_snapshot_id": lineage["recomputed_snapshot_id"],
                    "artifact_ids": dict(snapshot.artifact_ids),
                    "lineage": lineage_of(snapshot).to_dict(),
                    "artifact_validation": lineage["artifact_validation"],
                }
                if not lineage["identity_match"]:
                    result.update({"error": "REPLAY_IDENTITY_MISMATCH",
                                   "detail": "recomputed snapshot identity differs from persisted id"})
                return result
            if not lineage["identity_match"]:
                return {
                    "error": "REPLAY_IDENTITY_MISMATCH",
                    "detail": "recomputed snapshot identity differs from persisted id",
                    "content_snapshot_id": content_snapshot_id,
                    "mode": requested,
                    "recomputed_snapshot_id": lineage["recomputed_snapshot_id"],
                }
            return self._reprocess(snapshot, requested, pipeline_version, overrides)
        except ReplayIntegrityError as exc:
            return exc.to_dict(content_snapshot_id=content_snapshot_id)

    def check_current_verification_liveness(self, snapshot: Any) -> dict[str, Any]:
        """Check only the current source head, never historical closure.

        A pending historical artifact is valid once its exact job row exists.
        This opt-in check is for operational monitoring of a current head and
        reports a dangling pending job when no active work or newer refresh is
        available.
        """
        verification = self._verification
        if verification is None or self._artifacts is None:
            return {"checked": False}
        if hasattr(self._snapshots, "list_for_source"):
            siblings = self._snapshots.list_for_source(snapshot.source_type, snapshot.source_ref)
            if siblings:
                current = max(siblings, key=lambda item: (item.created_at, item.content_snapshot_id))
                if current.content_snapshot_id != snapshot.content_snapshot_id:
                    return {"checked": False, "status": "NOT_CURRENT_HEAD"}
        verification_id = str((snapshot.artifact_ids or {}).get("verification") or "")
        artifact = self._artifacts.get(verification_id) if verification_id else None
        pending = [
            item for item in (getattr(artifact, "results", ()) or ())
            if getattr(item, "status", None) == "VERIFICATION_PENDING"
        ]
        if not pending:
            return {"checked": True, "status": "NOT_PENDING"}
        for item in pending:
            job = verification.get_job(str(getattr(item, "verification_job_id", "")))
            if job is None:
                return {"checked": True, "error": "DANGLING_CURRENT_VERIFICATION_PENDING"}
            if str(job.status) in {"VERIFICATION_PENDING", "PENDING", "LEASED", "RETRYABLE"}:
                continue
            return {"checked": True, "error": "DANGLING_CURRENT_VERIFICATION_PENDING"}
        return {"checked": True, "status": "LIVE"}

    def _verify_lineage(self, snapshot: Any) -> dict[str, Any]:
        identity = snapshot_identity_payload(
            source_content_hash=snapshot.source_content_hash, pipeline_version=snapshot.pipeline_version,
            parser_version=snapshot.parser_version, asr_model=snapshot.asr_model,
            asr_model_version=snapshot.asr_model_version, vision_model=snapshot.vision_model,
            llm_model=snapshot.llm_model, prompt_bundle_version=snapshot.prompt_bundle_version,
            entity_alias_version=snapshot.entity_alias_version,
            verification_policy_version=snapshot.verification_policy_version,
            quant_market_snapshot_ids=snapshot.quant_market_snapshot_ids, code_sha=snapshot.code_sha,
            config_hash=snapshot.config_hash, source_artifact_id=snapshot.source_artifact_id,
            artifact_root_hash=snapshot.artifact_root_hash, producer_manifest=snapshot.producer_manifest,
            model_versions=snapshot.model_versions, prompt_versions=snapshot.prompt_versions,
            configuration=snapshot.configuration, external_snapshots=snapshot.external_snapshots,
            policy_versions=snapshot.policy_versions, snapshot_kind=snapshot.snapshot_kind,
            parent_snapshot_id=snapshot.parent_snapshot_id, supersedes_snapshot_id=snapshot.supersedes_snapshot_id,
        )
        recomputed = f"cs-{compute_content_snapshot_id(identity)[:32]}"
        snapshot_validation = self._verify_snapshot_ancestry(snapshot)
        self._validate_reference_closure(snapshot)
        return {"identity_match": recomputed == snapshot.content_snapshot_id,
                "recomputed_snapshot_id": recomputed,
                "artifact_validation": self._load_and_verify_artifacts(snapshot),
                "snapshot_validation": snapshot_validation}

    def _reference_records(self, snapshot: Any) -> list[dict[str, Any]]:
        manifest = dict(getattr(snapshot, "producer_manifest", {}) or {})
        records = manifest.get("reference_data") or []
        if not isinstance(records, list):
            raise ReplayIntegrityError("REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference_data is not a list")
        if any(not isinstance(item, dict) for item in records):
            raise ReplayIntegrityError(
                "REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference_data contains a non-object member"
            )
        return [dict(item) for item in records]

    def _validate_reference_closure(self, snapshot: Any) -> None:
        records = self._reference_records(snapshot)
        if not records:
            return
        if self._reference_snapshots is None:
            raise ReplayIntegrityError(
                "REPLAY_REFERENCE_SNAPSHOT_MISSING", "historical reference snapshot provider is unavailable"
            )
        for record in records:
            reference_id = str(record.get("reference_snapshot_id") or record.get("snapshot_id") or "")
            reference_type = str(record.get("reference_type") or "")
            subject_key = str(record.get("subject_key") or "")
            period_label = str(record.get("period_label") or "")
            required_fields = (
                "reference_type", "subject_key", "binding_key", "reference_snapshot_id",
                "data_version", "available_at",
            )
            missing_fields = [field for field in required_fields if not str(record.get(field) or "")]
            if reference_type == "fiscal_period" and not period_label:
                missing_fields.append("period_label")
            if missing_fields:
                raise ReplayIntegrityError(
                    "REPLAY_REFERENCE_SNAPSHOT_MISMATCH",
                    "reference pin metadata is incomplete",
                    missing_fields=sorted(set(missing_fields)),
                )
            try:
                if reference_type == "exchange_calendar":
                    value = self._reference_snapshots.get_exchange_calendar_snapshot(reference_id)
                elif reference_type == "fiscal_calendar":
                    value = self._reference_snapshots.get_fiscal_calendar_snapshot(reference_id)
                elif reference_type == "fiscal_period":
                    value = self._reference_snapshots.get_period_snapshot(
                        reference_id, subject_key=subject_key, period_label=period_label
                    )
                else:
                    raise TemporalReferenceSnapshotMismatchError(f"unknown reference type {reference_type}")
            except TemporalReferenceSnapshotNotFoundError as exc:
                raise ReplayIntegrityError(
                    "REPLAY_REFERENCE_SNAPSHOT_MISSING", str(exc), reference_snapshot_id=reference_id
                ) from exc
            except TemporalReferenceSnapshotMismatchError as exc:
                raise ReplayIntegrityError(
                    "REPLAY_REFERENCE_SNAPSHOT_MISMATCH", str(exc), reference_snapshot_id=reference_id
                ) from exc
            except TemporalReferenceProviderUnavailableError as exc:
                raise ReplayIntegrityError(
                    "REPLAY_REFERENCE_PROVIDER_UNAVAILABLE", str(exc), reference_snapshot_id=reference_id
                ) from exc
            except (KeyError, LookupError) as exc:
                raise ReplayIntegrityError(
                    "REPLAY_REFERENCE_SNAPSHOT_MISSING", str(exc), reference_snapshot_id=reference_id
                ) from exc
            except Exception as exc:  # provider protocol/transport errors fail closed
                raise ReplayIntegrityError(
                    "REPLAY_REFERENCE_SNAPSHOT_MISMATCH", str(exc), reference_snapshot_id=reference_id
                ) from exc
            if value is None:
                raise ReplayIntegrityError(
                    "REPLAY_REFERENCE_SNAPSHOT_MISSING", "reference snapshot payload is missing",
                    reference_snapshot_id=reference_id,
                )
            if str(getattr(value, "reference_snapshot_id", "")) != reference_id:
                raise ReplayIntegrityError("REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference id mismatch")
            expected_type = {
                "exchange_calendar": ExchangeCalendarRef,
                "fiscal_calendar": FiscalCalendarRef,
                "fiscal_period": ResolvedPeriod,
            }[reference_type]
            if not isinstance(value, expected_type):
                raise ReplayIntegrityError("REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference type mismatch")
            actual_subject = str(getattr(value, "subject_key", "") or "")
            if actual_subject and actual_subject != subject_key:
                raise ReplayIntegrityError("REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference subject mismatch")
            if reference_type == "fiscal_period" and str(getattr(value, "period_label", "") or "") != period_label:
                raise ReplayIntegrityError("REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference period mismatch")
            if record.get("data_version") and str(getattr(value, "data_version", "")) != str(record["data_version"]):
                raise ReplayIntegrityError("REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference data version mismatch")
            if record.get("available_at"):
                available_at = getattr(value, "available_at", None)
                try:
                    expected_available = datetime.fromisoformat(str(record["available_at"]).replace("Z", "+00:00"))
                    if expected_available.tzinfo is None:
                        expected_available = expected_available.replace(tzinfo=timezone.utc)
                    if available_at is not None and available_at.tzinfo is None:
                        available_at = available_at.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError) as exc:
                    raise ReplayIntegrityError(
                        "REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference available_at is invalid"
                    ) from exc
                if available_at is None or available_at != expected_available:
                    raise ReplayIntegrityError("REPLAY_REFERENCE_SNAPSHOT_MISMATCH", "reference available_at mismatch")
                snapshot_created = snapshot.created_at
                if snapshot_created.tzinfo is None:
                    snapshot_created = snapshot_created.replace(tzinfo=timezone.utc)
                if available_at > snapshot_created:
                    raise ReplayIntegrityError(
                        "REPLAY_REFERENCE_SNAPSHOT_MISMATCH",
                        "reference snapshot was unavailable at historical snapshot creation",
                        reference_snapshot_id=reference_id,
                    )

    def _verify_snapshot_ancestry(self, snapshot: Any) -> dict[str, Any]:
        """Validate refresh/replay snapshot parents as a finite deterministic DAG."""
        if self._snapshots is None:
            return {"checked": False, "snapshot_ids": []}
        visiting: set[str] = set()
        visited: set[str] = set()

        def walk(item: Any) -> None:
            identifier = str(item.content_snapshot_id)
            if identifier in visiting:
                raise ReplayIntegrityError("REPLAY_LINEAGE_CYCLE", f"snapshot parent cycle includes {identifier}")
            if identifier in visited:
                return
            visiting.add(identifier)
            for parent_id in (item.parent_snapshot_id, item.supersedes_snapshot_id):
                if not parent_id:
                    continue
                parent = self._snapshots.get(str(parent_id))
                if parent is None:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"snapshot {identifier} references missing parent {parent_id}")
                walk(parent)
            visiting.remove(identifier)
            visited.add(identifier)

        walk(snapshot)
        return {"checked": True, "snapshot_ids": sorted(visited)}

    def _load_and_verify_artifacts(self, snapshot: Any) -> dict[str, Any]:
        mapping = dict(snapshot.artifact_ids or {})
        if snapshot.artifact_root_hash and compute_artifact_root_hash(mapping) != snapshot.artifact_root_hash:
            raise ReplayIntegrityError("REPLAY_ARTIFACT_HASH_MISMATCH",
                                       "snapshot artifact root hash does not match artifact ids")
        if not mapping or self._artifacts is None:
            return {"checked": False, "artifact_count": len(mapping)}
        loaded: dict[str, Any] = {}
        visiting: set[str] = set()
        visited: set[str] = set()

        def walk(artifact_id: str) -> None:
            if artifact_id in visiting:
                raise ReplayIntegrityError("REPLAY_LINEAGE_CYCLE", f"artifact parent cycle includes {artifact_id}")
            if artifact_id in visited:
                return
            artifact = self._artifacts.get(artifact_id)
            if artifact is None:
                raise ReplayIntegrityError("REPLAY_ARTIFACT_MISSING", f"artifact {artifact_id} is missing",
                                           artifact_id=artifact_id)
            visiting.add(artifact_id)
            try:
                try:
                    self._artifacts.verify(artifact_id)
                except KeyError as exc:
                    raise ReplayIntegrityError("REPLAY_ARTIFACT_MISSING", f"artifact {artifact_id} is missing",
                                               artifact_id=artifact_id) from exc
                except Exception as exc:  # noqa: BLE001
                    raise ReplayIntegrityError("REPLAY_ARTIFACT_HASH_MISMATCH",
                                               f"artifact {artifact_id} failed integrity verification",
                                               artifact_id=artifact_id) from exc
                loaded[artifact_id] = artifact
                for parent_id in tuple(getattr(artifact, "parent_artifact_ids", ()) or ()):
                    walk(str(parent_id))
            finally:
                visiting.discard(artifact_id)
            visited.add(artifact_id)

        for artifact_id in mapping.values():
            walk(str(artifact_id))
        self._validate_artifact_references(snapshot, mapping, loaded)
        return {"checked": True, "artifact_count": len(loaded), "artifact_ids": sorted(loaded)}

    def _validate_artifact_references(self, snapshot: Any, mapping: dict[str, str], loaded: dict[str, Any]) -> None:
        by_type = {str(getattr(artifact, "artifact_type", "")): artifact for artifact in loaded.values()}
        source_id = str(mapping.get("source") or "")
        if snapshot.source_artifact_id and source_id != snapshot.source_artifact_id:
            raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID",
                                       "snapshot source artifact does not match artifact mapping")
        if source_id and source_id not in loaded:
            raise ReplayIntegrityError("REPLAY_ARTIFACT_MISSING", f"source artifact {source_id} is missing")
        for artifact in loaded.values():
            for field in ("source_artifact_id", "media_artifact_id", "transcript_artifact_id",
                          "evidence_artifact_id", "claim_artifact_id", "verification_artifact_id",
                          "knowledge_artifact_id", "frame_artifact_id"):
                reference = str(getattr(artifact, field, "") or "")
                if reference and reference not in loaded:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"{artifact.artifact_id}.{field} references missing {reference}",
                                               artifact_id=artifact.artifact_id, reference_id=reference)

        # A replay can legitimately load superseded artifacts through a
        # parent edge (for example an early empty evidence artifact).  The
        # snapshot slot, not dict iteration order or artifact type alone, is
        # authoritative for the active chain.
        evidence_artifact = loaded.get(str(mapping.get("evidence") or "")) or by_type.get("evidence")
        claim_artifact = loaded.get(str(mapping.get("claims") or "")) or by_type.get("claims")
        verification_artifact = loaded.get(str(mapping.get("verification") or "")) or by_type.get("verification")
        loaded_evidence_parents = {
            str(item) for item in (getattr(evidence_artifact, "parent_artifact_ids", ()) or ())
        }
        for evidence in getattr(evidence_artifact, "evidences", ()) or ():
            source_id = str(getattr(evidence, "source_artifact_id", "") or "")
            if not source_id:
                raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                           f"evidence {getattr(evidence, 'evidence_id', '')} has no source artifact")
            if source_id not in loaded:
                raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                           f"evidence references missing source artifact {source_id}")
            if source_id not in loaded_evidence_parents:
                raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID",
                                           f"evidence source {source_id} is not an EvidenceArtifact parent")
        claim_ids = {str(getattr(item, "claim_id", None) or
                          (item.get("claim_id") if isinstance(item, dict) else item))
                     for item in (getattr(claim_artifact, "claims", ()) or ())}
        if claim_artifact is not None:
            for claim in getattr(claim_artifact, "claims", ()) or ():
                persisted_claim = (
                    self._claims.get(str(claim))
                    if self._claims is not None and isinstance(claim, str)
                    else None
                )
                if isinstance(claim, str) and self._claims is not None and persisted_claim is None:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"claim artifact references missing claim {claim}")
                # Canonical claims intentionally have no source-specific
                # evidence ownership.  Evidence closure is checked below
                # through the fixed occurrence artifact and its role
                # memberships, never through persisted_claim.evidence_refs.

        # Occurrence and lifecycle artifacts contain immutable row IDs.  A
        # replay must resolve exactly those IDs; reading a latest projection
        # would allow history to change underneath an old snapshot.
        occurrence_artifact = loaded.get(str(mapping.get("occurrences") or ""))
        occurrence_ids = tuple(getattr(occurrence_artifact, "occurrence_ids", ()) or ())
        # Occurrence rows are bitemporal, immutable records.  Their artifact
        # stores only row IDs, so the active snapshot slots are the authority
        # for the artifact set against which every row reference is checked.
        # A row that happens to exist in the database must never make a
        # historical snapshot appear complete when the row's source chain is
        # outside that snapshot.
        active_source_id = str(mapping.get("source") or "")
        active_transcript_id = str(mapping.get("transcript") or "")
        active_semantic_id = str(mapping.get("semantic_segments") or "")
        active_evidence_id = str(mapping.get("evidence") or "")
        active_claim_id = str(mapping.get("claims") or "")
        if occurrence_ids:
            required_slots = {
                "source": active_source_id,
                "transcript": active_transcript_id,
                "semantic_segments": active_semantic_id,
                "evidence": active_evidence_id,
                "claims": active_claim_id,
            }
            missing_slots = [slot for slot, artifact_id in required_slots.items() if not artifact_id]
            if missing_slots:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_MISSING",
                    "occurrence closure is missing active snapshot artifact slots",
                    missing_slots=missing_slots,
                )
            active_artifacts = {
                slot: loaded.get(artifact_id)
                for slot, artifact_id in required_slots.items()
            }
            missing_artifacts = [slot for slot, artifact in active_artifacts.items() if artifact is None]
            if missing_artifacts:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_MISSING",
                    "occurrence closure is missing active snapshot artifacts",
                    missing_slots=missing_artifacts,
                )
            expected_types = {
                "source": "source",
                "transcript": "transcript",
                "semantic_segments": "semantic_segments",
                "evidence": "evidence",
                "claims": "claims",
            }
            for slot, expected_type in expected_types.items():
                actual_type = str(getattr(active_artifacts[slot], "artifact_type", ""))
                if actual_type != expected_type:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        f"snapshot {slot} slot points to {actual_type or 'unknown'} artifact",
                        artifact_id=required_slots[slot],
                        expected_type=expected_type,
                    )
            semantic_transcript_id = str(
                getattr(active_artifacts["semantic_segments"], "transcript_artifact_id", "")
            )
            if semantic_transcript_id != active_transcript_id:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "semantic segment artifact does not belong to snapshot transcript",
                )
            if str(getattr(active_artifacts["evidence"], "transcript_artifact_id", "")) not in {
                "", active_transcript_id
            }:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "evidence artifact does not belong to snapshot transcript",
                )
            occurrence_semantic_id = str(
                getattr(occurrence_artifact, "semantic_segment_artifact_id", "") or ""
            )
            if occurrence_semantic_id and occurrence_semantic_id != active_semantic_id:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "occurrence artifact references a semantic artifact outside the snapshot",
                    artifact_id=occurrence_semantic_id,
                )
            occurrence_evidence_id = str(
                getattr(occurrence_artifact, "evidence_artifact_id", "") or ""
            )
            if occurrence_evidence_id and occurrence_evidence_id != active_evidence_id:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "occurrence artifact references an evidence artifact outside the snapshot",
                    artifact_id=occurrence_evidence_id,
                )
        semantic_segment_ids = {
            str(getattr(item, "semantic_segment_id", None) or
                (item.get("semantic_segment_id") if isinstance(item, dict) else ""))
            for item in (getattr(loaded.get(active_semantic_id), "segments", ()) or ())
        }
        active_evidence_ids = {
            str(getattr(item, "evidence_id", None) or
                (item.get("evidence_id") if isinstance(item, dict) else ""))
            for item in (getattr(loaded.get(active_evidence_id), "evidences", ()) or ())
        }
        if occurrence_ids and self._occurrences is None:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_MISSING",
                "occurrence repository is unavailable for snapshot closure",
            )
        occurrence_rows = {}
        for occurrence_id in occurrence_ids:
            occurrence = self._occurrences.get(str(occurrence_id))
            if occurrence is None:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_MISSING",
                    f"occurrence row {occurrence_id} is missing",
                )
            if str(getattr(occurrence, "occurrence_id", "")) != str(occurrence_id):
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    f"occurrence row id does not match artifact: {occurrence_id}",
                )
            occurrence_claim_id = str(getattr(occurrence, "claim_id", ""))
            if occurrence_claim_id not in claim_ids:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    f"occurrence {occurrence_id} references a claim outside the snapshot",
                )
            if str(getattr(occurrence, "source_artifact_id", "")) != active_source_id:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    f"occurrence {occurrence_id} references a source outside the snapshot",
                )
            if str(getattr(occurrence, "transcript_artifact_id", "")) != active_transcript_id:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    f"occurrence {occurrence_id} references a transcript outside the snapshot",
                )
            semantic_segment_id = str(getattr(occurrence, "semantic_segment_id", ""))
            if semantic_segment_id not in semantic_segment_ids:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    f"occurrence {occurrence_id} references a semantic segment outside the snapshot",
                )
            for role, refs in (
                ("primary", getattr(occurrence, "evidence_refs", ()) or ()),
                ("condition", getattr(occurrence, "condition_evidence_refs", ()) or ()),
                ("invalidation", getattr(occurrence, "invalidation_evidence_refs", ()) or ()),
                ("temporal", getattr(occurrence, "temporal_evidence_refs", ()) or ()),
            ):
                outside = sorted({str(ref) for ref in refs} - active_evidence_ids)
                if outside:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        f"occurrence {occurrence_id} {role} evidence is outside the snapshot",
                        evidence_ids=outside,
                    )
            occurrence_rows[str(occurrence_id)] = occurrence

        lifecycle_artifact = loaded.get(str(mapping.get("lifecycle") or ""))
        lifecycle_ids = (
            tuple(getattr(lifecycle_artifact, "claim_lifecycle_event_ids", ()) or ())
            + tuple(getattr(lifecycle_artifact, "occurrence_lifecycle_event_ids", ()) or ())
        )
        if lifecycle_ids and self._lifecycle is None:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_MISSING",
                "lifecycle repository is unavailable for snapshot closure",
            )
        lifecycle_groups = (
            ("CLAIM", tuple(getattr(lifecycle_artifact, "claim_lifecycle_event_ids", ()) or ())),
            ("OCCURRENCE", tuple(getattr(lifecycle_artifact, "occurrence_lifecycle_event_ids", ()) or ())),
        )
        for expected_target_type, event_ids in lifecycle_groups:
            for event_id in event_ids:
                event = self._lifecycle.get(str(event_id))
                if event is None:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_MISSING",
                        f"lifecycle event row {event_id} is missing",
                    )
                if str(getattr(event, "lifecycle_event_id", "")) != str(event_id):
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        f"lifecycle event id does not match artifact: {event_id}",
                    )
                target_id = str(getattr(event, "target_id", ""))
                target_type = getattr(
                    getattr(event, "target_type", None), "value", getattr(event, "target_type", "")
                )
                if str(target_type) != expected_target_type:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        f"lifecycle event {event_id} target_type does not match its artifact membership",
                        expected_target_type=expected_target_type,
                        actual_target_type=str(target_type),
                    )
                valid_target_ids = claim_ids if expected_target_type == "CLAIM" else set(occurrence_rows)
                if not target_id or target_id not in valid_target_ids:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        f"lifecycle event {event_id} targets an object outside the snapshot",
                    )
        if verification_artifact is not None:
            for result in getattr(verification_artifact, "results", ()) or ():
                claim_id = str(getattr(result, "claim_id", None) or
                               (result.get("claim_id") if isinstance(result, dict) else ""))
                if claim_id and claim_id not in claim_ids:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"verification references missing claim {claim_id}")
            self._validate_verification_closure(snapshot, verification_artifact)
        if self._signal_outbox is not None and hasattr(self._signal_outbox, "list_for_snapshot"):
            for row in self._signal_outbox.list_for_snapshot(snapshot.content_snapshot_id):
                payload = dict(getattr(row, "payload", None) or {})
                if payload.get("content_snapshot_id") != snapshot.content_snapshot_id:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", "signal snapshot reference mismatch")
                claim_id = str(payload.get("claim_id") or "")
                if claim_id and claim_id not in claim_ids:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"signal references missing claim {claim_id}")
                verification_id = str(payload.get("verification_artifact_id") or "")
                if verification_id and verification_id not in loaded:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"signal references missing verification {verification_id}")
                if verification_id and verification_id != str(mapping.get("verification") or ""):
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID", "signal verification artifact mismatch"
                    )

    def _validate_verification_closure(self, snapshot: Any, verification_artifact: Any) -> None:
        """Validate fixed historical Job/Result references without latest reads."""
        entries = [
            item for item in (getattr(verification_artifact, "results", ()) or ())
            if hasattr(item, "provider")
        ]
        if not entries:
            return  # legacy bare VerificationResult artifact
        if self._verification is None:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_MISSING",
                "verification repository is unavailable for exact closure",
            )
        candidate = snapshot.created_at
        if candidate.tzinfo is None:
            from datetime import UTC
            candidate = candidate.replace(tzinfo=UTC)
        for entry in entries:
            if entry.status == "VERIFICATION_PENDING":
                job_id = str(entry.verification_job_id or "")
                job = self._verification.get_job(job_id) if job_id else None
                if job is None:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_MISSING",
                        f"verification job row {job_id} is missing",
                    )
                created_at = job.created_at
                if created_at is None:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", f"job {job_id} has no created_at")
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=candidate.tzinfo)
                if str(job.claim_id) != str(entry.claim_id) or str(job.provider) != str(entry.provider):
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", f"job {job_id} lineage mismatch")
                if created_at > candidate:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        f"job {job_id} was created after snapshot",
                    )
                # Intentionally do not inspect job.status: current execution
                # state is not part of historical Snapshot truth.
                continue
            verification_id = str(entry.verification_id or "")
            row = self._verification.get_result(verification_id) if verification_id else None
            if row is None:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_MISSING",
                    f"verification result row {verification_id} is missing",
                )
            if (
                str(row.claim_id) != str(entry.claim_id)
                or str(row.provider) != str(entry.provider)
                or str(row.status) != str(entry.status)
                or str(row.status) not in TERMINAL_STATUSES
                or (
                    entry.result is not None
                    and dict(row.result_payload or {}) != entry.result.model_dump(mode="json")
                )
            ):
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID", f"verification result {verification_id} lineage mismatch"
                )
            available_at = row.available_at
            if available_at is None:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID", f"verification result {verification_id} has no available_at"
                )
            if available_at.tzinfo is None:
                available_at = available_at.replace(tzinfo=candidate.tzinfo)
            if available_at > candidate:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID", f"verification result {verification_id} is future-dated"
                )

    def _task_options(self, snapshot: Any) -> dict[str, Any] | None:
        if self._artifacts is not None and hasattr(self._artifacts, "find_task_options_for_snapshot"):
            options = self._artifacts.find_task_options_for_snapshot(dict(snapshot.artifact_ids or {}))
            if options is not None:
                return dict(options)
        if self._tasks is not None and hasattr(self._tasks, "find_options_for_snapshot"):
            options = self._tasks.find_options_for_snapshot(snapshot.content_snapshot_id)
            if options is not None:
                return dict(options)
        return None

    def _reprocess(self, snapshot: Any, mode: str, pipeline_version: str | None,
                   overrides: dict[str, Any] | None) -> dict[str, Any]:
        if self._pipeline is None or self._artifacts is None:
            return {"error": "REPLAY_UNAVAILABLE", "mode": mode,
                    "source_snapshot_id": snapshot.content_snapshot_id}
        task_id: str | None = None
        try:
            self._load_and_verify_artifacts(snapshot)
            source_id = snapshot.source_artifact_id or (snapshot.artifact_ids or {}).get("source")
            source = self._artifacts.get(source_id) if source_id else None
            if source is None:
                raise ReplayIntegrityError("REPLAY_ARTIFACT_MISSING", "source artifact is unavailable",
                                           artifact_id=source_id)
            options = self._task_options(snapshot)
            if options is None:
                raise ReplayIntegrityError("REPLAY_INPUT_UNAVAILABLE",
                                           "immutable task options are unavailable for this snapshot")
            options = {key: value for key, value in options.items() if key not in self._RUNTIME_OPTIONS}
            options.update(dict(overrides or {}))
            records = self._reference_records(snapshot)
            if records:
                pins = {
                    str(item.get("binding_key") or (
                        f"{item.get('reference_type')}|{item.get('subject_key', '')}|{item.get('period_label', '')}"
                    )): str(item.get("reference_snapshot_id") or item.get("snapshot_id") or "")
                    for item in records
                }
                options["temporal_reference_provider"] = PinnedTemporalReferenceProvider(
                    self._reference_snapshots, pins
                )
            # Fixture extraction derives claim/knowledge availability from the
            # clock when no explicit timestamp was supplied.  Reuse the
            # immutable persisted claim timestamp so the golden replay has an
            # exact input, while still keeping that task-specific value out of
            # Snapshot identity.
            if "available_from" not in options and "as_of" not in options and self._claims is not None:
                claims_artifact = self._artifacts.get(str((snapshot.artifact_ids or {}).get("claims") or ""))
                claim_ids = list(getattr(claims_artifact, "claims", ()) or ())
                if claim_ids:
                    original_claim = self._claims.get(str(claim_ids[0]))
                    timestamp = getattr(original_claim, "fact_time", None) if original_claim is not None else None
                    if timestamp is not None:
                        options["available_from"] = (
                            timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
                        )
                        # Lifecycle projection has a deterministic transcript
                        # boundary clock; do not let this compatibility claim
                        # timestamp alter replay identity.
                        options["replay_lifecycle_timestamp"] = "derive_transcript_boundary"
            uri = str(getattr(source, "raw_storage_uri", "") or "")
            if uri and not uri.startswith("fixture://"):
                options["replay_raw_storage_uri"] = uri
                options["replay_expected_raw_hash"] = str(getattr(source, "raw_content_hash", "") or "")
            elif not (options.get("offline_fixture") or "transcript" in options or "segments" in options):
                raise ReplayIntegrityError("REPLAY_INPUT_UNAVAILABLE", "source raw media is not durably available")
            options["replay_snapshot_kind"] = "MIGRATION" if mode == "MIGRATION_REPLAY" else "REPROCESS"
            options["replay_parent_snapshot_id"] = snapshot.content_snapshot_id
            options["replay_supersedes_snapshot_id"] = snapshot.content_snapshot_id
            if pipeline_version:
                options["replay_pipeline_version"] = pipeline_version
            elif mode == "MIGRATION_REPLAY":
                raise ReplayIntegrityError("INVALID_REPLAY_REQUEST", "MIGRATION_REPLAY requires pipeline_version")
            task_id = f"replay-{uuid4().hex}"
            if self._tasks is not None and hasattr(self._tasks, "create"):
                persisted_options = {
                    key: value for key, value in options.items() if key not in self._RUNTIME_OPTIONS
                }
                self._tasks.create(ContentTask(task_id=task_id, source_type=snapshot.source_type,
                                               source_ref=snapshot.source_ref, options=persisted_options,
                                               status="RUNNING", max_retries=1))
            context = PipelineContext(task_id=task_id,
                                      source={"type": snapshot.source_type, "ref": snapshot.source_ref},
                                      options=options)
            result = self._pipeline.process(context)
            candidate_id = result.state.content_snapshot_id
            candidate = self._snapshots.get(candidate_id) if candidate_id else None
            comparison = self._compare_artifacts(snapshot, candidate, result.artifacts)
            differences = [item for item in comparison.values() if item.get("status") != "MATCH"]
            payload = {"replay_id": task_id, "mode": mode,
                       "source_snapshot_id": snapshot.content_snapshot_id,
                       "candidate_snapshot_id": candidate_id, "comparison": comparison,
                       "differences": differences}
            if differences:
                payload.update({"error": "REPLAY_NONDETERMINISTIC",
                                "detail": "reprocessed artifact hashes differ from the source snapshot"})
                if self._tasks is not None and hasattr(self._tasks, "fail"):
                    self._tasks.fail(task_id, "replay", "REPLAY_NONDETERMINISTIC: artifact comparison differs")
            elif self._tasks is not None and hasattr(self._tasks, "succeed"):
                self._tasks.succeed(task_id, {"content_snapshot_id": candidate_id, "replay": payload})
            return payload
        except ReplayIntegrityError as replay_error:
            if task_id is not None and self._tasks is not None and hasattr(self._tasks, "fail"):
                self._tasks.fail(task_id, "replay", "REPLAY_FAILED: integrity/input validation failed")
            if task_id is not None:
                # Preserve the durable task handle in the structured API
                # result so callers can audit the terminal FAILED row.
                replay_error.fields.setdefault("replay_id", task_id)
            raise
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            code = "REPLAY_INPUT_UNAVAILABLE" if "REPLAY_INPUT_UNAVAILABLE" in message else "REPLAY_FAILED"
            if "missing" in message.lower() and "raw" in message.lower():
                code = "REPLAY_INPUT_UNAVAILABLE"
            if task_id is not None and self._tasks is not None and hasattr(self._tasks, "fail"):
                self._tasks.fail(task_id, "replay", f"{code}: {message}")
            raise ReplayIntegrityError(code, message, replay_id=task_id) from exc

    def _compare_artifacts(self, old: Any, candidate: Any, registry: ArtifactRegistry) -> dict[str, dict[str, Any]]:
        old_ids = dict(old.artifact_ids or {})
        new_ids = dict(candidate.artifact_ids or {}) if candidate is not None else registry.artifact_ids()
        result: dict[str, dict[str, Any]] = {}
        for slot in sorted(set(old_ids) | set(new_ids)):
            old_id, new_id = old_ids.get(slot), new_ids.get(slot)
            old_hash = self._artifact_hash(old_id)
            new_hash = self._artifact_hash(new_id) or self._registry_hash(registry, new_id)
            status = "MATCH" if old_hash and new_hash and old_hash == new_hash else "DIFFERENT"
            result[slot] = {"status": status, "old": old_id, "new": new_id,
                            "old_hash": old_hash, "new_hash": new_hash}
        return result

    def _artifact_hash(self, artifact_id: str | None) -> str | None:
        if not artifact_id or self._artifacts is None:
            return None
        artifact = self._artifacts.get(artifact_id)
        return str(artifact.content_hash) if artifact is not None else None

    @staticmethod
    def _registry_hash(registry: ArtifactRegistry, artifact_id: str | None) -> str | None:
        if not artifact_id:
            return None
        return next((str(item.content_hash) for item in registry.artifacts() if item.artifact_id == artifact_id), None)


__all__ = ["ReplayIntegrityError", "ReplayService"]

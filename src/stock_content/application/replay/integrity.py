"""Replay integrity validation helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from stock_content.application.replay.errors import ReplayIntegrityError
from stock_content.domain.lineage import (
    compute_artifact_root_hash,
    compute_content_snapshot_id,
    snapshot_identity_payload,
)
from stock_content.ports.temporal_reference import (
    ExchangeCalendarRef,
    FiscalCalendarRef,
    ResolvedPeriod,
    TemporalReferenceProviderUnavailableError,
)
from stock_content.ports.temporal_reference_snapshot import (
    TemporalReferenceSnapshotMismatchError,
    TemporalReferenceSnapshotNotFoundError,
)


class ReplayIntegrityMixin:
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
        evidence_by_id = {
            str(getattr(item, "evidence_id", "") or ""): item
            for item in (getattr(evidence_artifact, "evidences", ()) or ())
        }
        occurrence_scope_artifact = loaded.get(str(mapping.get("occurrences") or ""))
        requires_occurrence_visual_scope = bool(
            tuple(getattr(occurrence_scope_artifact, "occurrence_ids", ()) or ())
        )
        deferred_visual_evidence_ids: set[str] = set()
        for evidence in getattr(evidence_artifact, "evidences", ()) or ():
            source_id = str(getattr(evidence, "source_artifact_id", "") or "")
            if not source_id:
                raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                           f"evidence {getattr(evidence, 'evidence_id', '')} has no source artifact")
            if source_id not in loaded:
                raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                           f"evidence references missing source artifact {source_id}")
            if (
                requires_occurrence_visual_scope
                and str(getattr(evidence, "source_type", "") or "").upper() in {"FRAME", "OCR", "VISION"}
            ):
                # Visual evidence is occurrence-owned.  Scope is unavailable
                # until immutable occurrence rows are loaded below, so defer
                # its admission rather than accepting a global frame relation.
                deferred_visual_evidence_ids.add(str(getattr(evidence, "evidence_id", "") or ""))
            elif source_id not in loaded_evidence_parents:
                self._validate_admitted_visual_evidence_source(
                    evidence=evidence,
                    evidence_artifact=evidence_artifact,
                    source_artifact=loaded[source_id],
                    loaded=loaded,
                    occurrence=None,
                )
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
                ("secondary", getattr(occurrence, "secondary_evidence_refs", ()) or ()),
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

        admitted_secondary_visual_ids: set[str] = set()
        for occurrence in occurrence_rows.values():
            for evidence_id in getattr(occurrence, "secondary_evidence_refs", ()) or ():
                evidence = evidence_by_id.get(str(evidence_id))
                if evidence is None:
                    continue
                if str(getattr(evidence, "source_type", "") or "").upper() not in {"FRAME", "OCR", "VISION"}:
                    continue
                source_id = str(getattr(evidence, "source_artifact_id", "") or "")
                source_artifact = loaded.get(source_id)
                if source_artifact is None:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_MISSING",
                        "secondary visual evidence source is missing from snapshot",
                        artifact_id=source_id,
                    )
                self._validate_admitted_visual_evidence_source(
                    evidence=evidence,
                    evidence_artifact=evidence_artifact,
                    source_artifact=source_artifact,
                    loaded=loaded,
                    occurrence=occurrence,
                )
                admitted_secondary_visual_ids.add(str(evidence_id))
        if deferred_visual_evidence_ids - admitted_secondary_visual_ids:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                "visual evidence is not owned by an occurrence secondary relation",
                evidence_ids=sorted(deferred_visual_evidence_ids - admitted_secondary_visual_ids),
            )

        # New producer snapshots seal the complete per-occurrence visual
        # packet independently of the mutable query projection.  A historical
        # snapshot legitimately has no such slot; never synthesize one during
        # replay.  When present, however, every byte must still agree with the
        # immutable occurrence rows and every referenced visual artifact must
        # be in this artifact's parent closure.
        visual_packet_artifact_id = str(mapping.get("knowledge_visual_evidence") or "")
        if visual_packet_artifact_id:
            visual_packet_artifact = loaded.get(visual_packet_artifact_id)
            if visual_packet_artifact is None or str(
                getattr(visual_packet_artifact, "artifact_type", "")
            ) != "knowledge_visual_evidence":
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "knowledge visual evidence slot does not resolve to its artifact",
                    artifact_id=visual_packet_artifact_id,
                )
            actual_packets = list(getattr(visual_packet_artifact, "occurrence_packets", ()) or ())
            expected_packets = [
                dict((getattr(occurrence, "provenance", {}) or {}).get("visual_evidence") or {})
                for occurrence in occurrence_rows.values()
            ]
            if actual_packets != expected_packets:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "sealed knowledge visual evidence differs from occurrence projection",
                    artifact_id=visual_packet_artifact_id,
                )
            visual_parents = {
                str(parent_id)
                for parent_id in (getattr(visual_packet_artifact, "parent_artifact_ids", ()) or ())
            }
            crosscheck_artifact = loaded.get(str(mapping.get("transcript_visual_crosscheck") or ""))
            for packet in actual_packets:
                occurrence = (
                    occurrence_rows.get(str(packet.get("occurrence_id") or ""))
                    if isinstance(packet, dict)
                    else None
                )
                for window in list(packet.get("windows") or ()) if isinstance(packet, dict) else ():
                    for frame in list(window.get("frames") or ()) if isinstance(window, dict) else ():
                        self._validate_visual_packet_frame(
                            packet=packet,
                            frame=frame,
                            evidence_window_id=str(window.get("evidence_window_id") or ""),
                            window_status=str(window.get("status") or ""),
                            window_reason=str(window.get("reason") or ""),
                            semantic_segment_id=str(getattr(occurrence, "semantic_segment_id", "") or ""),
                            crosscheck_artifact=crosscheck_artifact,
                            visual_parent_ids=visual_parents,
                            loaded=loaded,
                            artifact_id=visual_packet_artifact_id,
                        )

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
        self._validate_security_entity_alignment(mapping, loaded)

    @staticmethod
    def _validate_security_entity_alignment(mapping: dict[str, str], loaded: dict[str, Any]) -> None:
        """Validate audit-only mentions against the sealed owning window."""
        alignment_id = str(mapping.get("security_entity_alignment") or "")
        if not alignment_id:
            return  # Historical snapshots predate this additive slot.
        alignment = loaded.get(alignment_id)
        if alignment is None or getattr(alignment, "artifact_type", "") != "security_entity_alignment":
            raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING", "security alignment artifact missing")
        transcript_id = str(getattr(alignment, "transcript_artifact_id", "") or "")
        crosscheck_id = str(getattr(alignment, "crosscheck_artifact_id", "") or "")
        transcript = loaded.get(transcript_id)
        crosscheck = loaded.get(crosscheck_id)
        parents = set(getattr(alignment, "parent_artifact_ids", ()) or ())
        if (
            transcript_id != str(mapping.get("transcript") or "")
            or crosscheck_id != str(mapping.get("transcript_visual_crosscheck") or "")
            or transcript is None or crosscheck is None
            or getattr(transcript, "artifact_type", "") != "transcript"
            or getattr(crosscheck, "artifact_type", "") != "transcript_visual_crosscheck"
            or not {transcript_id, crosscheck_id} <= parents
        ):
            raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", "security alignment root closure invalid")
        segments = {str(item.segment_id): item for item in getattr(transcript, "segments", ()) or ()}
        records = (*getattr(alignment, "security_mentions", ()),
                   *getattr(alignment, "displayed_target_candidates", ()))
        for record in records:
            kind = str(getattr(record, "record_type", ""))
            segment_id = str(getattr(record, "asr_segment_id", "") or "")
            if kind in {"ASR_MENTION", "ALIGNED_MENTION"}:
                segment = segments.get(segment_id)
                source_id = str(getattr(record, "source_artifact_id", "") or "")
                if segment is None or source_id not in {
                    transcript_id, str(getattr(segment, "source_artifact_id", "") or "")
                }:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", "ASR mention source mismatch")
            if kind == "ASR_MENTION":
                continue
            frame_artifact_id = str(getattr(record, "frame_artifact_id", "") or "")
            ocr_artifact_id = str(getattr(record, "ocr_artifact_id", "") or "")
            frame = loaded.get(frame_artifact_id)
            ocr = loaded.get(ocr_artifact_id) if ocr_artifact_id else None
            window_id = str(getattr(record, "evidence_window_id", "") or "")
            visual_refs = {
                frame_artifact_id, ocr_artifact_id, str(getattr(record, "vision_artifact_id", "") or "")
            } - {""}
            if (
                frame is None or not visual_refs <= parents
                or getattr(frame, "artifact_type", "") != "frame"
                or getattr(frame, "frame_id", "") != getattr(record, "frame_id", "")
                or getattr(frame, "timestamp_ms", None) != getattr(record, "timestamp_ms", None)
                or getattr(frame, "image_hash", "") != getattr(record, "image_hash", "")
                or window_id not in set(getattr(frame, "evidence_window_ids", ()) or ())
                or (ocr_artifact_id and (
                    ocr is None or getattr(ocr, "artifact_type", "") != "ocr"
                    or getattr(ocr, "frame_artifact_id", "") != frame_artifact_id
                    or getattr(ocr, "frame_id", "") != getattr(record, "frame_id", "")
                    or window_id not in set(getattr(ocr, "evidence_window_ids", ()) or ())
                ))
            ):
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID", "displayed mention window/frame/OCR mismatch"
                )
            vision_id = str(getattr(record, "vision_artifact_id", "") or "")
            if vision_id:
                vision = loaded.get(vision_id)
                if (vision is None or vision_id not in parents
                    or getattr(vision, "frame_artifact_id", "") != frame_artifact_id
                    or window_id not in set(getattr(vision, "evidence_window_ids", ()) or ())):
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_INVALID", "displayed mention vision mismatch")
            if kind == "ALIGNED_MENTION":
                scoped = [
                    relation for relation in (getattr(crosscheck, "relations", ()) or ())
                    if str(getattr(relation, "frame_id", "") or "") == str(getattr(record, "frame_id", "") or "")
                    and tuple(getattr(relation, "evidence_window_ids", ()) or ()) == (window_id,)
                ]
                if len(scoped) != 1 or segment_id not in set(getattr(scoped[0], "transcript_segment_ids", ()) or ()):
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID", "aligned mention transcript outside sealed window"
                    )

    @staticmethod
    def _validate_visual_packet_frame(*, packet: Any, frame: Any, evidence_window_id: str,
                                      semantic_segment_id: str, crosscheck_artifact: Any,
                                      visual_parent_ids: set[str], loaded: dict[str, Any], artifact_id: str,
                                      window_status: str = "", window_reason: str = "") -> None:
        if not isinstance(packet, dict) or not isinstance(frame, dict):
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                "knowledge visual evidence packet is malformed",
                artifact_id=artifact_id,
            )
        frame_artifact_id = str(frame.get("frame_artifact_id") or "")
        frame_artifact = loaded.get(frame_artifact_id)
        if not frame_artifact_id or frame_artifact_id not in visual_parent_ids or frame_artifact is None:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_MISSING", "visual packet frame is outside sealed parent closure",
                artifact_id=artifact_id, frame_artifact_id=frame_artifact_id,
            )
        expected_frame_hash = f"sha256:{getattr(frame_artifact, 'content_hash', '')}"
        if frame.get("frame_artifact_hash") != expected_frame_hash:
            raise ReplayIntegrityError(
                "REPLAY_ARTIFACT_HASH_MISMATCH", "visual packet frame hash does not match artifact",
                artifact_id=artifact_id, frame_artifact_id=frame_artifact_id,
            )
        if (
            str(frame.get("frame_id") or "") != str(getattr(frame_artifact, "frame_id", "") or "")
            or frame.get("timestamp_ms") != getattr(frame_artifact, "timestamp_ms", None)
            or str(frame.get("image_hash") or "") != str(getattr(frame_artifact, "image_hash", "") or "")
            or evidence_window_id not in set(getattr(frame_artifact, "evidence_window_ids", ()) or ())
        ):
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID", "visual packet frame fields do not match sealed artifact",
                artifact_id=artifact_id, frame_artifact_id=frame_artifact_id,
            )
        matching_relations = [
            relation for relation in (getattr(crosscheck_artifact, "relations", ()) or ())
            if str(getattr(relation, "frame_id", "") or "") == str(frame.get("frame_id") or "")
            and tuple(getattr(relation, "evidence_window_ids", ()) or ()) == (evidence_window_id,)
            and tuple(getattr(relation, "semantic_segment_ids", ()) or ()) == (semantic_segment_id,)
        ]
        # A GAP can legitimately be sealed precisely because no scoped
        # crosscheck exists (or because several conflicting scoped checks do).
        # Such a frame remains UNKNOWN and must never be replay-promoted into
        # support. All other packets still require one exact scoped relation.
        packet_status = str(packet.get("status") or "")
        mixed_human_review = packet_status == "HUMAN_REVIEW_REQUIRED" and any(
            isinstance(window, dict) and window.get("status") == "HUMAN_REVIEW_REQUIRED"
            for window in packet.get("windows") or ()
        )
        unresolved_gap = (
            window_status == "GAP"
            and (packet_status == "GAP" or mixed_human_review)
            and str(frame.get("relation") or "") == "UNKNOWN"
            and (
                (window_reason == "CROSSCHECK_MISSING" and len(matching_relations) == 0)
                or (window_reason == "CROSSCHECK_SCOPE_AMBIGUOUS" and len(matching_relations) != 1)
            )
        )
        if not unresolved_gap and (
            len(matching_relations) != 1
            or str(getattr(matching_relations[0], "relation", "")) != str(frame.get("relation") or "")
        ):
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID", "visual packet relation is not uniquely sealed for its scope",
                artifact_id=artifact_id,
            )
        for modality in ("ocr", "vision"):
            for result in list(frame.get(modality) or ()):
                if not isinstance(result, dict):
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        "visual packet modality result is malformed",
                        artifact_id=artifact_id,
                    )
                result_artifact_id = str(result.get("artifact_id") or "")
                result_artifact = loaded.get(result_artifact_id)
                if not result_artifact_id or result_artifact_id not in visual_parent_ids or result_artifact is None:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_MISSING", "visual packet modality is outside sealed parent closure",
                        artifact_id=artifact_id, result_artifact_id=result_artifact_id,
                    )
                if result.get("artifact_hash") != f"sha256:{getattr(result_artifact, 'content_hash', '')}":
                    raise ReplayIntegrityError(
                        "REPLAY_ARTIFACT_HASH_MISMATCH", "visual packet modality hash does not match artifact",
                        artifact_id=artifact_id, result_artifact_id=result_artifact_id,
                    )
                expected_type = modality
                if str(getattr(result_artifact, "artifact_type", "")) != expected_type:
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID", "visual packet modality type does not match artifact",
                        artifact_id=artifact_id, result_artifact_id=result_artifact_id,
                    )
                if modality == "ocr":
                    summary = str(getattr(result_artifact, "text", "") or "")
                    model_name = getattr(result_artifact, "engine", "")
                    model_version = getattr(result_artifact, "engine_version", "")
                else:
                    label = getattr(result_artifact, "label", "")
                    labels = getattr(result_artifact, "labels", ()) or ()
                    summary = str(label or " ".join(labels))
                    model_name = getattr(result_artifact, "model_name", "")
                    model_version = getattr(result_artifact, "model_version", "")
                if (
                    str(getattr(result_artifact, "frame_artifact_id", "") or "") != frame_artifact_id
                    or str(getattr(result_artifact, "frame_id", "") or "") != str(frame.get("frame_id") or "")
                    or getattr(result_artifact, "timestamp_ms", None) != frame.get("timestamp_ms")
                    or str(getattr(result_artifact, "image_hash", "") or "") != str(frame.get("image_hash") or "")
                    or str(result.get("summary") or "") != summary
                    or result.get("confidence") != getattr(result_artifact, "confidence_score", None)
                    or dict(result.get("model") or {}).get("name") != model_name
                    or dict(result.get("model") or {}).get("version") != model_version
                ):
                    raise ReplayIntegrityError(
                        "REPLAY_LINEAGE_REFERENCE_INVALID",
                        "visual packet modality fields do not match sealed artifact",
                        artifact_id=artifact_id, result_artifact_id=result_artifact_id,
                    )

    @staticmethod
    def _validate_admitted_visual_evidence_source(
        *, evidence: Any, evidence_artifact: Any, source_artifact: Any,
        loaded: dict[str, Any], occurrence: Any | None,
    ) -> None:
        """Accept only visual evidence admitted by the sealed crosscheck graph.

        Historical EPIC-043 snapshots keep their visual evidence on the
        occurrence ``SECONDARY`` role.  Their EvidenceArtifact predates the
        visual parents being copied into that artifact, while the immutable
        transcript-visual-crosscheck artifact already seals the Frame ->
        OCR/Vision admission graph.  Do not treat membership in the snapshot
        alone as proof: require that sealed graph and its admitted relation.
        """
        source_type = str(getattr(evidence, "source_type", "") or "").upper()
        expected_type = {"FRAME": "frame", "OCR": "ocr", "VISION": "vision"}.get(source_type)
        artifact_id = str(getattr(source_artifact, "artifact_id", "") or "")
        actual_type = str(getattr(source_artifact, "artifact_type", "") or "")
        if expected_type is None or actual_type != expected_type:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                f"evidence source {artifact_id} is not an EvidenceArtifact parent",
                artifact_id=artifact_id,
            )

        frame_artifact_id = (
            artifact_id if expected_type == "frame"
            else str(getattr(source_artifact, "frame_artifact_id", "") or "")
        )
        frame_artifact = loaded.get(frame_artifact_id)
        if frame_artifact is None or str(getattr(frame_artifact, "artifact_type", "")) != "frame":
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                "visual evidence does not resolve to a snapshot frame",
                artifact_id=artifact_id,
                frame_artifact_id=frame_artifact_id,
            )
        if expected_type != "frame" and frame_artifact_id not in {
            str(item) for item in (getattr(source_artifact, "parent_artifact_ids", ()) or ())
        }:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                "visual evidence source is not linked to its frame by a parent edge",
                artifact_id=artifact_id,
                frame_artifact_id=frame_artifact_id,
            )
        frame_id = str(getattr(source_artifact, "frame_id", "") or "")
        if not frame_id or frame_id != str(getattr(frame_artifact, "frame_id", "") or ""):
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                "visual evidence source frame identity does not match its frame artifact",
                artifact_id=artifact_id,
            )

        evidence_transcript_id = str(getattr(evidence_artifact, "transcript_artifact_id", "") or "")
        if occurrence is None:
            # Pre-item-level snapshots have no occurrence membership to bind.
            # Keep their historical read path, while every snapshot that does
            # carry occurrence IDs takes the exact-scope branch below.
            for crosscheck in loaded.values():
                if str(getattr(crosscheck, "artifact_type", "")) != "transcript_visual_crosscheck":
                    continue
                parent_ids = {str(item) for item in (getattr(crosscheck, "parent_artifact_ids", ()) or ())}
                if artifact_id not in parent_ids or frame_artifact_id not in parent_ids:
                    continue
                if evidence_transcript_id and str(getattr(crosscheck, "transcript_artifact_id", "") or "") != (
                    evidence_transcript_id
                ):
                    continue
                matches = [
                    relation
                    for relation in (getattr(crosscheck, "relations", ()) or ())
                    if str(getattr(relation, "frame_id", "") or "") == frame_id
                    and str(getattr(relation, "frame_artifact_id", "") or "") == frame_artifact_id
                    and str(getattr(relation, "relation", "") or "")
                    in {"SUPPORTS", "CONTRADICTS", "SUPPORTS_DISPLAYED_SECONDARY"}
                ]
                if len(matches) == 1:
                    return
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                "legacy visual evidence source is outside the admitted snapshot crosscheck graph",
                artifact_id=artifact_id,
            )
        packet = dict((getattr(occurrence, "provenance", {}) or {}).get("visual_evidence") or {})
        semantic_segment_id = str(getattr(occurrence, "semantic_segment_id", "") or "")
        scoped_windows = {
            str(window.get("evidence_window_id") or "")
            for window in packet.get("windows") or ()
            if isinstance(window, dict)
            and any(
                isinstance(frame, dict) and str(frame.get("frame_id") or "") == frame_id
                for frame in window.get("frames") or ()
            )
        }
        if not semantic_segment_id or not scoped_windows:
            raise ReplayIntegrityError(
                "REPLAY_LINEAGE_REFERENCE_INVALID",
                "visual evidence lacks an exact owning occurrence/window scope",
                artifact_id=artifact_id,
            )
        if expected_type != "frame":
            packet_has_modality = any(
                str(result.get("artifact_id") or "") == artifact_id
                for window in packet.get("windows") or ()
                if isinstance(window, dict)
                for frame in window.get("frames") or ()
                if isinstance(frame, dict) and str(frame.get("frame_id") or "") == frame_id
                for result in frame.get("ocr" if expected_type == "ocr" else "vision") or ()
                if isinstance(result, dict)
            )
            if not packet_has_modality:
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "secondary visual modality is absent from the owning occurrence packet",
                    artifact_id=artifact_id,
                )
        occurrence_semantic = dict((getattr(occurrence, "provenance", {}) or {}).get("bundle_v2") or {})
        attribution = dict(occurrence_semantic.get("attribution") or {})
        permits_displayed_secondary = (
            occurrence_semantic.get("claim_nature") in {
                "ATTRIBUTED_SECONDARY_POLICY_REPORT", "ATTRIBUTED_SECONDARY_MACRO_FACT_REPORT",
            }
            and occurrence_semantic.get("source_grade") == "SECONDARY"
            and bool(attribution.get("attributed"))
            and "displayed" in str(attribution.get("source_label") or "").lower()
        )
        exact_relations = []
        for crosscheck in loaded.values():
            if str(getattr(crosscheck, "artifact_type", "")) != "transcript_visual_crosscheck":
                continue
            parent_ids = {str(item) for item in (getattr(crosscheck, "parent_artifact_ids", ()) or ())}
            if artifact_id not in parent_ids or frame_artifact_id not in parent_ids:
                continue
            if evidence_transcript_id and str(getattr(crosscheck, "transcript_artifact_id", "") or "") != (
                evidence_transcript_id
            ):
                continue
            for relation in getattr(crosscheck, "relations", ()) or ():
                relation_type = str(getattr(relation, "relation", "") or "")
                admitted = relation_type in {"SUPPORTS", "CONTRADICTS"}
                displayed_secondary = (
                    relation_type == "SUPPORTS_DISPLAYED_SECONDARY" and permits_displayed_secondary
                )
                if (
                    (admitted or displayed_secondary)
                    and str(getattr(relation, "frame_id", "") or "") == frame_id
                    and str(getattr(relation, "frame_artifact_id", "") or "") == frame_artifact_id
                    and tuple(getattr(relation, "semantic_segment_ids", ()) or ()) == (semantic_segment_id,)
                    and tuple(getattr(relation, "evidence_window_ids", ()) or ()) in {
                        (window_id,) for window_id in scoped_windows
                    }
                    # Displayed-secondary pages are intentionally excluded
                    # from the normal eligible frame set: they prove only
                    # that the attributed page was shown.  The immutable
                    # relation plus exact Frame -> OCR/Vision parent graph is
                    # their narrow admission boundary.
                    and (displayed_secondary or frame_id in set(getattr(crosscheck, "eligible_frame_ids", ()) or ()))
                ):
                    exact_relations.append(relation)
        if len(exact_relations) == 1:
            return
        raise ReplayIntegrityError(
            "REPLAY_LINEAGE_REFERENCE_INVALID",
            "visual evidence source lacks one exact admitted snapshot crosscheck relation",
            artifact_id=artifact_id,
        )


__all__ = ["ReplayIntegrityMixin"]

"""ContentSnapshot Replay V2 and immutable lineage verification."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from stock_content.application.pipeline import PipelineContext
from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.artifacts import ArtifactRegistry
from stock_content.domain.lineage import (
    compute_artifact_root_hash,
    compute_content_snapshot_id,
    lineage_of,
    snapshot_identity_payload,
)
from stock_content.domain.models import ContentTask


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
            "replay_pipeline_version",
        }
    )

    def __init__(self, snapshots: SnapshotService, *, artifact_repository: Any | None = None,
                 signal_outbox: Any | None = None, task_repository: Any | None = None,
                 pipeline: Any | None = None, claim_repository: Any | None = None) -> None:
        self._snapshots = snapshots
        self._artifacts = artifact_repository
        self._signal_outbox = signal_outbox
        self._tasks = task_repository
        self._pipeline = pipeline
        self._claims = claim_repository

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
        return {"identity_match": recomputed == snapshot.content_snapshot_id,
                "recomputed_snapshot_id": recomputed,
                "artifact_validation": self._load_and_verify_artifacts(snapshot),
                "snapshot_validation": snapshot_validation}

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

        evidence_artifact = by_type.get("evidence")
        claim_artifact = by_type.get("claims")
        verification_artifact = by_type.get("verification")
        evidence_ids = {str(getattr(item, "evidence_id", ""))
                        for item in (getattr(evidence_artifact, "evidences", ()) or ())}
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
                refs = getattr(persisted_claim or claim, "evidence_refs", None)
                if refs is None and isinstance(claim, dict):
                    refs = claim.get("evidence_refs")
                for evidence_id in refs or ():
                    if str(evidence_id) not in evidence_ids:
                        raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                                   f"claim references missing evidence {evidence_id}")
        if verification_artifact is not None:
            for result in getattr(verification_artifact, "results", ()) or ():
                claim_id = str(getattr(result, "claim_id", None) or
                               (result.get("claim_id") if isinstance(result, dict) else ""))
                if claim_id and claim_id not in claim_ids:
                    raise ReplayIntegrityError("REPLAY_LINEAGE_REFERENCE_MISSING",
                                               f"verification references missing claim {claim_id}")
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
                self._tasks.create(ContentTask(task_id=task_id, source_type=snapshot.source_type,
                                               source_ref=snapshot.source_ref, options=options,
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

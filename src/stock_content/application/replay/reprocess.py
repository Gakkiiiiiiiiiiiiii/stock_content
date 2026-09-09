"""Replay task reconstruction and deterministic artifact comparison."""
from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from stock_content.application.pipeline import PipelineContext
from stock_content.application.replay.errors import ReplayIntegrityError
from stock_content.application.sealed_media import SealedMediaValidationError, validate_sealed_media
from stock_content.domain.artifacts import ArtifactRegistry
from stock_content.domain.models import ContentTask
from stock_content.ports.temporal_reference_snapshot import PinnedTemporalReferenceProvider


class ReplayReprocessMixin:
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
            mapped_source_id = str((snapshot.artifact_ids or {}).get("source") or "")
            if not mapped_source_id or mapped_source_id != str(getattr(source, "artifact_id", "") or ""):
                raise ReplayIntegrityError(
                    "REPLAY_LINEAGE_REFERENCE_INVALID",
                    "replay source artifact is not the source snapshot member",
                )
            options = self._task_options(snapshot)
            if options is None:
                raise ReplayIntegrityError("REPLAY_INPUT_UNAVAILABLE",
                                           "immutable task options are unavailable for this snapshot")
            options = {key: value for key, value in options.items() if key not in self._RUNTIME_OPTIONS}
            options.update(dict(overrides or {}))
            # These values are replay-worker capabilities, not caller input.
            # They are repopulated below only from the source artifact which
            # already passed snapshot lineage validation.
            for key in (
                "replay_raw_storage_uri", "replay_expected_raw_hash",
                "replay_sealed_source_artifact_id", "replay_sealed_snapshot_id",
                "replay_sealed_source_metadata", "replay_sealed_media_root",
            ):
                options.pop(key, None)
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
            # The source artifact is the sealed capture input for replay.  A
            # migration may change the pipeline that *consumes* the source,
            # but it must not recapture source availability, business time, or
            # the original capture pipeline with the replay worker's clock.
            # Those values remain meaningful provenance and therefore belong
            # in the source artifact's deterministic identity.
            source_metadata = dict(getattr(source, "source_metadata", {}) or {})
            source_available_at = source_metadata.get("source_available_from")
            if source_available_at is not None:
                options["replay_source_available_at"] = source_available_at
            source_business_as_of = source_metadata.get("business_as_of")
            if source_business_as_of is not None:
                options["replay_source_business_as_of"] = source_business_as_of
            source_pipeline_version = source_metadata.get("pipeline_version")
            if source_pipeline_version is not None:
                options["replay_source_pipeline_version"] = source_pipeline_version
            if uri and not uri.startswith("fixture://"):
                private_root = os.getenv("CONTENT_RAW_STORAGE_DIR", "").strip()
                try:
                    sealed_media = validate_sealed_media(
                        uri,
                        private_root=private_root,
                        expected_hash=str(getattr(source, "raw_content_hash", "") or ""),
                        expected_length=getattr(source, "raw_content_length", None),
                    )
                except SealedMediaValidationError as exc:
                    raise ReplayIntegrityError(
                        "REPLAY_INPUT_UNAVAILABLE", "sealed raw media is unavailable"
                    ) from exc
                options["replay_raw_storage_uri"] = uri
                options["replay_expected_raw_hash"] = sealed_media.content_hash
                options["replay_sealed_source_artifact_id"] = mapped_source_id
                options["replay_sealed_snapshot_id"] = snapshot.content_snapshot_id
                options["replay_sealed_source_metadata"] = source_metadata
                options["replay_sealed_media_root"] = private_root
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
                                               status="RUNNING", max_retries=1, task_kind="replay"))
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
                    self._finish_replay_failure(task_id, "REPLAY_NONDETERMINISTIC: artifact comparison differs")
            elif self._tasks is not None and (
                hasattr(self._tasks, "succeed_unleased") or hasattr(self._tasks, "succeed")
            ):
                self._finish_replay_success(task_id, {"content_snapshot_id": candidate_id, "replay": payload})
            return payload
        except ReplayIntegrityError as replay_error:
            if task_id is not None and self._tasks is not None and hasattr(self._tasks, "fail"):
                self._finish_replay_failure(task_id, "REPLAY_FAILED: integrity/input validation failed")
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
                self._finish_replay_failure(task_id, f"{code}: {message}")
            raise ReplayIntegrityError(code, message, replay_id=task_id) from exc

    def _finish_replay_success(self, task_id: str, result: dict[str, Any]) -> None:
        finish = getattr(self._tasks, "succeed_unleased", None)
        if finish is not None:
            finish(task_id, result)
        else:  # Compatibility with narrow in-memory replay fakes.
            self._tasks.succeed(task_id, result)

    def _finish_replay_failure(self, task_id: str, error: str) -> None:
        finish = getattr(self._tasks, "fail_unleased", None)
        if finish is not None:
            finish(task_id, "replay", error)
        else:  # Compatibility with narrow in-memory replay fakes.
            self._tasks.fail(task_id, "replay", error)

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
        return next(
            (str(item.content_hash) for item in registry.artifacts() if item.artifact_id == artifact_id),
            None,
        )



__all__ = ["ReplayReprocessMixin"]

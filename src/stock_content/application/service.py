from __future__ import annotations

import hashlib
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from stock_content.application.conflict_service import ConflictService
from stock_content.application.pipeline import ContentPipeline, PipelineContext
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.stages import cleanup_work_directory
from stock_content.application.verification_service import VerificationService, run_verification_pass
from stock_content.domain.artifacts import serialize_artifact
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.models import ContentTask, TranscriptSegment
from stock_content.ports.repositories import (
    ChapterRepository,
    ContentTaskRepository,
    KnowledgeIndex,
    KnowledgeRepository,
    SummaryRepository,
    VideoRepository,
)

LOGGER = logging.getLogger(__name__)


def source_identity_hash_of(source_type: str, source_ref: str) -> str:
    """§4 P0-3：source_identity_hash —— 判断是否是同一来源。"""
    return hashlib.sha256(f"{source_type}:{source_ref}".encode("utf-8")).hexdigest()


class ContentApplication:
    def __init__(
        self,
        task_repository: ContentTaskRepository,
        video_repository: VideoRepository,
        knowledge_repository: KnowledgeRepository,
        knowledge_index: KnowledgeIndex,
        pipeline: ContentPipeline,
        chapter_repository: ChapterRepository | None = None,
        summary_repository: SummaryRepository | None = None,
        snapshot_service: SnapshotService | None = None,
        artifact_repository: Any | None = None,
        claim_repository: Any | None = None,
        signal_outbox=None,
        verification_job_repository: Any | None = None,
        occurrence_repository: Any | None = None,
        lifecycle_repository: Any | None = None,
        pipeline_config: dict[str, Any] | None = None,
        temporal_reference_snapshot_provider: Any | None = None,
    ) -> None:
        self._tasks = task_repository
        self._videos = video_repository
        self._knowledge = knowledge_repository
        self._index = knowledge_index
        self._pipeline = pipeline
        self._chapters = chapter_repository
        self._summaries = summary_repository
        self._snapshots = snapshot_service or SnapshotService()
        self._artifact_repository = artifact_repository or next(
            (
                getattr(stage, "_artifact_repository", None)
                for stage in getattr(pipeline, "_stages", [])
                if getattr(stage, "_artifact_repository", None) is not None
            ),
            None,
        )
        self._claim_repository = claim_repository or next(
            (
                getattr(stage, "_claims", None)
                for stage in getattr(pipeline, "_stages", [])
                if getattr(stage, "_claims", None) is not None
            ),
            None,
        )
        self._signal_outbox = signal_outbox
        self._occurrence_repository = occurrence_repository or next(
            (getattr(stage, "_repository", None) for stage in getattr(pipeline, "_stages", [])
             if getattr(stage, "name", "") == "claim_occurrence_persistence"),
            None,
        )
        self._lifecycle_repository = lifecycle_repository or next(
            (getattr(stage, "_repository", None) for stage in getattr(pipeline, "_stages", [])
             if getattr(stage, "name", "") == "lifecycle_projection"),
            None,
        )
        self._verification_jobs = verification_job_repository
        self._pipeline_config = dict(pipeline_config or {})
        if self._verification_jobs is None and self._claim_repository is not None:
            sessions = getattr(self._claim_repository, "_sessions", None)
            if sessions is not None:
                # Keep the application usable for callers that provide the
                # canonical SQL claim repository but do not explicitly wire
                # its sibling durable verification repository.
                from stock_content.adapters.postgres.repositories.verification_job_repository import (
                    PostgresVerificationJobRepository,
                )

                self._verification_jobs = PostgresVerificationJobRepository(sessions)
        self._replay = ReplayService(
            self._snapshots,
            artifact_repository=self._artifact_repository,
            signal_outbox=self._signal_outbox,
            task_repository=self._tasks,
            pipeline=self._pipeline,
            claim_repository=self._claim_repository,
            occurrence_repository=self._occurrence_repository,
            lifecycle_repository=self._lifecycle_repository,
            verification_repository=self._verification_jobs,
            temporal_reference_snapshot_provider=temporal_reference_snapshot_provider,
        )
        # §5 P1-2/P1-4：Claim 验证生命周期与知识冲突（内存注册表，生产由 worker+DB 驱动）。
        self._verification_lifecycle = VerificationService()
        self._conflict_service = ConflictService()
        self._claims_registry: dict[str, FinancialClaim] = {}

    def enqueue(self, source_type: str, source_ref: str, options: dict | None = None) -> dict:
        options = options or {}
        if self._pipeline_config:
            options = {
                **options,
                "pipeline_config": {
                    **self._pipeline_config,
                    **dict(options.get("pipeline_config") or {}),
                },
            }
        # §4 P0-3：三个禁止混用的概念：
        #   request_idempotency_key —— 防 HTTP 重试重复创建任务；
        #   source_identity_hash   —— 是否同一来源；
        #   source_content_hash    —— 源内容本身是否变更（由调用方或 resolve 阶段提供）。
        request_idempotency_key = str(options.get("idempotency_key") or "") or None
        identity_hash = source_identity_hash_of(source_type, source_ref)
        source_content_hash = str(options.get("source_content_hash") or "") or None
        task = ContentTask(
            task_id=uuid4().hex,
            source_type=source_type,
            source_ref=source_ref,
            options=options,
            input_hash=identity_hash,
            idempotency_key=request_idempotency_key,
            trace_id=str(options.get("trace_id") or "") or None,
        )
        created = self._tasks.create(task)
        return {
            "task_id": created.task_id,
            "status": created.status,
            "stage": created.stage,
            "source_identity_hash": identity_hash,
            "source_content_hash": source_content_hash,
            "request_idempotency_key": request_idempotency_key,
        }

    def get_task(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def process_next(self, worker_id: str, lease_seconds: int = 900) -> dict | None:
        task = self._tasks.claim_pending(worker_id, lease_seconds)
        if task is None:
            return None
        context = PipelineContext(
            task_id=task.task_id,
            source={"type": task.source_type, "ref": task.source_ref},
            options=task.options,
            trace={
                key: str(getattr(task, key, None) or task.options.get(key) or "")
                for key in ("trace_id", "decision_id")
                if getattr(task, key, None) or task.options.get(key)
            },
        )
        try:
            resume = self._restore_resume_context(context, task)
            result = self._pipeline.process(
                context,
                lambda stage, progress: self._tasks.update_progress(task.task_id, stage, progress),
                lambda stage, checkpoint, progress: self._tasks.checkpoint(task.task_id, stage, checkpoint, progress),
                resume=resume,
            )
            payload = {
                "video_id": result.state.video.video_id,
                "chapter_count": len(result.state.chapters),
                "knowledge_count": len(result.state.knowledge),
                "summary": result.state.summary.core_summary,
            }
            # P0 C-03/C-04：ContentSnapshot 由 SnapshotRecordingStage 在 persist 前基于
            # pipeline 已生成的 typed Artifact 记录；失败时 pipeline 异常 → task FAILED。
            snapshot_id = result.state.content_snapshot_id
            if not snapshot_id:
                self._tasks.fail(task.task_id, "content_snapshot", "CONTENT_SNAPSHOT_PERSIST_FAILED: missing snapshot")
                return {
                    "task_id": task.task_id,
                    "status": "FAILED",
                    "stage": "content_snapshot",
                    "error": "CONTENT_SNAPSHOT_PERSIST_FAILED: missing snapshot",
                }
            payload["content_snapshot_id"] = snapshot_id
            self._tasks.succeed(task.task_id, payload)
            return {"task_id": task.task_id, "status": "SUCCEEDED", **payload}
        except Exception as exc:
            self._tasks.fail(task.task_id, context.current_stage, f"{type(exc).__name__}: {exc}")
            return {
                "task_id": task.task_id,
                "status": "FAILED",
                "stage": context.current_stage,
                "error": str(exc),
            }
        finally:
            cleanup_work_directory(context)

    def _restore_resume_context(self, context: PipelineContext, task: ContentTask) -> bool:
        """Restore only the durable, faithful prefix of a previous attempt."""
        repository = self._artifact_repository
        if repository is None or not hasattr(repository, "load_checkpoints"):
            return False
        stage_versions = {
            str(stage.name): str(getattr(stage, "stage_version", "1.0.0"))
            for stage in getattr(self._pipeline, "_stages", [])
        }
        records, persisted = repository.load_checkpoints(task.task_id, stage_versions)
        if not records:
            return False
        context.restored_artifacts = dict(persisted)
        restorable = {
            "resolve",
            "download",
            "frame",
            "audio",
            "asr",
            "diarization",
            "transcript_postprocess",
            "ocr",
            "vision",
            "semantic_segmentation",
            "semantic_context",
            "atomic_claim_extraction",
            "evidence_grounding",
            "temporal_normalization",
            "claim_canonicalization",
            "claim_occurrence_persistence",
            "lifecycle_projection",
        }
        prefix: list[Any] = []
        for record in records:
            if record.status != "SUCCEEDED" or record.stage not in restorable:
                break
            prefix.append(record)
        context.checkpoints = prefix
        # Rebuild only the active slots from checkpoint outputs, in execution
        # order.  ``persisted`` is intentionally a complete, unordered map:
        # it also contains superseded immutable versions needed by resume
        # hash validation, but must not decide which transcript/visual item is
        # active.
        context.artifacts = type(context.artifacts)()
        for record in prefix:
            for artifact_id in record.output_artifact_ids:
                artifact = persisted.get(artifact_id)
                if artifact is None:
                    raise RuntimeError(
                        f"ARTIFACT_INTEGRITY_ERROR: checkpoint artifact missing {artifact_id}"
                    )
                slot = artifact.artifact_type
                if slot in {"frame", "ocr", "vision"}:
                    context.artifacts.add({"frame": "frames", "ocr": "ocr", "vision": "vision"}[slot], artifact)
                elif slot in {
                    "source", "media", "transcript", "semantic_segments", "evidence", "claims",
                    "occurrences", "lifecycle", "verification", "knowledge", "summary"
                }:
                    context.artifacts.set(slot, artifact)
        self._restore_typed_prefix(context, persisted)
        # The presence of durable checkpoint rows is itself meaningful.  Do
        # not silently turn a partially recorded attempt into a clean
        # download: the pipeline will intentionally rerun only the first
        # unrestorable boundary when no successful prefix exists.
        if not prefix:
            context.checkpoints = []
        source = context.artifacts.source
        fixture = self._fixture_options(task.options)
        media_stages = {"download", "frame", "audio", "asr", "diarization", "transcript_postprocess", "ocr", "vision"}
        has_media_checkpoint = any(
            record.status == "SUCCEEDED" and record.stage in media_stages
            for record in records
        )
        if not fixture and has_media_checkpoint:
            if source is None or not source.raw_storage_uri:
                raise RuntimeError("ARTIFACT_INTEGRITY_ERROR: durable raw media unavailable for resume")
            raw_uri = str(source.raw_storage_uri)
            if raw_uri.startswith("file://"):
                raw_uri = raw_uri[7:]
            # A resumable real-media attempt must have a local durable handle;
            # a source URL is not evidence that the bytes are still available.
            if not Path(raw_uri).is_file():
                raise RuntimeError("ARTIFACT_INTEGRITY_ERROR: durable raw media unavailable for resume")
            expected_hash = str(source.raw_content_hash or source.source_content_hash or "")
            digest = hashlib.sha256()
            length = 0
            with Path(raw_uri).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    length += len(chunk)
            if expected_hash and digest.hexdigest() != expected_hash:
                raise RuntimeError("ARTIFACT_INTEGRITY_ERROR: durable raw media hash mismatch")
            if source.raw_content_length is not None and length != source.raw_content_length:
                raise RuntimeError("ARTIFACT_INTEGRITY_ERROR: durable raw media length mismatch")
            # Media-derived outputs use task-scoped ephemeral handles.  A new
            # worker must recreate that workspace before any extractor runs;
            # only the raw bytes and immutable artifacts are durable.
            if context.runtime.work_dir is None:
                context.runtime.work_dir = Path(
                    tempfile.mkdtemp(prefix=f"content-{context.task_id[:8]}-")
                )
            context.runtime.video_path = Path(raw_uri)
            # ``audio`` checkpoints historically contain no filesystem path;
            # when ASR is the first failed stage, restore the ephemeral audio
            # handle from the durable video without rerunning the audio stage.
            if any(record.stage == "audio" and record.status == "SUCCEEDED" for record in records):
                audio_runner = next(
                    (
                        stage
                        for stage in getattr(self._pipeline, "_stages", [])
                        if getattr(stage, "name", "") == "audio"
                    ),
                    None,
                )
                extractor = getattr(getattr(audio_runner, "_stage", audio_runner), "_extractor", None)
                if extractor is not None:
                    context.runtime.audio_path = extractor.extract(context.runtime.video_path, context.runtime.work_dir)
        return bool(records)

    @staticmethod
    def _fixture_options(options: dict[str, Any]) -> bool:
        return bool(options.get("offline_fixture") or "transcript" in options or "segments" in options)

    def _restore_typed_prefix(self, context: PipelineContext, persisted: dict[str, Any]) -> None:
        source = context.artifacts.source
        transcript = context.artifacts.transcript
        if source:
            context.state.metadata = dict(source.source_metadata or {})
        if transcript:
            context.state.segments = [
                TranscriptSegment(
                    segment_index=item.segment_index,
                    start_seconds=item.start_seconds,
                    end_seconds=item.end_seconds,
                    text=item.text,
                    confidence=item.confidence,
                    speaker_id=item.speaker_id or "UNKNOWN",
                )
                for item in transcript.segments
            ]
            context.state.transcript = " ".join(item.text for item in context.state.segments)
        context.state.frames = [
            {
                "frame_id": item.frame_id,
                "timestamp_ms": item.timestamp_ms,
                "image_hash": item.image_hash,
                "storage_ref": item.storage_ref,
                "image_path": item.storage_ref,
            }
            for item in context.artifacts.frames
        ]
        context.state.ocr_evidence = [
            {
                **dict(item.blocks[0] if item.blocks else {}),
                "frame_id": next(
                    (
                        frame.frame_id
                        for frame in context.artifacts.frames
                        if frame.artifact_id == item.frame_artifact_id
                    ),
                    "",
                ),
                "evidence_text": item.text,
            }
            for item in context.artifacts.ocr
        ]
        context.state.frame_insights = [dict(item.payload) for item in context.artifacts.vision]
        context.state.evidence = list(getattr(context.artifacts.evidence, "evidences", ()) or ())
        if context.artifacts.semantic_segments:
            context.state.semantic_segments = list(context.artifacts.semantic_segments.segments or ())
        if context.artifacts.occurrences:
            occurrence_ids = tuple(context.artifacts.occurrences.occurrence_ids or ())
            if occurrence_ids and self._occurrence_repository is None:
                raise RuntimeError(
                    "ARTIFACT_INTEGRITY_ERROR: occurrence repository unavailable for checkpoint rows"
                )
            context.state.occurrences = []
            for occurrence_id in occurrence_ids:
                occurrence = self._occurrence_repository.get(str(occurrence_id))
                if occurrence is None:
                    raise RuntimeError(
                        f"ARTIFACT_INTEGRITY_ERROR: occurrence row missing {occurrence_id}"
                    )
                context.state.occurrences.append(occurrence)
        if context.artifacts.lifecycle:
            lifecycle_ids = (
                tuple(context.artifacts.lifecycle.claim_lifecycle_event_ids or ())
                + tuple(context.artifacts.lifecycle.occurrence_lifecycle_event_ids or ())
            )
            if lifecycle_ids and self._lifecycle_repository is None:
                raise RuntimeError(
                    "ARTIFACT_INTEGRITY_ERROR: lifecycle repository unavailable for checkpoint rows"
                )
            context.state.lifecycle_events = []
            for event_id in lifecycle_ids:
                event = self._lifecycle_repository.get(str(event_id))
                if event is None:
                    raise RuntimeError(
                        f"ARTIFACT_INTEGRITY_ERROR: lifecycle event row missing {event_id}"
                    )
                context.state.lifecycle_events.append(event)
        if context.artifacts.claims and self._claim_repository is not None:
            claim_ids = tuple(context.artifacts.claims.claims or ())
            context.state.claims = []
            for claim_id in claim_ids:
                claim = self._claim_repository.get(str(claim_id))
                if claim is None:
                    raise RuntimeError(
                        f"ARTIFACT_INTEGRITY_ERROR: claim row missing {claim_id}"
                    )
                context.state.claims.append(claim)
        elif context.artifacts.claims and context.artifacts.claims.claims:
            raise RuntimeError(
                "ARTIFACT_INTEGRITY_ERROR: claim repository unavailable for checkpoint rows"
            )

    def get_content_snapshot(self, content_snapshot_id: str) -> dict | None:
        snapshot = self._snapshots.get(content_snapshot_id)
        return snapshot.to_dict() if snapshot else None

    def get_artifact(self, artifact_id: str) -> dict | None:
        artifact = self._artifact_repository.get(artifact_id) if self._artifact_repository else None
        return serialize_artifact(artifact) if artifact else None

    def get_artifact_lineage(self, artifact_id: str) -> dict | None:
        if self._artifact_repository is None:
            return None
        payload = self._artifact_repository.lineage(artifact_id)
        return payload or None

    def get_claim_evidence(self, claim_id: str) -> list[str] | None:
        if self._claim_repository is None:
            return None
        claim = self._claim_repository.get(claim_id)
        if claim is None:
            return None
        if claim.claim_schema_version == "claim.final.v1":
            if self._occurrence_repository is None:
                return []
            evidence_ids: set[str] = set()
            for occurrence in self._occurrence_repository.list_for_claim(claim_id):
                for role in (
                    "evidence_refs", "condition_evidence_refs",
                    "invalidation_evidence_refs", "temporal_evidence_refs",
                ):
                    evidence_ids.update(str(item) for item in (getattr(occurrence, role, ()) or ()))
            return sorted(evidence_ids)
        return list(self._claim_repository.evidence(claim_id))

    def get_claim_verifications(self, claim_id: str) -> list[dict] | None:
        if self._claim_repository is None or self._claim_repository.get(claim_id) is None:
            return None
        return list(self._claim_repository.verifications(claim_id))

    def get_snapshot_lineage(self, content_snapshot_id: str) -> dict | None:
        snapshot = self._snapshots.get(content_snapshot_id)
        if snapshot is None:
            return None

        lineage_errors: list[str] = []

        def artifact_tree(artifact_id: str, visiting: tuple[str, ...] = ()) -> dict | None:
            """Build one complete artifact DAG, recording every integrity error."""
            if self._artifact_repository is None:
                lineage_errors.append("artifact lineage repository unavailable")
                return None
            if artifact_id in visiting:
                cycle = " -> ".join((*visiting, artifact_id))
                lineage_errors.append(f"artifact lineage cycle detected: {cycle}")
                return None
            artifact = self._artifact_repository.get(artifact_id)
            if artifact is None:
                parent = visiting[-1] if visiting else content_snapshot_id
                lineage_errors.append(f"artifact lineage missing: {parent} -> {artifact_id}")
                return None

            current_path = (*visiting, artifact_id)
            parent_ids = sorted({str(item) for item in (artifact.parent_artifact_ids or ())})
            parents = []
            for parent_id in parent_ids:
                parent_tree = artifact_tree(parent_id, current_path)
                if parent_tree is not None:
                    parents.append(parent_tree)
            if len(parents) != len(parent_ids):
                return None

            payload = serialize_artifact(artifact)
            payload["parents"] = parents
            return payload

        artifact_items = []
        artifact_refs = sorted((snapshot.artifact_ids or {}).items())
        for slot, artifact_id in artifact_refs:
            artifact = artifact_tree(str(artifact_id))
            if artifact is not None:
                artifact_items.append({"slot": slot, "artifact": artifact})

        def snapshot_tree(item: Any, visiting: tuple[str, ...] = ()) -> dict[str, Any] | None:
            """Build the complete snapshot ancestry without exposing partial trees.

            A refresh can carry both a logical parent and the snapshot it
            supersedes.  Treat both fields as graph edges, preserving their
            stable field order while de-duplicating an edge when both fields
            point to the same snapshot.  ``visiting`` is path-local so a
            shared ancestor remains a valid DAG edge, while cycles are still
            detected deterministically.
            """
            identifier = str(item.content_snapshot_id)
            if identifier in visiting:
                cycle = " -> ".join((*visiting, identifier))
                lineage_errors.append(f"snapshot lineage cycle detected: {cycle}")
                return None

            current_path = (*visiting, identifier)
            parent_ids: list[str] = []
            for raw_parent_id in (item.parent_snapshot_id, item.supersedes_snapshot_id):
                if not raw_parent_id:
                    continue
                parent_id = str(raw_parent_id)
                if parent_id not in parent_ids:
                    parent_ids.append(parent_id)

            parents: list[dict[str, Any]] = []
            for parent_id in parent_ids:
                parent = self._snapshots.get(parent_id)
                if parent is None:
                    lineage_errors.append(
                        f"snapshot lineage parent missing: {identifier} -> {parent_id}"
                    )
                    continue
                parent_tree = snapshot_tree(parent, current_path)
                if parent_tree is not None:
                    parents.append(parent_tree)

            if len(parents) != len(parent_ids):
                return None
            return {
                "content_snapshot_id": identifier,
                "snapshot_kind": item.snapshot_kind,
                "parent_snapshot_id": item.parent_snapshot_id,
                "supersedes_snapshot_id": item.supersedes_snapshot_id,
                "parents": parents,
            }

        snapshot_tree_payload = snapshot_tree(snapshot)
        lineage_complete = not lineage_errors
        return {
            "snapshot": snapshot.to_dict(),
            "artifacts": artifact_items if lineage_complete else [],
            "snapshot_lineage": snapshot_tree_payload if lineage_complete else None,
            "lineage_complete": lineage_complete,
            "lineage_errors": sorted(set(lineage_errors)),
        }

    def get_signal_lineage(self, signal_id: str) -> dict | None:
        if self._signal_outbox is None or not hasattr(self._signal_outbox, "get_by_signal_id"):
            return None
        row = self._signal_outbox.get_by_signal_id(signal_id)
        if row is None:
            return None
        payload = dict(row.payload or {})
        snapshot_id = str(row.content_snapshot_id or payload.get("content_snapshot_id") or "")
        claim_id = str(row.claim_id or payload.get("claim_id") or "")
        snapshot_lineage = self.get_snapshot_lineage(snapshot_id) if snapshot_id else None
        snapshot = self._snapshots.get(snapshot_id) if snapshot_id else None
        claim = self._claim_repository.get(claim_id) if self._claim_repository and claim_id else None
        # Final claims intentionally do not own source evidence.  For signal
        # lineage, resolve evidence from the exact snapshot's immutable
        # occurrence membership instead of the global/latest claim projection.
        evidence = self._snapshot_claim_evidence(snapshot, claim_id) if claim_id and snapshot else None
        artifact_items = list((snapshot_lineage or {}).get("artifacts") or [])
        artifacts_by_slot = {str(item.get("slot")): item.get("artifact") for item in artifact_items}
        return {
            "signal": payload,
            "snapshot": (snapshot_lineage or {}).get("snapshot"),
            "claim": claim.model_dump(mode="json") if claim else None,
            "evidence_ids": evidence,
            "evidence": artifacts_by_slot.get("evidence"),
            "source": artifacts_by_slot.get("source"),
            "artifacts": artifact_items,
        }

    def _snapshot_claim_evidence(self, snapshot: Any, claim_id: str) -> list[str] | None:
        """Return role evidence owned by ``claim_id`` in one snapshot only."""
        if self._artifact_repository is None or self._occurrence_repository is None:
            return None
        mapping = dict(getattr(snapshot, "artifact_ids", {}) or {})
        occurrence_artifact = self._artifact_repository.get(str(mapping.get("occurrences") or ""))
        evidence_artifact = self._artifact_repository.get(str(mapping.get("evidence") or ""))
        if occurrence_artifact is None or evidence_artifact is None:
            return []
        allowed = {
            str(item.evidence_id)
            for item in (getattr(evidence_artifact, "evidences", ()) or ())
        }
        refs: set[str] = set()
        for occurrence_id in getattr(occurrence_artifact, "occurrence_ids", ()) or ():
            occurrence = self._occurrence_repository.get(str(occurrence_id))
            if occurrence is None or str(occurrence.claim_id) != str(claim_id):
                continue
            for role in (
                "evidence_refs", "condition_evidence_refs",
                "invalidation_evidence_refs", "temporal_evidence_refs",
            ):
                refs.update(str(item) for item in (getattr(occurrence, role, ()) or ()))
        return sorted(refs & allowed)

    def get_snapshot_signals(self, content_snapshot_id: str, claim_id: str | None = None) -> list[dict]:
        if self._signal_outbox is None:
            return []
        return [
            dict(row.payload or {})
            for row in self._signal_outbox.list_for_snapshot(content_snapshot_id, claim_id=claim_id)
        ]

    def factor_signals_v4(self, symbols: list[str], start: datetime, end: datetime) -> list[dict]:
        if self._signal_outbox is None:
            return []
        wanted = set(symbols)
        items = []
        for row in self._signal_outbox.list_all(include_published=True):
            payload = dict(row.payload or {})
            if wanted and payload.get("symbol") not in wanted:
                continue
            available = payload.get("available_from") or payload.get("published_at")
            try:
                when = datetime.fromisoformat(str(available).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=start.tzinfo)
            if start <= when <= end:
                items.append(payload)
        return items

    def replay_content_snapshot(
        self,
        content_snapshot_id: str,
        mode: str | None = None,
        pipeline_version: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict:
        return self._replay.replay(
            content_snapshot_id,
            mode=mode,
            pipeline_version=pipeline_version,
            overrides=overrides,
        )

    def list_snapshots_for_video(self, video_id: str) -> list[dict] | None:
        video = self._videos.get(video_id)
        if video is None:
            return None
        snapshots = self._snapshots.list_for_source(str(video["source_type"]), str(video["source_ref"]))
        return [snapshot.to_dict() for snapshot in snapshots]

    # ---- FinancialClaim / Verification / Conflict（§5 P1）----

    def register_claim(self, claim: FinancialClaim, trace_id: str | None = None) -> dict:
        """登记 claim：进入验证生命周期 + 冲突检测（不阻塞主链路）。"""
        if self._claim_repository is not None:
            self._claim_repository.save(claim)
            # Persist the verification state after the claim commit.  The
            # repository enforces (claim_id, provider) idempotency, so HTTP
            # retries cannot create duplicate jobs or results.
            if self._verification_jobs is not None:
                self._verification_jobs.enqueue([claim], provider="quant", trace_id=trace_id)
        self._claims_registry[claim.claim_id] = claim
        verified_ids = self._verification_lifecycle.verified_claim_ids()
        item = self._verification_lifecycle.submit(claim)
        conflicts = self._conflict_service.register_claims(
            list(self._claims_registry.values()), verified_claim_ids=verified_ids
        )
        persistent = self._persistent_verification(claim.claim_id)
        return {
            "claim_id": claim.claim_id,
            "fact_category": claim.fact_category,
            "verification_status": (persistent or {}).get("status", item.status),
            "conflicts": [conflict.to_dict() for conflict in conflicts],
        }

    def get_claim(self, claim_id: str) -> dict | None:
        claim = self._claim_repository.get(claim_id) if self._claim_repository is not None else None
        claim = claim or self._claims_registry.get(claim_id)
        if claim is None:
            return None
        persistent = self._persistent_verification(claim_id)
        item = self._verification_lifecycle.get(claim_id)
        return {
            **claim.model_dump(mode="json"),
            "verification_status": (persistent or {}).get("status", item.status if item else "EXTRACTED"),
            **({"verification": persistent} if persistent is not None else {}),
        }

    def get_claim_verification(self, claim_id: str) -> dict | None:
        persistent = self._persistent_verification(claim_id)
        if persistent is not None:
            return persistent
        item = self._verification_lifecycle.get(claim_id)
        return item.to_dict() if item else None

    def retry_verification(self, claim_id: str | None = None) -> dict:
        """POST /api/v1/verification/retry：手动触发一轮到期核验。"""
        if claim_id:
            if self._claim_repository is not None and self._claim_repository.get(claim_id) is None:
                return {"error": "CLAIM_NOT_FOUND", "claim_id": claim_id}
            if self._verification_jobs is not None:
                job_info = self._persistent_job(claim_id)
                if job_info is not None:
                    job_id = str(job_info["job_id"])
                    status = str(job_info.get("status") or "")
                    # Never revoke another worker's active lease through a
                    # manual retry request.  PENDING already is the desired
                    # durable retry state; terminal/DLQ rows may be reopened.
                    if status not in {"PENDING", "LEASED"}:
                        self._verification_jobs.requeue(job_id)
                    return {
                        "processed": 1,
                        "pending": 1,
                        "dlq": [],
                        "statuses": {claim_id: "VERIFICATION_PENDING"},
                        "job_id": job_id,
                    }
            item = self._verification_lifecycle.get(claim_id)
            if item is None:
                return {"error": "CLAIM_NOT_FOUND", "claim_id": claim_id}
            processed = [self._verification_lifecycle.attempt(claim_id)]
        else:
            processed = run_verification_pass(self._verification_lifecycle)
        return {
            "processed": len(processed),
            "pending": self._verification_lifecycle.pending_count(),
            "dlq": self._verification_lifecycle.dlq(),
            "statuses": {entry.claim.claim_id: entry.status for entry in processed},
        }

    def _persistent_verification(self, claim_id: str) -> dict | None:
        """Read durable verification state before consulting process memory."""
        if self._claim_repository is None or not hasattr(self._claim_repository, "verifications"):
            return None
        rows = list(self._claim_repository.verifications(claim_id))
        if not rows:
            return None
        # A result is authoritative over the still-present job row.  The
        # repository returns deterministic rows but we explicitly prefer the
        # immutable result to make this rule clear at the API boundary.
        result_rows = [row for row in rows if row.get("verification_id")]
        if result_rows:
            return result_rows[-1]
        job = rows[-1]
        status = str(job.get("status") or "VERIFICATION_PENDING")
        if status in {"PENDING", "LEASED"}:
            status = "VERIFICATION_PENDING"
        return {**job, "status": status}

    def _persistent_job(self, claim_id: str) -> dict | None:
        if self._claim_repository is None or not hasattr(self._claim_repository, "verifications"):
            return None
        for row in self._claim_repository.verifications(claim_id):
            if row.get("job_id"):
                return row
        return None

    def list_conflicts(self, status: str | None = None) -> list[dict]:
        return self._conflict_service.list_conflicts(status)

    def search_knowledge(
        self,
        query: str,
        filters: dict,
        limit: int,
        *,
        availability_as_of: datetime | None = None,
        target_start: str | None = None,
        target_end: str | None = None,
        temporal_role: str | None = None,
        semantic_segment_id: str | None = None,
        business_as_of: datetime | None = None,
        knowledge_as_of: datetime | None = None,
        pit_mode: str | None = None,
    ) -> list[dict]:
        effective_filters = dict(filters or {})
        if pit_mode is None and "pit_mode" not in effective_filters:
            pit_mode = str(self._pipeline_config.get("public_pit_default_mode") or "PUBLIC_STRICT")
        for key, value in {
            "availability_as_of": availability_as_of,
            "target_start": target_start,
            "target_end": target_end,
            "temporal_role": temporal_role,
            "semantic_segment_id": semantic_segment_id,
            "business_as_of": business_as_of,
            "knowledge_as_of": knowledge_as_of,
            "pit_mode": pit_mode,
        }.items():
            if value is not None:
                effective_filters[key] = value
        try:
            knowledge_uids = self._index.search(query, limit * 2)
            if knowledge_uids:
                # Qdrant is candidate-only; every item is hydrated and
                # filtered by the PostgreSQL authority before it is returned.
                hydrated = self._knowledge.hydrate(knowledge_uids, effective_filters)
                if len(hydrated) >= limit:
                    return hydrated[:limit]
                # Candidate search can return only rows rejected by an
                # authoritative PIT filter.  Fill the shortfall from the
                # relational authority, preserving candidate order and
                # removing any rows already hydrated from the index.
                fallback = self._knowledge.search(query, effective_filters, limit)
                seen = {
                    str(item.get("knowledge_uid"))
                    for item in hydrated
                    if item.get("knowledge_uid") is not None
                }
                for item in fallback:
                    uid = item.get("knowledge_uid")
                    if uid is not None and str(uid) in seen:
                        continue
                    hydrated.append(item)
                    if uid is not None:
                        seen.add(str(uid))
                    if len(hydrated) >= limit:
                        break
                return hydrated[:limit]
        except Exception as exc:
            # The relational index remains available during Qdrant outages.
            LOGGER.warning("semantic search unavailable; using relational fallback: %s", exc)
        return self._knowledge.search(query, effective_filters, limit)

    def factor_signals_v5(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        minimum_support_status: str,
        *,
        availability_as_of: datetime | None = None,
        pit_mode: str | None = None,
    ) -> list[dict]:
        """Return the v5 lineage projection from PostgreSQL authority."""
        method = getattr(self._knowledge, "factor_signals_v5", None)
        if method is None:
            return []
        return method(
            symbols,
            start,
            end,
            minimum_support_status,
            availability_as_of=availability_as_of,
            pit_mode=pit_mode,
        )

    def factor_signals(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        minimum_support_status: str,
    ) -> list[dict]:
        return self._knowledge.factor_signals(symbols, start, end, minimum_support_status)

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        return self._videos.get(video_id)

    def list_videos(self, limit: int) -> list[dict]:
        return self._videos.list(limit)

    def get_segments(self, video_id: str) -> list[dict] | None:
        video = self._videos.get(video_id)
        return video.get("segments", []) if video else None

    def get_chapters(self, video_id: str) -> list[dict] | None:
        if self._videos.get(video_id) is None:
            return None
        return self._chapters.list_for_video(video_id) if self._chapters else []

    def get_summary(self, video_id: str) -> dict | None:
        return self._summaries.get(video_id) if self._summaries else None

    def list_video_knowledge(self, video_id: str, limit: int) -> list[dict] | None:
        if self._videos.get(video_id) is None:
            return None
        return self._knowledge.list_for_video(video_id, limit)

    def get_knowledge(self, knowledge_uid: str) -> dict | None:
        return self._knowledge.get(knowledge_uid)

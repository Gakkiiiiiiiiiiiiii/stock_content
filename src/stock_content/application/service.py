from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from stock_content.application.conflict_service import ConflictService
from stock_content.application.pipeline import ContentPipeline, PipelineContext
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.stages import cleanup_work_directory
from stock_content.application.verification_service import VerificationService, run_verification_pass
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.models import ContentTask
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
    ) -> None:
        self._tasks = task_repository
        self._videos = video_repository
        self._knowledge = knowledge_repository
        self._index = knowledge_index
        self._pipeline = pipeline
        self._chapters = chapter_repository
        self._summaries = summary_repository
        self._snapshots = snapshot_service or SnapshotService()
        self._replay = ReplayService(self._snapshots)
        # §5 P1-2/P1-4：Claim 验证生命周期与知识冲突（内存注册表，生产由 worker+DB 驱动）。
        self._verification_lifecycle = VerificationService()
        self._conflict_service = ConflictService()
        self._claims_registry: dict[str, FinancialClaim] = {}

    def enqueue(self, source_type: str, source_ref: str, options: dict | None = None) -> dict:
        options = options or {}
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
        )
        try:
            result = self._pipeline.process(
                context,
                lambda stage, progress: self._tasks.update_progress(task.task_id, stage, progress),
                lambda stage, checkpoint, progress: self._tasks.checkpoint(task.task_id, stage, checkpoint, progress),
            )
            payload = {
                "video_id": result.data["video"].video_id,
                "chapter_count": len(result.data["chapters"]),
                "knowledge_count": len(result.data["knowledge"]),
                "summary": result.data["summary"].core_summary,
            }
            # P0 C-03/C-04：ContentSnapshot 由 SnapshotRecordingStage 在 persist 前基于
            # pipeline 已生成的 typed Artifact 记录；失败时 pipeline 异常 → task FAILED。
            snapshot_id = result.data.get("content_snapshot_id")
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

    def get_content_snapshot(self, content_snapshot_id: str) -> dict | None:
        snapshot = self._snapshots.get(content_snapshot_id)
        return snapshot.to_dict() if snapshot else None

    def replay_content_snapshot(self, content_snapshot_id: str) -> dict:
        return self._replay.replay(content_snapshot_id)

    def list_snapshots_for_video(self, video_id: str) -> list[dict] | None:
        video = self._videos.get(video_id)
        if video is None:
            return None
        snapshots = self._snapshots.list_for_source(str(video["source_type"]), str(video["source_ref"]))
        return [snapshot.to_dict() for snapshot in snapshots]

    # ---- FinancialClaim / Verification / Conflict（§5 P1）----

    def register_claim(self, claim: FinancialClaim) -> dict:
        """登记 claim：进入验证生命周期 + 冲突检测（不阻塞主链路）。"""
        self._claims_registry[claim.claim_id] = claim
        verified_ids = self._verification_lifecycle.verified_claim_ids()
        item = self._verification_lifecycle.submit(claim)
        conflicts = self._conflict_service.register_claims(
            list(self._claims_registry.values()), verified_claim_ids=verified_ids
        )
        return {
            "claim_id": claim.claim_id,
            "fact_category": claim.fact_category,
            "verification_status": item.status,
            "conflicts": [conflict.to_dict() for conflict in conflicts],
        }

    def get_claim(self, claim_id: str) -> dict | None:
        claim = self._claims_registry.get(claim_id)
        if claim is None:
            return None
        item = self._verification_lifecycle.get(claim_id)
        return {
            **claim.model_dump(mode="json"),
            "verification_status": item.status if item else "EXTRACTED",
        }

    def get_claim_verification(self, claim_id: str) -> dict | None:
        item = self._verification_lifecycle.get(claim_id)
        return item.to_dict() if item else None

    def retry_verification(self, claim_id: str | None = None) -> dict:
        """POST /api/v1/verification/retry：手动触发一轮到期核验。"""
        if claim_id:
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

    def list_conflicts(self, status: str | None = None) -> list[dict]:
        return self._conflict_service.list_conflicts(status)

    def search_knowledge(self, query: str, filters: dict, limit: int) -> list[dict]:
        try:
            knowledge_uids = self._index.search(query, limit * 2)
            items = self._knowledge.hydrate(knowledge_uids, filters)
            if items:
                return items[:limit]
        except Exception as exc:
            # The relational index remains available during Qdrant outages.
            LOGGER.warning("semantic search unavailable; using relational fallback: %s", exc)
        return self._knowledge.search(query, filters, limit)

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

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from stock_content.application.pipeline import ContentPipeline, PipelineContext
from stock_content.application.stages import cleanup_work_directory
from stock_content.domain.models import ContentTask
from stock_content.ports.repositories import (
    ContentTaskRepository,
    KnowledgeIndex,
    KnowledgeRepository,
    VideoRepository,
)

LOGGER = logging.getLogger(__name__)


class ContentApplication:
    def __init__(
        self,
        task_repository: ContentTaskRepository,
        video_repository: VideoRepository,
        knowledge_repository: KnowledgeRepository,
        knowledge_index: KnowledgeIndex,
        pipeline: ContentPipeline,
    ) -> None:
        self._tasks = task_repository
        self._videos = video_repository
        self._knowledge = knowledge_repository
        self._index = knowledge_index
        self._pipeline = pipeline

    def enqueue(self, source_type: str, source_ref: str, options: dict | None = None) -> dict:
        task = ContentTask(
            task_id=uuid4().hex,
            source_type=source_type,
            source_ref=source_ref,
            options=options or {},
        )
        self._tasks.create(task)
        return {"task_id": task.task_id, "status": task.status, "stage": task.stage}

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
            )
            payload = {
                "video_id": result.data["video"].video_id,
                "chapter_count": len(result.data["chapters"]),
                "knowledge_count": len(result.data["knowledge"]),
                "summary": result.data["summary"].core_summary,
            }
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

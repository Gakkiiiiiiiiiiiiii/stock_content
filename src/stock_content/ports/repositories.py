from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from stock_content.domain.models import (
    ContentTask,
    KnowledgeUnit,
    TranscriptSegment,
    VideoAsset,
    VideoChapter,
    VideoSummary,
)


class KnowledgeRepository(Protocol):
    def get(self, knowledge_uid: str) -> dict | None: ...

    def list_for_video(self, video_id: str, limit: int) -> list[dict]: ...

    def search(self, query: str, filters: dict, limit: int) -> list[dict]: ...

    def replace_for_video(self, video_id: str, units: list[KnowledgeUnit]) -> None: ...

    def hydrate(self, knowledge_uids: list[str], filters: dict) -> list[dict]: ...

    def factor_signals(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        minimum_support_status: str,
    ) -> list[dict]: ...


class ContentTaskRepository(Protocol):
    def create(self, task: ContentTask) -> ContentTask: ...

    def get(self, task_id: str) -> ContentTask | None: ...

    def claim_pending(self, worker_id: str, lease_seconds: int) -> ContentTask | None: ...

    def update_progress(self, task_id: str, stage: str, progress: int) -> None: ...

    def succeed(self, task_id: str, result: dict[str, Any]) -> None: ...

    def fail(self, task_id: str, stage: str, error: str) -> None: ...


class VideoRepository(Protocol):
    def upsert(self, video: VideoAsset, segments: list[TranscriptSegment]) -> None: ...

    def get(self, video_id: str) -> dict | None: ...

    def list(self, limit: int) -> list[dict]: ...


class ChapterRepository(Protocol):
    def replace_for_video(self, video_id: str, chapters: list[VideoChapter]) -> None: ...

    def list_for_video(self, video_id: str) -> list[dict]: ...


class SummaryRepository(Protocol):
    def upsert(self, summary: VideoSummary) -> None: ...

    def get(self, video_id: str) -> dict | None: ...


class KnowledgeIndex(Protocol):
    def index(self, units: list[KnowledgeUnit]) -> None: ...

    def search(self, query: str, limit: int) -> list[str]: ...

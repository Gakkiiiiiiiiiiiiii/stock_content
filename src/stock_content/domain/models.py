from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ContentTask:
    task_id: str
    source_type: str
    source_ref: str
    status: str = "PENDING"
    stage: str = "queued"
    progress: int = 0
    retry_count: int = 0
    max_retries: int = 3
    error: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAsset:
    video_id: str
    source_type: str
    source_ref: str
    title: str
    author: str | None = None
    duration_seconds: float | None = None
    transcript_text: str = ""
    source_hash: str | None = None


@dataclass
class TranscriptSegment:
    segment_index: int
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None


@dataclass
class VideoChapter:
    chapter_id: str
    chapter_index: int
    title: str
    summary: str
    start_seconds: float
    end_seconds: float
    chapter_type: str = "ANALYSIS"


@dataclass
class KnowledgeUnit:
    knowledge_uid: str
    video_id: str
    chapter_id: str | None
    statement: str
    kind: str = "CLAIM"
    subject: str | None = None
    ticker: str | None = None
    sentiment: str = "NEUTRAL"
    support_status: str = "SOURCE_SUPPORTED"
    review_status: str = "PENDING"
    confidence: float = 0.6
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))
    available_from: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class VideoSummary:
    video_id: str
    core_summary: str
    markdown: str
    confidence: float

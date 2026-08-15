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
    checkpoint: dict[str, Any] = field(default_factory=dict)
    input_hash: str | None = None
    idempotency_key: str | None = None
    trace_id: str | None = None

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
    canonical_url: str | None = None
    published_at: datetime | None = None
    source_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved_at: datetime | None = None


@dataclass
class TranscriptSegment:
    segment_index: int
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    speaker_id: str = "UNKNOWN"
    speaker_confidence: float | None = None
    correction_records: list[dict[str, Any]] = field(default_factory=list)


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
    knowledge_kind: str = "STATE"
    knowledge_version: int = 1
    subject: str | None = None
    subject_key: str | None = None
    predicate_key: str | None = None
    ticker: str | None = None
    sentiment: str = "NEUTRAL"
    support_status: str = "SOURCE_SUPPORTED"
    truth_status: str = "NOT_CHECKED"
    review_status: str = "UNREVIEWED"
    lifecycle_status: str = "ACTIVE"
    confidence: float = 0.6
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))
    available_from: datetime = field(default_factory=lambda: datetime.now(UTC))
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_statement_hash: str | None = None
    content_hash: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoSummary:
    video_id: str
    core_summary: str
    markdown: str
    confidence: float

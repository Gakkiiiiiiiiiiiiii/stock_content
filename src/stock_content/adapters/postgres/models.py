from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ContentTaskRow(Base):
    __tablename__ = "content_ingest_task"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_ref: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)
    stage: Mapped[str] = mapped_column(String(40), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    checkpoint: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    input_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class VideoAssetRow(Base):
    __tablename__ = "video_asset"

    video_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_ref: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(255))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    transcript_text: Mapped[str] = mapped_column(Text, default="")
    source_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    source_version: Mapped[str | None] = mapped_column(String(128))
    # ``metadata`` is reserved by SQLAlchemy's declarative base.  Preserve the
    # physical column name for the schema while using a non-reserved attribute.
    source_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class VideoSegmentRow(Base):
    __tablename__ = "video_segment"
    __table_args__ = (UniqueConstraint("video_id", "segment_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    segment_index: Mapped[int] = mapped_column(Integer)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text)
    normalized_text: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    speaker_id: Mapped[str] = mapped_column(String(64), default="UNKNOWN", index=True)
    speaker_confidence: Mapped[float | None] = mapped_column(Float)
    correction_records: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class VideoChapterRow(Base):
    __tablename__ = "video_chapter"
    __table_args__ = (UniqueConstraint("video_id", "chapter_index"),)

    chapter_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    chapter_index: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    chapter_type: Mapped[str] = mapped_column(String(40), default="ANALYSIS")


class KnowledgeUnitRow(Base):
    __tablename__ = "knowledge_unit"

    knowledge_uid: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    chapter_id: Mapped[str | None] = mapped_column(String(64), index=True)
    statement: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(40), default="CLAIM", index=True)
    knowledge_kind: Mapped[str] = mapped_column(String(40), default="STATE", index=True)
    knowledge_version: Mapped[int] = mapped_column(Integer, default=1)
    predicate_key: Mapped[str | None] = mapped_column(String(255), index=True)
    subject_key: Mapped[str | None] = mapped_column(String(255), index=True)
    subject: Mapped[str | None] = mapped_column(String(255), index=True)
    ticker: Mapped[str | None] = mapped_column(String(32), index=True)
    sentiment: Mapped[str] = mapped_column(String(20), default="NEUTRAL")
    support_status: Mapped[str] = mapped_column(String(40), default="SOURCE_SUPPORTED", index=True)
    truth_status: Mapped[str] = mapped_column(String(40), default="NOT_CHECKED", index=True)
    review_status: Mapped[str] = mapped_column(String(20), default="UNREVIEWED", index=True)
    lifecycle_status: Mapped[str] = mapped_column(String(20), default="ACTIVE", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.6)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    available_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    source_statement_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnowledgeEvidenceRow(Base):
    __tablename__ = "knowledge_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_uid: Mapped[str] = mapped_column(
        ForeignKey("knowledge_unit.knowledge_uid", ondelete="CASCADE"), index=True
    )
    evidence_type: Mapped[str] = mapped_column(String(40), default="TRANSCRIPT")
    source_id: Mapped[str | None] = mapped_column(String(128), index=True)
    video_id: Mapped[str | None] = mapped_column(String(64), index=True)
    frame_id: Mapped[str | None] = mapped_column(String(64), index=True)
    evidence_text: Mapped[str] = mapped_column(Text)
    start_seconds: Mapped[float | None] = mapped_column(Float)
    end_seconds: Mapped[float | None] = mapped_column(Float)
    structured_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float)
    source_reliability: Mapped[float | None] = mapped_column(Float)


class VideoSummaryRow(Base):
    __tablename__ = "video_summary"

    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), primary_key=True)
    core_summary: Mapped[str] = mapped_column(Text)
    markdown: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class VideoFrameRow(Base):
    __tablename__ = "video_frame"
    frame_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, index=True)
    image_hash: Mapped[str] = mapped_column(String(64), index=True)
    extraction_reason: Mapped[str] = mapped_column(String(40))
    storage_ref: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OcrEvidenceRow(Base):
    __tablename__ = "ocr_evidence"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_id: Mapped[str] = mapped_column(ForeignKey("video_frame.frame_id", ondelete="CASCADE"), index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, index=True)
    text: Mapped[str] = mapped_column(Text)
    bbox: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float)
    ocr_engine: Mapped[str] = mapped_column(String(80))
    engine_version: Mapped[str | None] = mapped_column(String(80))


class VisionEvidenceRow(Base):
    __tablename__ = "vision_evidence"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_id: Mapped[str] = mapped_column(ForeignKey("video_frame.frame_id", ondelete="CASCADE"), index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, index=True)
    label: Mapped[str] = mapped_column(String(80), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str | None] = mapped_column(String(120))
    model_version: Mapped[str | None] = mapped_column(String(80))


class TemporalWindowRow(Base):
    __tablename__ = "temporal_window"
    window_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    start_ms: Mapped[int] = mapped_column(Integer, index=True)
    end_ms: Mapped[int] = mapped_column(Integer, index=True)
    transcript: Mapped[str] = mapped_column(Text, default="")
    speaker_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    frame_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    ocr_items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    vision_items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class FinancialEntityRow(Base):
    __tablename__ = "financial_entity"
    entity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    raw_mention: Mapped[str] = mapped_column(String(255))
    entity_type: Mapped[str] = mapped_column(String(40), index=True)
    canonical_name: Mapped[str | None] = mapped_column(String(255))
    canonical_key: Mapped[str | None] = mapped_column(String(255), index=True)
    ticker: Mapped[str | None] = mapped_column(String(32), index=True)
    exchange: Mapped[str | None] = mapped_column(String(24))
    confidence: Mapped[float | None] = mapped_column(Float)
    resolution_source: Mapped[str | None] = mapped_column(String(80))


class FinancialNumericFactRow(Base):
    __tablename__ = "financial_numeric_fact"
    numeric_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    knowledge_uid: Mapped[str | None] = mapped_column(String(64), index=True)
    raw_text: Mapped[str | None] = mapped_column(Text)
    metric: Mapped[str | None] = mapped_column(String(80), index=True)
    value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(40))
    period: Mapped[str | None] = mapped_column(String(40))
    currency: Mapped[str | None] = mapped_column(String(16))
    comparison_type: Mapped[str | None] = mapped_column(String(40))
    qualifier: Mapped[str | None] = mapped_column(String(40))
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence_ref: Mapped[str | None] = mapped_column(String(64))
    as_of_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    available_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class FinancialEventRow(Base):
    __tablename__ = "financial_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    knowledge_uid: Mapped[str | None] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    subject_key: Mapped[str | None] = mapped_column(String(255), index=True)
    objects: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    effective_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    available_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    direction: Mapped[str | None] = mapped_column(String(20))
    strength: Mapped[float | None] = mapped_column(Float)
    numeric_refs: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_refs: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float)


class KnowledgeVerificationRow(Base):
    __tablename__ = "knowledge_verification"
    verification_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    knowledge_uid: Mapped[str] = mapped_column(
        ForeignKey("knowledge_unit.knowledge_uid", ondelete="CASCADE"), index=True
    )
    verifier_type: Mapped[str] = mapped_column(String(40), index=True)
    decision: Mapped[str] = mapped_column(String(40), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    reason_code: Mapped[str | None] = mapped_column(String(80))
    model_name: Mapped[str | None] = mapped_column(String(120))
    model_version: Mapped[str | None] = mapped_column(String(80))
    prompt_version: Mapped[str | None] = mapped_column(String(80))
    raw_output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnowledgeCrossVideoRow(Base):
    __tablename__ = "knowledge_cross_video"
    knowledge_uid: Mapped[str] = mapped_column(
        ForeignKey("knowledge_unit.knowledge_uid", ondelete="CASCADE"), primary_key=True
    )
    corroborating_video_count: Mapped[int] = mapped_column(Integer, default=0)
    contradicting_video_count: Mapped[int] = mapped_column(Integer, default=0)
    independent_source_count: Mapped[int] = mapped_column(Integer, default=0)
    author_attention_score: Mapped[float] = mapped_column(Float, default=0.0)
    consensus_score: Mapped[float] = mapped_column(Float, default=0.0)
    disagreement_score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnowledgeConflictRow(Base):
    __tablename__ = "knowledge_conflict"
    conflict_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    knowledge_uid: Mapped[str] = mapped_column(
        ForeignKey("knowledge_unit.knowledge_uid", ondelete="CASCADE"), index=True
    )
    related_knowledge_uid: Mapped[str] = mapped_column(String(64), index=True)
    conflict_type: Mapped[str] = mapped_column(String(40), index=True)
    resolution: Mapped[str] = mapped_column(String(40))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnowledgeLifecycleEventRow(Base):
    __tablename__ = "knowledge_lifecycle_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    knowledge_uid: Mapped[str] = mapped_column(
        ForeignKey("knowledge_unit.knowledge_uid", ondelete="CASCADE"), index=True
    )
    knowledge_version: Mapped[int] = mapped_column(Integer)
    from_status: Mapped[str | None] = mapped_column(String(40))
    to_status: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str] = mapped_column(Text)
    trigger: Mapped[str] = mapped_column(String(80))
    actor: Mapped[str] = mapped_column(String(80))
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisDocumentRow(Base):
    __tablename__ = "analysis_document"
    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    markdown: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QualityMetricRow(Base):
    __tablename__ = "content_quality_metric"
    metric_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_asset.video_id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    value: Mapped[float] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


JSONPayload = JSON().with_variant(JSONB(), "postgresql")


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
    # Stable transcript coordinate.  SQL upgrades fill legacy rows with an
    # explicit legacy_trseg_ identity before new writes are accepted.
    segment_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
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
    content_attention_score: Mapped[float] = mapped_column(Float, default=0.0)
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
    __tablename__ = "knowledge_lifecycle_event_legacy"
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


class LifecycleEventLedgerRow(Base):
    """Forward-compatible append-only ledger (migration 020)."""
    __tablename__ = "knowledge_lifecycle_event"
    __table_args__ = (
        CheckConstraint("target_type IN ('CLAIM','OCCURRENCE')"),
    )
    lifecycle_event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    target_type: Mapped[str] = mapped_column(String(32), index=True)
    target_id: Mapped[str] = mapped_column(String(128), index=True)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str] = mapped_column(String(32))
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    reason_code: Mapped[str] = mapped_column(String(80))
    policy_version: Mapped[str] = mapped_column(String(64))
    supersedes_event_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SemanticSegmentRow(Base):
    __tablename__ = "semantic_segment"
    __table_args__ = (
        UniqueConstraint("transcript_artifact_id", "segment_index"),
        CheckConstraint("start_segment_index <= end_segment_index"),
        CheckConstraint("start_ms <= end_ms"),
    )
    semantic_segment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    transcript_artifact_id: Mapped[str] = mapped_column(String(128), index=True)
    video_id: Mapped[str] = mapped_column(String(64), index=True)
    segment_index: Mapped[int] = mapped_column(Integer)
    start_segment_id: Mapped[str] = mapped_column(String(128))
    end_segment_id: Mapped[str] = mapped_column(String(128))
    start_segment_index: Mapped[int] = mapped_column(Integer)
    end_segment_index: Mapped[int] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    topic: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(String(255))
    segment_type: Mapped[str] = mapped_column(String(40), default="ANALYSIS")
    model_id: Mapped[str] = mapped_column(String(160), default="")
    prompt_version: Mapped[str] = mapped_column(String(80), default="")
    confidence: Mapped[float | None] = mapped_column(Float)
    artifact_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TemporalBindingRow(Base):
    __tablename__ = "claim_temporal_binding"
    __table_args__ = (
        CheckConstraint("value_type IN ('NONE','DATE','TIMESTAMP')"),
        CheckConstraint(
            "value_type <> 'DATE' OR (start_time IS NULL AND end_time IS NULL AND "
            "earliest_start_time IS NULL AND latest_start_time IS NULL AND "
            "earliest_end_time IS NULL AND latest_end_time IS NULL)"
        ),
        CheckConstraint(
            "value_type <> 'TIMESTAMP' OR (start_date IS NULL AND end_date IS NULL AND "
            "earliest_start_date IS NULL AND latest_start_date IS NULL AND "
            "earliest_end_date IS NULL AND latest_end_date IS NULL)"
        ),
        CheckConstraint(
            "value_type <> 'NONE' OR (start_time IS NULL AND end_time IS NULL AND "
            "earliest_start_time IS NULL AND latest_start_time IS NULL AND "
            "earliest_end_time IS NULL AND latest_end_time IS NULL AND start_date IS NULL AND "
            "end_date IS NULL AND earliest_start_date IS NULL AND latest_start_date IS NULL AND "
            "earliest_end_date IS NULL AND latest_end_date IS NULL)"
        ),
    )
    claim_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    temporal_binding_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    role: Mapped[str] = mapped_column(String(40))
    scope: Mapped[str] = mapped_column(String(32))
    value_type: Mapped[str] = mapped_column(String(16))
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    earliest_start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latest_start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    earliest_end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latest_end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    earliest_start_date: Mapped[date | None] = mapped_column(Date)
    latest_start_date: Mapped[date | None] = mapped_column(Date)
    earliest_end_date: Mapped[date | None] = mapped_column(Date)
    latest_end_date: Mapped[date | None] = mapped_column(Date)
    period_label: Mapped[str | None] = mapped_column(String(64))
    raw_expression: Mapped[str | None] = mapped_column(Text)
    expression_key: Mapped[str | None] = mapped_column(String(128))
    precision: Mapped[str] = mapped_column(String(32))
    granularity: Mapped[str | None] = mapped_column(String(32))
    assertion_status: Mapped[str] = mapped_column(String(32))
    metric_temporal_nature: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)
    timezone: Mapped[str | None] = mapped_column(String(64))
    calendar_type: Mapped[str | None] = mapped_column(String(32))
    calendar_id: Mapped[str | None] = mapped_column(String(64))
    market_session: Mapped[str | None] = mapped_column(String(32))
    recurrence: Mapped[dict[str, Any] | None] = mapped_column(JSONPayload)
    normalization_status: Mapped[str] = mapped_column(String(32))
    normalization_reason: Mapped[str | None] = mapped_column(Text)
    normalization_version: Mapped[str] = mapped_column(String(64))
    source_evidence_refs: Mapped[list[str]] = mapped_column(JSONPayload, default=list)
    reference_snapshot_id: Mapped[str | None] = mapped_column(String(128))
    reference_data_version: Mapped[str | None] = mapped_column(String(128))
    reference_available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TemporalRelationRow(Base):
    __tablename__ = "claim_temporal_relation"
    claim_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    temporal_relation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    relation_type: Mapped[str] = mapped_column(String(32))
    from_binding_id: Mapped[str] = mapped_column(String(128))
    to_binding_id: Mapped[str] = mapped_column(String(128))
    lag_value: Mapped[float | None] = mapped_column(Float)
    lag_unit: Mapped[str | None] = mapped_column(String(32))
    lag_min: Mapped[float | None] = mapped_column(Float)
    lag_max: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)


class ClaimOccurrenceRow(Base):
    __tablename__ = "claim_occurrence"
    __table_args__ = (
        CheckConstraint("available_from >= ingested_at"),
        CheckConstraint("available_from >= extraction_completed_at"),
        CheckConstraint("available_from >= snapshot_committed_at"),
    )
    occurrence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(128), index=True)
    source_artifact_id: Mapped[str] = mapped_column(String(128), index=True)
    transcript_artifact_id: Mapped[str] = mapped_column(String(128), index=True)
    semantic_segment_id: Mapped[str] = mapped_column(String(64), index=True)
    assertion_locator_hash: Mapped[str] = mapped_column(String(128))
    asserted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_availability_quality: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    extraction_completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    snapshot_committed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    available_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_support_status: Mapped[str] = mapped_column(String(40), default="SOURCE_LOCATED")
    source_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    extractor_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    raw_temporal_expressions: Mapped[list[Any]] = mapped_column(JSONPayload, default=list)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)


class ClaimOccurrenceEvidenceRow(Base):
    __tablename__ = "claim_occurrence_evidence"
    __table_args__ = (UniqueConstraint("occurrence_id", "evidence_id", "evidence_role"),)
    occurrence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evidence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evidence_role: Mapped[str] = mapped_column(String(32), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)


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


class ContentSnapshotRow(Base):
    """详细修改方案 §4 P0-2：ContentSnapshot 权威存储（PostgreSQL 为事实真值）。"""

    __tablename__ = "content_snapshot"

    content_snapshot_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_ref: Mapped[str] = mapped_column(Text)
    source_content_hash: Mapped[str] = mapped_column(String(64), index=True)
    identity: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)
    artifact_ids: Mapped[dict[str, str]] = mapped_column(JSONPayload, default=dict)
    quant_market_snapshot_ids: Mapped[list[str]] = mapped_column(JSONPayload, default=list)
    pipeline_version: Mapped[str] = mapped_column(String(40), default="pipeline.v2")
    schema_version: Mapped[str] = mapped_column(String(40), default="content.snapshot.v1")
    code_sha: Mapped[str | None] = mapped_column(String(64))
    config_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    # v2 lineage (kept additive for 012/013 databases)
    source_artifact_id: Mapped[str | None] = mapped_column(String(96), index=True)
    artifact_root_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    snapshot_kind: Mapped[str] = mapped_column(String(32), default="INITIAL")
    parent_snapshot_id: Mapped[str | None] = mapped_column(String(80), index=True)
    supersedes_snapshot_id: Mapped[str | None] = mapped_column(String(80), index=True)
    producer_manifest: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)
    lifecycle_artifact_id: Mapped[str | None] = mapped_column(String(128), index=True)


class ContentArtifactRow(Base):
    __tablename__ = "content_artifact"
    __table_args__ = (
        UniqueConstraint("artifact_type", "content_hash", name="uq_content_artifact_type_hash"),
    )
    artifact_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    artifact_type: Mapped[str] = mapped_column(String(40), index=True)
    schema_version: Mapped[str] = mapped_column(String(40), default="artifact.v1")
    producer_stage: Mapped[str] = mapped_column(String(80), default="")
    producer_version: Mapped[str] = mapped_column(String(80), default="1.0.0")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    parent_artifact_ids: Mapped[list[str]] = mapped_column(JSONPayload, default=list)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ContentArtifactEdgeRow(Base):
    __tablename__ = "content_artifact_edge"
    edge_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("content_artifact.artifact_id", ondelete="CASCADE"), index=True)
    parent_artifact_id: Mapped[str] = mapped_column(String(96), index=True)
    relation: Mapped[str] = mapped_column(String(40), default="PARENT")


class ContentStageCheckpointRow(Base):
    __tablename__ = "content_stage_checkpoint"
    checkpoint_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("content_ingest_task.task_id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(80), index=True)
    stage_version: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(32), index=True)
    artifact_ids: Mapped[list[str]] = mapped_column(JSONPayload, default=list)
    artifact_hashes: Mapped[list[str]] = mapped_column(JSONPayload, default=list)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FinancialClaimRow(Base):
    __tablename__ = "financial_claim"
    claim_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    claim_type: Mapped[str] = mapped_column(String(48), index=True)
    fact_category: Mapped[str] = mapped_column(String(32), index=True)
    subject_type: Mapped[str] = mapped_column(String(80))
    subject_id: Mapped[str] = mapped_column(String(255), index=True)
    predicate: Mapped[str] = mapped_column(String(255), index=True)
    value: Mapped[Any] = mapped_column(JSONPayload)
    unit: Mapped[str | None] = mapped_column(String(40))
    currency: Mapped[str | None] = mapped_column(String(16))
    fact_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_confidence: Mapped[float] = mapped_column(Float)
    extractor_confidence: Mapped[float] = mapped_column(Float)
    extraction_model_id: Mapped[str] = mapped_column(String(120), default="unknown")
    extraction_prompt_version: Mapped[str] = mapped_column(String(120), default="unknown")
    condition_text: Mapped[str | None] = mapped_column(Text)
    invalidation_text: Mapped[str | None] = mapped_column(Text)
    claim_schema_version: Mapped[str] = mapped_column(String(40), default="claim.v2")
    normalization_version: Mapped[str] = mapped_column(String(40), default="normalization.v1")
    source_support_status: Mapped[str] = mapped_column(String(24), default="UNSUPPORTED")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ClaimEvidenceRow(Base):
    __tablename__ = "claim_evidence"
    __table_args__ = (
        UniqueConstraint("claim_id", "evidence_id", name="uq_claim_evidence_claim_evidence"),
    )
    member_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("financial_claim.claim_id", ondelete="CASCADE"), index=True)
    evidence_id: Mapped[str] = mapped_column(String(96), index=True)
    relation: Mapped[str] = mapped_column(String(32), default="SUPPORTS")


class ClaimArtifactMemberRow(Base):
    __tablename__ = "claim_artifact_member"
    member_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("content_artifact.artifact_id", ondelete="CASCADE"), index=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("financial_claim.claim_id", ondelete="CASCADE"), index=True)


class ClaimVerificationJobRow(Base):
    __tablename__ = "claim_verification_job"
    job_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("financial_claim.claim_id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(32), index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=5)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_error: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(96), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    __table_args__ = (UniqueConstraint("claim_id", "provider"),)


class ClaimVerificationResultRow(Base):
    __tablename__ = "claim_verification_result"
    verification_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("financial_claim.claim_id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(32), index=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(128), index=True)
    market_data_version: Mapped[str | None] = mapped_column(String(80))
    result_payload: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)
    trace_id: Mapped[str | None] = mapped_column(String(96))
    fact_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    adjustment: Mapped[str | None] = mapped_column(String(32))
    verification_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    verification_rule_version: Mapped[str | None] = mapped_column(String(64))
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ContentSnapshotArtifactRow(Base):
    __tablename__ = "content_snapshot_artifact"
    member_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    content_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("content_snapshot.content_snapshot_id", ondelete="CASCADE"), index=True
    )
    artifact_id: Mapped[str] = mapped_column(ForeignKey("content_artifact.artifact_id"), index=True)
    slot: Mapped[str] = mapped_column(String(40), index=True)
    artifact_role: Mapped[str | None] = mapped_column(String(40))


class SignalOutboxRow(Base):
    __tablename__ = "content_signal_outbox"
    outbox_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    content_snapshot_id: Mapped[str | None] = mapped_column(String(80), index=True)
    claim_id: Mapped[str | None] = mapped_column(String(96), index=True)
    schema_version: Mapped[str] = mapped_column(String(48), default="content-factor-signal.v4")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONPayload, default=dict)
    status: Mapped[str] = mapped_column(String(24), default="PENDING", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_error: Mapped[str | None] = mapped_column(Text)
    lease_owner: Mapped[str | None] = mapped_column(String(128), index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @property
    def retry_count(self) -> int:
        """Design-level name; ``attempts`` remains the legacy column."""
        return self.attempts

    @retry_count.setter
    def retry_count(self, value: int) -> None:
        self.attempts = value


class ContentSourceHeadRow(Base):
    __tablename__ = "content_source_head"
    source_identity_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    latest_snapshot_id: Mapped[str] = mapped_column(String(80))
    latest_verified_snapshot_id: Mapped[str | None] = mapped_column(String(80))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

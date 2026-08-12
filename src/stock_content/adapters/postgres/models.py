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
    confidence: Mapped[float | None] = mapped_column(Float)


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
    subject: Mapped[str | None] = mapped_column(String(255), index=True)
    ticker: Mapped[str | None] = mapped_column(String(32), index=True)
    sentiment: Mapped[str] = mapped_column(String(20), default="NEUTRAL")
    support_status: Mapped[str] = mapped_column(String(40), default="SOURCE_SUPPORTED", index=True)
    review_status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.6)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    available_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnowledgeEvidenceRow(Base):
    __tablename__ = "knowledge_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_uid: Mapped[str] = mapped_column(
        ForeignKey("knowledge_unit.knowledge_uid", ondelete="CASCADE"), index=True
    )
    evidence_type: Mapped[str] = mapped_column(String(40), default="TRANSCRIPT")
    evidence_text: Mapped[str] = mapped_column(Text)
    start_seconds: Mapped[float | None] = mapped_column(Float)
    end_seconds: Mapped[float | None] = mapped_column(Float)


class VideoSummaryRow(Base):
    __tablename__ = "video_summary"

    video_id: Mapped[str] = mapped_column(
        ForeignKey("video_asset.video_id", ondelete="CASCADE"), primary_key=True
    )
    core_summary: Mapped[str] = mapped_column(Text)
    markdown: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

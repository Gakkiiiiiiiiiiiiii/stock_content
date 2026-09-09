"""Real PostgreSQL persistence coverage for OCR visual geometry."""

from __future__ import annotations

from sqlalchemy import select

from stock_content.adapters.postgres.models import OcrEvidenceRow
from stock_content.adapters.postgres.repositories import (
    PostgresChapterRepository,
    PostgresFinancialEntityRepository,
    PostgresFinancialRepository,
    PostgresKnowledgeRepository,
    PostgresMultimodalRepository,
    PostgresSummaryRepository,
    PostgresVideoRepository,
)
from stock_content.adapters.postgres.repositories.content_task_repository import PostgresContentTaskRepository
from stock_content.application.fenced_effects import PostgresFencedEffectUnitOfWork
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import PersistStage
from stock_content.domain.models import ContentTask, VideoAsset, VideoSummary
from stock_content.domain.worker_capability import TaskKind


def test_postgres_fenced_persist_preserves_paddle_polygon_with_empty_knowledge(postgres_database):
    """The live persist path accepts OCR polygons even when no KUs survive review."""
    sessions = postgres_database.session_factory
    tasks = PostgresContentTaskRepository(sessions)
    tasks.create(ContentTask("pg-ocr-polygon", "xiaoe", "course/video", task_kind="video_pipeline"))
    claimed = tasks.claim_pending("video-worker", TaskKind.VIDEO_PIPELINE.value, 60)
    assert claimed is not None

    context = PipelineContext(
        task_id=claimed.task_id,
        source={"type": "xiaoe", "ref": "course/video"},
        worker_id="video-worker",
        fencing_token=claimed.fencing_token,
    )
    context.state.video = VideoAsset(
        video_id="video-pg-ocr-polygon",
        source_type="xiaoe",
        source_ref="course/video",
        title="OCR polygon persistence",
    )
    context.state.chapters = []
    context.state.knowledge = []
    context.state.summary = VideoSummary(
        video_id=context.state.video.video_id,
        core_summary="No admitted knowledge units.",
        markdown="",
        confidence=0.0,
    )
    context.state.frames = [
        {
            "frame_id": "frame-pg-ocr-polygon",
            "timestamp_ms": 1_000,
            "image_hash": "a" * 64,
            "trigger_source": "KNOWLEDGE_POINT",
            "storage_ref": "private/frame.jpg",
        }
    ]
    polygon = [[1, 2], [30, 2], [30, 40], [1, 40]]
    context.state.ocr_evidence = [
        {
            "frame_id": "frame-pg-ocr-polygon",
            "timestamp_ms": 1_000,
            "evidence_text": "600519",
            "bbox": polygon,
            "confidence_score": 0.99,
            "ocr_engine": "paddleocr",
            "ocr_engine_version": "3.7",
        }
    ]
    # PersistStage writes visual evidence only after the crosscheck admission
    # boundary.  This makes the regression exercise the exact production path.
    context.state.eligible_frame_insights = [{"frame_id": "frame-pg-ocr-polygon"}]

    PersistStage(
        PostgresVideoRepository(sessions),
        PostgresChapterRepository(sessions),
        PostgresKnowledgeRepository(sessions),
        PostgresSummaryRepository(sessions),
        PostgresMultimodalRepository(sessions),
        PostgresFinancialRepository(sessions),
        PostgresFinancialEntityRepository(sessions),
        fenced_effects=PostgresFencedEffectUnitOfWork(sessions),
    ).execute(context)

    with sessions() as session:
        stored = session.scalar(select(OcrEvidenceRow).where(OcrEvidenceRow.frame_id == "frame-pg-ocr-polygon"))
    assert stored is not None
    assert stored.bbox == polygon

"""Real PostgreSQL persistence coverage for OCR visual geometry."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from stock_content.adapters.postgres.models import KnowledgeUnitRow, OcrEvidenceRow
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
from stock_content.application.stages import KnowledgeExtractionStage, PersistStage
from stock_content.domain.claim_canonicalizer import ClaimCanonicalizer
from stock_content.domain.claim_draft import ClaimOccurrenceDraft
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.knowledge_projection_builder import KnowledgeProjectionBuilder
from stock_content.domain.models import ContentTask, VideoAsset, VideoSummary
from stock_content.domain.temporal_semantics import OccurrenceTimes
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


def test_postgres_fenced_persist_compacts_legacy_occurrence_uids_and_denies_chinese_tickers(postgres_database):
    """Persist the twelve v2 knowledge classes that exposed the live overflow."""
    sessions = postgres_database.session_factory
    tasks = PostgresContentTaskRepository(sessions)
    tasks.create(ContentTask("pg-knowledge-uids", "xiaoe", "course/video", task_kind="video_pipeline"))
    claimed = tasks.claim_pending("video-worker", TaskKind.VIDEO_PIPELINE.value, 60)
    assert claimed is not None
    now = datetime(2026, 9, 9, tzinfo=UTC)
    classes = [
        ("压舱层配置三到四成。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("核心层配置四到五成。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("卫星层配置一到二成。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("单一标的不超过两成。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("单次风险预算为总资产两成。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("用二十日均线作为退出规则。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("不使用杠杆。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("连续三次失误后暂停交易。", "PORTFOLIO_RISK_MANAGEMENT"),
        ("银行资本补充是课程讨论的主题。", "FINANCIAL_SECTOR_CAPITAL_POLICY"),
        ("AI推理算力是课程讨论的主题。", "AI_INFERENCE_COMPUTE"),
        ("2030年信息基础设施投资是课程讨论的主题。", "INFORMATION_INFRASTRUCTURE_POLICY"),
        ("截至8月末黄金储备是课程讨论的主题。", "CENTRAL_BANK_GOLD_RESERVES"),
    ]
    records = []
    canonicalizer = ClaimCanonicalizer()
    projection_builder = KnowledgeProjectionBuilder()
    for index, (statement, domain) in enumerate(classes):
        subject = f"中文主题{index + 1}"
        claim = canonicalizer.canonicalize(
            ClaimOccurrenceDraft(
                semantic_segment_id=f"segment-{index}",
                knowledge_kind="METHOD",
                claim_type="OPINION",
                subject_type="CONTENT",
                subject_key=subject,
                subject_name=subject,
                predicate_key=f"method-{index}",
                conclusion=statement,
                extraction_confidence=0.9,
                bundle_v2={
                    "primary_domain": domain,
                    "detail": {"explanation": statement, "scope": "课程口播"},
                },
            )
        )
        assert claim.ticker is None
        occurrence = ClaimOccurrence(
            occurrence_id=("co_" + "A" * 64) if index == 0 else "",
            claim_id=claim.claim_id,
            source_artifact_id=f"source-{index}",
            transcript_artifact_id="transcript-1",
            semantic_segment_id=f"segment-{index}",
            evidence_refs=[f"evidence-{index}"],
            times=OccurrenceTimes(
                ingested_at=now,
                extraction_completed_at=now,
                snapshot_committed_at=now,
                available_from=now,
            ),
        )
        projection = projection_builder.build(claim, occurrence)
        if index:
            assert len(occurrence.occurrence_id) == 64
        else:
            assert len(occurrence.occurrence_id) == 67
            assert len(projection["knowledge_uid"]) == 64
        records.append(
            {
                "knowledge_uid": projection["knowledge_uid"],
                "statement": projection["statement"],
                "knowledge_kind": projection["knowledge_kind"],
                "subject_key": projection["subject_key"],
                "ticker": projection["ticker"],
                "support_status": "SOURCE_SUPPORTED",
                "truth_status": "NOT_CHECKED",
                "review_status": "UNREVIEWED",
                "lifecycle_status": "ACTIVE",
                "support_score": 0.9,
                "extraction_confidence": 0.9,
                "as_of_time": now,
                "attributes": projection["attributes"],
                "extractor_version": "test.v2",
                "schema_version": "claim.final.v1",
                "semantic_hash": claim.claim_id,
            }
        )
    units = KnowledgeExtractionStage._to_domain("video-pg-knowledge-uids", records, now)
    assert len(units) == 12
    assert len({unit.knowledge_uid for unit in units}) == 12
    assert max(len(unit.knowledge_uid) for unit in units) <= 64
    assert all(unit.ticker is None for unit in units)

    context = PipelineContext(
        task_id=claimed.task_id,
        source={"type": "xiaoe", "ref": "course/video"},
        worker_id="video-worker",
        fencing_token=claimed.fencing_token,
    )
    context.state.video = VideoAsset(
        video_id="video-pg-knowledge-uids",
        source_type="xiaoe",
        source_ref="course/video",
        title="Knowledge UID persistence",
    )
    context.state.knowledge = units
    context.state.summary = VideoSummary(
        video_id=context.state.video.video_id,
        core_summary="Twelve v2 knowledge units.",
        markdown="",
        confidence=0.9,
    )
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
        stored = session.scalars(
            select(KnowledgeUnitRow)
            .where(KnowledgeUnitRow.video_id == "video-pg-knowledge-uids")
            .order_by(KnowledgeUnitRow.knowledge_uid)
        ).all()
    assert len(stored) == 12
    assert all(len(row.knowledge_uid) <= 64 and row.ticker is None for row in stored)
    assert {row.attributes["bundle_v2"]["primary_domain"] for row in stored} == {domain for _, domain in classes}

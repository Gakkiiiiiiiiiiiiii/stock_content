from __future__ import annotations

import os

from stock_content.adapters.http import HttpExternalFactProvider, QuantExternalFactProvider
from stock_content.adapters.media import (
    FasterWhisperRecognizer,
    FfmpegAudioExtractor,
    FfmpegFrameExtractor,
    HttpVisionAnalyzer,
    PaddleOcrEngine,
    PyannoteDiarizer,
)
from stock_content.adapters.postgres import Database
from stock_content.adapters.postgres.repositories import (
    PostgresChapterRepository,
    PostgresContentTaskRepository,
    PostgresFinancialEntityRepository,
    PostgresFinancialRepository,
    PostgresKnowledgeRepository,
    PostgresMultimodalRepository,
    PostgresSummaryRepository,
    PostgresVerificationRepository,
    PostgresVideoRepository,
    SqlSnapshotStore,
)
from stock_content.adapters.qdrant import NullKnowledgeIndex, QdrantKnowledgeIndex
from stock_content.adapters.sources import BilibiliSourceAdapter, XiaoeHlsSourceAdapter
from stock_content.application.pipeline import ContentPipeline
from stock_content.application.service import ContentApplication
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.stages import (
    ASRStage,
    AudioStage,
    BuildVideoStage,
    ChapterStage,
    DownloadStage,
    FinancialEnrichmentStage,
    FrameExtractionStage,
    IndexStage,
    KnowledgeExtractionStage,
    MultimodalContextStage,
    OCRStage,
    PersistStage,
    ResolveSourceStage,
    SpeakerDiarizationStage,
    SummaryStage,
    TemporalWindowStage,
    TranscriptPostprocessStage,
    VerificationStage,
    VisionStage,
)
from stock_content.domain.chapter import ChapterSegmenter
from stock_content.domain.external_fact_verifier import ExternalFactVerifier
from stock_content.domain.multimodal_context_builder import MultimodalContextBuilder
from stock_content.domain.summary import SummaryGenerator
from stock_content.domain.temporal_window_builder import TemporalWindowBuilder
from stock_content.domain.transcript_postprocessor import TranscriptPostprocessor


def build_application(database_url: str | None = None, enable_qdrant: bool | None = None) -> ContentApplication:
    database = Database(database_url)
    database.create_schema()
    tasks = PostgresContentTaskRepository(database.session_factory)
    videos = PostgresVideoRepository(database.session_factory)
    chapters = PostgresChapterRepository(database.session_factory)
    knowledge = PostgresKnowledgeRepository(database.session_factory)
    multimodal = PostgresMultimodalRepository(database.session_factory)
    financial = PostgresFinancialRepository(database.session_factory)
    entities = PostgresFinancialEntityRepository(database.session_factory)
    verifications = PostgresVerificationRepository(database.session_factory)
    summaries = PostgresSummaryRepository(database.session_factory)
    use_qdrant = enable_qdrant if enable_qdrant is not None else bool(os.getenv("CONTENT_QDRANT_URL"))
    index = QdrantKnowledgeIndex() if use_qdrant else NullKnowledgeIndex()
    # §87：可选 QuantFactClient 仅用于 external fact verification；
    # 核心 ingestion 不强依赖 Quant。
    if os.getenv("CONTENT_EXTERNAL_FACT_PROVIDER", "").lower() == "quant":
        quant_provider = QuantExternalFactProvider()
        external_provider = quant_provider if quant_provider.configured() else HttpExternalFactProvider()
    else:
        external_provider = HttpExternalFactProvider()
    sources = {"bilibili": BilibiliSourceAdapter(), "xiaoe_hls": XiaoeHlsSourceAdapter()}
    pipeline = ContentPipeline(
        [
            ResolveSourceStage(sources),
            DownloadStage(sources),
            FrameExtractionStage(FfmpegFrameExtractor()),
            AudioStage(FfmpegAudioExtractor()),
            ASRStage(FasterWhisperRecognizer()),
            SpeakerDiarizationStage(PyannoteDiarizer()),
            TranscriptPostprocessStage(TranscriptPostprocessor()),
            OCRStage(PaddleOcrEngine()),
            VisionStage(HttpVisionAnalyzer()),
            MultimodalContextStage(MultimodalContextBuilder()),
            BuildVideoStage(),
            ChapterStage(ChapterSegmenter()),
            TemporalWindowStage(TemporalWindowBuilder()),
            KnowledgeExtractionStage(
                external_verifier=ExternalFactVerifier(
                    provider=external_provider if external_provider.configured() else None
                )
            ),
            VerificationStage(),
            FinancialEnrichmentStage(),
            SummaryStage(SummaryGenerator()),
            PersistStage(videos, chapters, knowledge, summaries, multimodal, financial, entities, verifications),
            IndexStage(index),
        ]
    )
    return ContentApplication(
        tasks,
        videos,
        knowledge,
        index,
        pipeline,
        chapters,
        summaries,
        snapshot_service=SnapshotService(SqlSnapshotStore(database.session_factory)),
    )

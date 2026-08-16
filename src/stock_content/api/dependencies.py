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
from stock_content.application.stage_runner import wrap_all
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
    SnapshotRecordingStage,
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

# P0 C-01：生产 Stage 版本常量（稳定、可追溯，断点恢复依赖版本兼容判定）。
STAGE_VERSIONS: dict[str, str] = {
    "resolve": "1.0.0",
    "download": "1.0.0",
    "frame": "1.0.0",
    "audio": "1.0.0",
    "asr": "1.0.0",
    "diarization": "1.0.0",
    "transcript_postprocess": "1.0.0",
    "ocr": "1.0.0",
    "vision": "1.0.0",
    "multimodal_context": "1.0.0",
    "transcript": "1.0.0",  # BuildVideoStage.name == "transcript"
    "chapter": "1.0.0",
    "temporal_window": "1.0.0",
    "knowledge": "1.0.0",
    "verification": "1.0.0",
    "financial_enrichment": "1.0.0",
    "summary": "1.0.0",
    "content_snapshot": "1.0.0",
    "persist": "1.0.0",
    "index": "1.0.0",
}


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
    # P0 C-03/C-04：快照在 persist 前基于 pipeline 已生成 Artifact 记录；失败即 task 失败。
    snapshot_service = SnapshotService(SqlSnapshotStore(database.session_factory))
    stages = [
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
        SnapshotRecordingStage(snapshot_service),
        PersistStage(videos, chapters, knowledge, summaries, multimodal, financial, entities, verifications),
        IndexStage(index),
    ]
    # P0 C-01：生产主链路全部经 StageRunner，产出 Artifact Checkpoint v2。
    # 版本号来自稳定常量，不得随机生成。
    pipeline = ContentPipeline(wrap_all(stages, STAGE_VERSIONS))
    return ContentApplication(
        tasks,
        videos,
        knowledge,
        index,
        pipeline,
        chapters,
        summaries,
        snapshot_service=snapshot_service,
    )

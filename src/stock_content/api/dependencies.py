from __future__ import annotations

import os

from stock_content.adapters.media import FasterWhisperRecognizer, FfmpegAudioExtractor
from stock_content.adapters.postgres import Database
from stock_content.adapters.postgres.repositories import (
    PostgresChapterRepository,
    PostgresContentTaskRepository,
    PostgresKnowledgeRepository,
    PostgresSummaryRepository,
    PostgresVideoRepository,
)
from stock_content.adapters.qdrant import NullKnowledgeIndex, QdrantKnowledgeIndex
from stock_content.adapters.sources import BilibiliSourceAdapter, XiaoeHlsSourceAdapter
from stock_content.application.pipeline import ContentPipeline
from stock_content.application.service import ContentApplication
from stock_content.application.stages import (
    ASRStage,
    AudioStage,
    BuildVideoStage,
    ChapterStage,
    DownloadStage,
    IndexStage,
    KnowledgeExtractionStage,
    PersistStage,
    ResolveSourceStage,
    SummaryStage,
    VerificationStage,
)
from stock_content.domain.chapter import ChapterSegmenter
from stock_content.domain.knowledge import KnowledgeExtractor
from stock_content.domain.summary import SummaryGenerator


def build_application(database_url: str | None = None, enable_qdrant: bool | None = None) -> ContentApplication:
    database = Database(database_url)
    database.create_schema()
    tasks = PostgresContentTaskRepository(database.session_factory)
    videos = PostgresVideoRepository(database.session_factory)
    chapters = PostgresChapterRepository(database.session_factory)
    knowledge = PostgresKnowledgeRepository(database.session_factory)
    summaries = PostgresSummaryRepository(database.session_factory)
    use_qdrant = enable_qdrant if enable_qdrant is not None else bool(os.getenv("CONTENT_QDRANT_URL"))
    index = QdrantKnowledgeIndex() if use_qdrant else NullKnowledgeIndex()
    sources = {"bilibili": BilibiliSourceAdapter(), "xiaoe_hls": XiaoeHlsSourceAdapter()}
    pipeline = ContentPipeline(
        [
            ResolveSourceStage(sources),
            DownloadStage(sources),
            AudioStage(FfmpegAudioExtractor()),
            ASRStage(FasterWhisperRecognizer()),
            BuildVideoStage(),
            ChapterStage(ChapterSegmenter()),
            KnowledgeExtractionStage(KnowledgeExtractor()),
            VerificationStage(),
            SummaryStage(SummaryGenerator()),
            PersistStage(videos, chapters, knowledge, summaries),
            IndexStage(index),
        ]
    )
    return ContentApplication(tasks, videos, knowledge, index, pipeline, chapters, summaries)

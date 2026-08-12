from __future__ import annotations

import hashlib
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_content.application.pipeline import PipelineContext
from stock_content.domain.chapter import ChapterSegmenter
from stock_content.domain.knowledge import KnowledgeExtractor
from stock_content.domain.models import TranscriptSegment, VideoAsset
from stock_content.domain.summary import SummaryGenerator
from stock_content.ports.media import AudioExtractor, SourceAdapter, SpeechRecognizer
from stock_content.ports.repositories import (
    ChapterRepository,
    KnowledgeIndex,
    KnowledgeRepository,
    SummaryRepository,
    VideoRepository,
)


class ResolveSourceStage:
    name = "resolve"

    def __init__(self, adapters: dict[str, SourceAdapter]) -> None:
        self._adapters = adapters

    def execute(self, context: PipelineContext) -> PipelineContext:
        fixture = context.options.get("metadata")
        context.data["metadata"] = fixture or self._adapters[context.source["type"]].resolve(context.source["ref"])
        return context


class DownloadStage:
    name = "download"

    def __init__(self, adapters: dict[str, SourceAdapter], work_root: Path | None = None) -> None:
        self._adapters = adapters
        self._work_root = work_root

    def execute(self, context: PipelineContext) -> PipelineContext:
        if "transcript" in context.options or "segments" in context.options:
            return context
        directory = Path(tempfile.mkdtemp(prefix=f"content-{context.task_id[:8]}-", dir=self._work_root))
        context.data["work_dir"] = directory
        context.data["video_path"] = self._adapters[context.source["type"]].download(
            context.source["ref"], directory
        )
        return context


def cleanup_work_directory(context: PipelineContext) -> None:
    directory = context.data.get("work_dir")
    if isinstance(directory, Path) and directory.name.startswith(f"content-{context.task_id[:8]}-"):
        shutil.rmtree(directory, ignore_errors=True)


class AudioStage:
    name = "audio"

    def __init__(self, extractor: AudioExtractor) -> None:
        self._extractor = extractor

    def execute(self, context: PipelineContext) -> PipelineContext:
        if "video_path" in context.data:
            context.data["audio_path"] = self._extractor.extract(
                context.data["video_path"], context.data["work_dir"]
            )
        return context


class ASRStage:
    name = "asr"

    def __init__(self, recognizer: SpeechRecognizer) -> None:
        self._recognizer = recognizer

    @staticmethod
    def _fixture(options: dict[str, Any]) -> list[TranscriptSegment]:
        raw_segments = options.get("segments")
        if raw_segments:
            return [TranscriptSegment(segment_index=index, **item) for index, item in enumerate(raw_segments)]
        transcript = str(options.get("transcript") or "").strip()
        if not transcript:
            return []
        return [
            TranscriptSegment(
                segment_index=0,
                start_seconds=0,
                end_seconds=max(1.0, len(transcript) / 4),
                text=transcript,
                confidence=1.0,
            )
        ]

    def execute(self, context: PipelineContext) -> PipelineContext:
        segments = self._fixture(context.options)
        if not segments and "audio_path" in context.data:
            segments = self._recognizer.transcribe(context.data["audio_path"], context.options.get("language"))
        if not segments:
            raise ValueError("ASR returned no transcript segments")
        context.data["segments"] = segments
        context.data["transcript"] = " ".join(segment.text for segment in segments)
        return context


class ChapterStage:
    name = "chapter"

    def __init__(self, segmenter: ChapterSegmenter) -> None:
        self._segmenter = segmenter

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.data["chapters"] = self._segmenter.segment(context.data["segments"])
        return context


class KnowledgeExtractionStage:
    name = "knowledge"

    def __init__(self, extractor: KnowledgeExtractor) -> None:
        self._extractor = extractor

    def execute(self, context: PipelineContext) -> PipelineContext:
        as_of = context.options.get("as_of")
        if isinstance(as_of, str):
            as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        context.data["knowledge"] = self._extractor.extract(
            context.data["video"].video_id,
            context.data["chapters"],
            as_of or datetime.now(UTC),
        )
        return context


class VerificationStage:
    name = "verification"

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = context.data["transcript"]
        for unit in context.data["knowledge"]:
            unit.support_status = "SOURCE_SUPPORTED" if unit.statement in transcript else "UNSUPPORTED"
            unit.confidence = 0.75 if unit.support_status == "SOURCE_SUPPORTED" else 0.3
        return context


class SummaryStage:
    name = "summary"

    def __init__(self, generator: SummaryGenerator) -> None:
        self._generator = generator

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.data["summary"] = self._generator.generate(
            context.data["video"], context.data["chapters"], context.data["knowledge"]
        )
        return context


class PersistStage:
    name = "persist"

    def __init__(
        self,
        videos: VideoRepository,
        chapters: ChapterRepository,
        knowledge: KnowledgeRepository,
        summaries: SummaryRepository,
    ) -> None:
        self._videos = videos
        self._chapters = chapters
        self._knowledge = knowledge
        self._summaries = summaries

    def execute(self, context: PipelineContext) -> PipelineContext:
        video = context.data["video"]
        self._videos.upsert(video, context.data["segments"])
        self._chapters.replace_for_video(video.video_id, context.data["chapters"])
        self._knowledge.replace_for_video(video.video_id, context.data["knowledge"])
        self._summaries.upsert(context.data["summary"])
        return context


class IndexStage:
    name = "index"

    def __init__(self, index: KnowledgeIndex) -> None:
        self._index = index

    def execute(self, context: PipelineContext) -> PipelineContext:
        self._index.index(context.data["knowledge"])
        return context


class BuildVideoStage:
    name = "transcript"

    def execute(self, context: PipelineContext) -> PipelineContext:
        metadata = context.data["metadata"]
        source_key = f'{context.source["type"]}:{context.source["ref"]}'
        video_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:32]
        transcript = context.data["transcript"]
        context.data["video"] = VideoAsset(
            video_id=video_id,
            source_type=context.source["type"],
            source_ref=context.source["ref"],
            title=metadata.get("title") or context.source["ref"],
            author=metadata.get("author"),
            duration_seconds=metadata.get("duration_seconds"),
            transcript_text=transcript,
            source_hash=hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        )
        return context

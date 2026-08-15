from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_content.adapters.http.model_client import ContentModelClient
from stock_content.application.pipeline import PipelineContext
from stock_content.domain.chapter import ChapterSegmenter
from stock_content.domain.claim_evidence_verifier import ClaimEvidenceVerifier
from stock_content.domain.cross_modal_evidence_verifier import CrossModalEvidenceVerifier
from stock_content.domain.external_fact_verifier import ExternalFactVerifier
from stock_content.domain.financial_event_extractor import FinancialEventExtractor
from stock_content.domain.financial_numeric import parse_financial_numerics
from stock_content.domain.knowledge import KnowledgeExtractor
from stock_content.domain.knowledge_deduplicator import KnowledgeDeduplicator
from stock_content.domain.knowledge_temporal_policy import KnowledgeTemporalPolicy
from stock_content.domain.knowledge_unit_extractor import KnowledgeUnitExtractor
from stock_content.domain.knowledge_unit_normalizer import KnowledgeUnitNormalizer
from stock_content.domain.models import KnowledgeUnit, TranscriptSegment, VideoAsset
from stock_content.domain.semantic_entailment_judge import SemanticEntailmentJudge
from stock_content.domain.summary import SummaryGenerator
from stock_content.domain.transcript_postprocessor import TranscriptPostprocessor
from stock_content.ports.media import AudioExtractor, SourceAdapter, SpeechRecognizer
from stock_content.ports.repositories import (
    ChapterRepository,
    KnowledgeIndex,
    KnowledgeRepository,
    MultimodalRepository,
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
        context.data["video_path"] = self._adapters[context.source["type"]].download(context.source["ref"], directory)
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
            context.data["audio_path"] = self._extractor.extract(context.data["video_path"], context.data["work_dir"])
        return context


class FrameExtractionStage:
    name = "frame"

    def __init__(self, extractor) -> None:
        self._extractor = extractor

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.options.get("frames") is not None:
            context.data["frames"] = list(context.options["frames"])
        elif "video_path" in context.data:
            context.data["frames"] = self._extractor.extract(context.data["video_path"], context.data["work_dir"])
        else:
            context.data["frames"] = []
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


class SpeakerDiarizationStage:
    name = "diarization"

    def __init__(self, diarizer) -> None:
        self._diarizer = diarizer

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.data["segments"] = self._diarizer.annotate(
            str(context.data.get("audio_path") or "") or None, context.data["segments"]
        )
        status = getattr(self._diarizer, "last_status", "UNKNOWN")
        context.data["diarization_status"] = status
        if status in {"UNAVAILABLE", "FAILED", "DEGRADED"}:
            context.data.setdefault("quality_warnings", []).append(f"DIARIZATION_{status}")
        return context


class TranscriptPostprocessStage:
    name = "transcript_postprocess"

    def __init__(self, postprocessor: TranscriptPostprocessor) -> None:
        self._postprocessor = postprocessor

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.data["segments"] = self._postprocessor.process(context.data["segments"])
        context.data["transcript"] = " ".join(segment.text for segment in context.data["segments"])
        return context


class OCRStage:
    name = "ocr"

    def __init__(self, engine) -> None:
        self._engine = engine

    def execute(self, context: PipelineContext) -> PipelineContext:
        supplied = context.options.get("ocr_evidence")
        evidence = list(supplied) if supplied is not None else []
        if supplied is None:
            for frame in context.data.get("frames", []):
                result = self._engine.recognize(str(frame["image_path"]))
                blocks = list(result.get("blocks") or [])
                item = {
                    **frame,
                    "ocr_text": str(result.get("text") or ""),
                    "ocr_evidence": {"blocks": blocks},
                    "ocr_engine": result.get("engine"),
                    "ocr_engine_version": result.get("engine_version"),
                }
                context.data.setdefault("frame_insights", []).append(item)
                evidence.extend(
                    [
                        {
                            **block,
                            "frame_id": frame["frame_id"],
                            "timestamp_ms": frame["timestamp_ms"],
                            "source_type": "OCR",
                            "confidence_score": block.get("score"),
                            "evidence_text": block.get("text"),
                        }
                        for block in blocks
                    ]
                )
        context.data["ocr_evidence"] = evidence
        return context


class VisionStage:
    name = "vision"

    def __init__(self, analyzer) -> None:
        self._analyzer = analyzer

    def execute(self, context: PipelineContext) -> PipelineContext:
        insights = list(context.data.get("frame_insights") or [])
        if context.options.get("frame_insights") is not None:
            insights.extend(context.options["frame_insights"])
        elif context.data.get("frames"):
            for frame in context.data["frames"]:
                result = self._analyzer.analyze(str(frame["image_path"]), context.data["transcript"])
                insights.append({**frame, **result, "model": getattr(self._analyzer, "_model", None)})
        context.data["frame_insights"] = insights
        return context


class MultimodalContextStage:
    name = "multimodal_context"

    def __init__(self, builder) -> None:
        self._builder = builder

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = {
            "segments": [
                {"text": item.text, "start_ms": int(item.start_seconds * 1000), "end_ms": int(item.end_seconds * 1000)}
                for item in context.data["segments"]
            ]
        }
        context.data["multimodal_context"] = self._builder.build(transcript, context.data.get("frame_insights") or [])
        return context


class TemporalWindowStage:
    name = "temporal_window"

    def __init__(self, builder) -> None:
        self._builder = builder

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = {
            "segments": [
                {
                    "text": item.text,
                    "start_ms": int(item.start_seconds * 1000),
                    "end_ms": int(item.end_seconds * 1000),
                    "speaker_id": item.speaker_id,
                    "confidence_score": item.confidence,
                }
                for item in context.data["segments"]
            ]
        }
        context.data["temporal_windows"] = self._builder.build(transcript, context.data.get("frame_insights") or [])
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

    def __init__(
        self,
        model_client: ContentModelClient | None = None,
        fixture_extractor: KnowledgeExtractor | None = None,
        external_verifier: ExternalFactVerifier | None = None,
    ) -> None:
        self._model_client = model_client or ContentModelClient()
        self._structured_extractor = KnowledgeUnitExtractor(self._model_client)
        self._normalizer = KnowledgeUnitNormalizer(
            verifier=ClaimEvidenceVerifier(judge=SemanticEntailmentJudge(self._model_client))
        )
        self._cross_modal = CrossModalEvidenceVerifier()
        self._temporal = KnowledgeTemporalPolicy()
        self._deduplicator = KnowledgeDeduplicator()
        self._external = external_verifier or ExternalFactVerifier()
        self._fixture_extractor = fixture_extractor or KnowledgeExtractor()

    @staticmethod
    def _timestamp(value: object) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return datetime.now(UTC)

    @staticmethod
    def _chapter_payload(context: PipelineContext) -> list[dict]:
        payload = []
        for chapter in context.data["chapters"]:
            segments = [
                segment
                for segment in context.data["segments"]
                if segment.start_seconds < chapter.end_seconds and chapter.start_seconds < segment.end_seconds
            ]
            window = {
                "start_ms": int(chapter.start_seconds * 1000),
                "end_ms": int(chapter.end_seconds * 1000),
                "transcript_text": " ".join(segment.text for segment in segments),
                "segments": [
                    {
                        "text": segment.normalized_text or segment.text,
                        "raw_text": segment.raw_text or segment.text,
                        "start_ms": int(segment.start_seconds * 1000),
                        "end_ms": int(segment.end_seconds * 1000),
                        "confidence_score": segment.confidence,
                        "speaker_id": segment.speaker_id,
                    }
                    for segment in segments
                ],
                "ocr_blocks": list(context.data.get("ocr_evidence") or []),
                "frame_refs": list(context.data.get("frame_insights") or []),
            }
            payload.append(
                {
                    "chapter_index": chapter.chapter_index,
                    "title": chapter.title,
                    "chapter_type": chapter.chapter_type,
                    "primary_domain": "GENERAL",
                    "windows": [
                        item
                        for item in context.data.get("temporal_windows", [])
                        if int(item.get("start_ms") or 0) < int(chapter.end_seconds * 1000)
                        and int(chapter.start_seconds * 1000) < int(item.get("end_ms") or 0)
                    ]
                    or [window],
                    "entities": [],
                }
            )
        return payload

    @staticmethod
    def _to_domain(video_id: str, records: list[dict], available_from: datetime) -> list[KnowledgeUnit]:
        units = []
        for record in records:
            as_of = record.get("as_of_time") or available_from
            if isinstance(as_of, str):
                as_of = KnowledgeExtractionStage._timestamp(as_of)
            valid_from = record.get("valid_from")
            valid_to = record.get("valid_to")
            if isinstance(valid_from, str):
                valid_from = KnowledgeExtractionStage._timestamp(valid_from)
            if isinstance(valid_to, str):
                valid_to = KnowledgeExtractionStage._timestamp(valid_to)
            subject = record.get("subject_name") or record.get("subject_key")
            attributes = dict(record.get("attributes") or {})
            attributes["evidence"] = list(record.get("evidence") or [])
            attributes["event_type"] = record.get("event_type")
            units.append(
                KnowledgeUnit(
                    knowledge_uid=str(record["knowledge_uid"]),
                    video_id=video_id,
                    chapter_id=record.get("chapter_id"),
                    statement=str(record["statement"]),
                    kind=str(record.get("knowledge_kind") or "STATE"),
                    knowledge_kind=str(record.get("knowledge_kind") or "STATE"),
                    knowledge_version=int(record.get("knowledge_version") or 1),
                    subject=subject,
                    subject_key=record.get("subject_key"),
                    predicate_key=record.get("predicate_key"),
                    ticker=record.get("ticker") or record.get("subject_key"),
                    sentiment=str(record.get("sentiment") or "NEUTRAL"),
                    support_status=str(record.get("support_status") or "UNSUPPORTED"),
                    truth_status=str(record.get("truth_status") or "NOT_CHECKED"),
                    review_status=str(record.get("review_status") or "UNREVIEWED"),
                    lifecycle_status=str(record.get("lifecycle_status") or "EXTRACTED"),
                    confidence=float(record.get("support_score") or record.get("extraction_confidence") or 0.0),
                    as_of=as_of,
                    available_from=available_from,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    source_statement_hash=record.get("semantic_hash"),
                    content_hash=record.get("content_hash"),
                    attributes=attributes,
                    provenance={
                        "extractor_version": record.get("extractor_version"),
                        "schema_version": record.get("schema_version"),
                        "model": (record.get("attributes") or {}).get("model"),
                    },
                )
            )
        return units

    def _fixture_records(self, context: PipelineContext, timestamp: datetime) -> list[dict]:
        """Offline fixtures are explicit and never a production fallback."""
        records: list[dict] = []
        for unit in self._fixture_extractor.extract(
            context.data["video"].video_id, context.data["chapters"], timestamp
        ):
            chapter = next((item for item in context.data["chapters"] if item.chapter_id == unit.chapter_id), None)
            start_ms = int((chapter.start_seconds if chapter else 0) * 1000)
            end_ms = int((chapter.end_seconds if chapter else 0) * 1000)
            records.append(
                {
                    "knowledge_uid": unit.knowledge_uid,
                    "statement": unit.statement,
                    "knowledge_kind": "STATE",
                    "subject_key": unit.ticker or unit.subject,
                    "subject_name": unit.subject,
                    "predicate_key": unit.kind.lower(),
                    "ticker": unit.ticker,
                    "sentiment": unit.sentiment,
                    "support_status": "SOURCE_SUPPORTED",
                    "truth_status": "NOT_CHECKED",
                    "review_status": "UNREVIEWED",
                    "lifecycle_status": "ACTIVE",
                    "support_score": 0.75,
                    "as_of_time": timestamp,
                    "evidence": [
                        {
                            "source_type": "ASR",
                            "evidence_text": unit.statement,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "confidence_score": 1.0,
                            "is_primary": True,
                        }
                    ],
                    "attributes": {"offline_fixture": True},
                }
            )
        return records

    def execute(self, context: PipelineContext) -> PipelineContext:
        available_from = self._timestamp(context.options.get("available_from") or context.options.get("as_of"))
        if context.options.get("offline_fixture") is True:
            records = self._fixture_records(context, available_from)
        else:
            metadata = dict(context.data["metadata"])
            metadata.setdefault("platform", context.source["type"])
            metadata.setdefault("platform_video_id", context.source["ref"])
            metadata.setdefault("publish_time", available_from.isoformat())
            records = self._structured_extractor.extract(metadata, self._chapter_payload(context))
            records = self._normalizer.normalize(records, metadata)
            records = self._cross_modal.verify_many(records, list(context.data.get("ocr_evidence") or []))
            records = self._external.verify_many(records)
            records = self._temporal.apply(records, available_from)
            records = self._deduplicator.deduplicate(records)
        context.data["knowledge"] = self._to_domain(context.data["video"].video_id, records, available_from)
        return context


class VerificationStage:
    name = "verification"

    def execute(self, context: PipelineContext) -> PipelineContext:
        # Source, semantic and cross-modal verification are performed before
        # conversion to the persistence model.  This stage is retained as a
        # named checkpoint for worker compatibility and deliberately does not
        # reintroduce substring-based verification.
        return context


class FinancialEnrichmentStage:
    """Materialise numeric facts and events once for every downstream consumer."""

    name = "financial_enrichment"

    def __init__(self, extractor: FinancialEventExtractor | None = None) -> None:
        self._extractor = extractor or FinancialEventExtractor()

    def execute(self, context: PipelineContext) -> PipelineContext:
        records: list[dict] = []
        numeric_facts: list[dict] = []
        for unit in context.data["knowledge"]:
            numerics = [asdict(item) for item in parse_financial_numerics(unit.statement)]
            attributes = dict(unit.attributes or {})
            attributes["financial_numerics"] = numerics
            unit.attributes = attributes
            records.append(
                {
                    "knowledge_uid": unit.knowledge_uid,
                    "statement": unit.statement,
                    "subject_key": unit.subject_key,
                    "ticker": unit.ticker,
                    "sentiment": unit.sentiment,
                    "confidence": unit.confidence,
                    "as_of": unit.as_of,
                    "valid_from": unit.valid_from,
                    "available_from": unit.available_from,
                    "numeric_ids": [],
                    "evidence_ids": [],
                }
            )
            numeric_facts.extend([{"knowledge_uid": unit.knowledge_uid, **item} for item in numerics])
        context.data["financial_numeric_facts"] = numeric_facts
        context.data["financial_events"] = self._extractor.extract(records)
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
        multimodal: MultimodalRepository | None = None,
        financial=None,
    ) -> None:
        self._videos = videos
        self._chapters = chapters
        self._knowledge = knowledge
        self._summaries = summaries
        self._multimodal = multimodal
        self._financial = financial

    def execute(self, context: PipelineContext) -> PipelineContext:
        video = context.data["video"]
        self._videos.upsert(video, context.data["segments"])
        self._chapters.replace_for_video(video.video_id, context.data["chapters"])
        self._knowledge.replace_for_video(video.video_id, context.data["knowledge"])
        if self._multimodal:
            self._multimodal.replace(
                video.video_id,
                list(context.data.get("frames") or []),
                list(context.data.get("ocr_evidence") or []),
                list(context.data.get("frame_insights") or []),
                list(context.data.get("temporal_windows") or []),
            )
        if self._financial:
            self._financial.replace(
                video.video_id,
                list(context.data.get("financial_numeric_facts") or []),
                list(context.data.get("financial_events") or []),
            )
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
        source_key = f"{context.source['type']}:{context.source['ref']}"
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

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from stock_content.adapters.http.model_client import ContentModelClient
from stock_content.application.pipeline import PipelineContext
from stock_content.application.snapshot_service import SnapshotService, choose_snapshot_commit_candidate
from stock_content.application.stage_runner import StageResult
from stock_content.domain.artifacts import (
    ClaimArtifact,
    ClaimOccurrenceArtifact,
    EvidenceArtifact,
    EvidenceItem,
    FrameArtifact,
    KnowledgeArtifact,
    LifecycleArtifact,
    MediaArtifact,
    OCRArtifact,
    SourceArtifact,
    SummaryArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    VerificationArtifact,
    VisionArtifact,
    artifact_id_of,
    canonical_json,
)
from stock_content.domain.atomic_claim_extractor import AtomicClaimExtractor
from stock_content.domain.chapter import ChapterSegmenter
from stock_content.domain.claim_canonicalizer import ClaimCanonicalizer
from stock_content.domain.claim_draft import ClaimOccurrenceDraft
from stock_content.domain.claim_draft_grounder import ClaimDraftGrounder
from stock_content.domain.claim_evidence_verifier import ClaimEvidenceVerifier
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.cross_modal_evidence_verifier import CrossModalEvidenceVerifier
from stock_content.domain.external_fact_verifier import ExternalFactVerifier
from stock_content.domain.financial_event_extractor import FinancialEventExtractor
from stock_content.domain.financial_numeric import parse_financial_numerics
from stock_content.domain.knowledge import KnowledgeExtractor
from stock_content.domain.knowledge_deduplicator import KnowledgeDeduplicator
from stock_content.domain.knowledge_projection_builder import KnowledgeProjectionBuilder
from stock_content.domain.knowledge_temporal_policy import KnowledgeTemporalPolicy
from stock_content.domain.knowledge_unit_extractor import KnowledgeUnitExtractor
from stock_content.domain.knowledge_unit_normalizer import KnowledgeUnitNormalizer
from stock_content.domain.lifecycle_event import KnowledgeLifecycleEvent
from stock_content.domain.lineage import default_code_sha
from stock_content.domain.models import KnowledgeUnit, TranscriptSegment, VideoAsset
from stock_content.domain.semantic_context_builder import SemanticContextBuilder
from stock_content.domain.semantic_entailment_judge import SemanticEntailmentJudge
from stock_content.domain.semantic_segmenter import SemanticSegmenter
from stock_content.domain.summary import SummaryGenerator
from stock_content.domain.temporal_normalizer import TemporalNormalizer
from stock_content.domain.temporal_semantics import (
    MetricTemporalNature,
    OccurrenceTimes,
    TemporalAssertionStatus,
    TemporalRole,
    TemporalScope,
)
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


def _stage_result(context: PipelineContext, *slots: str) -> StageResult:
    produced = []
    for slot in slots:
        value = context.artifacts.get(slot)
        if isinstance(value, list):
            produced.extend(value)
        elif value is not None:
            produced.append(value)
    return StageResult(context=context, produced_artifacts=tuple(produced))


def _resolved_datetime(value: Any, field_name: str) -> datetime | None:
    """Parse resolver-provided timestamps with one explicit legacy policy.

    Resolver payloads historically contained both ISO strings and Unix epoch
    timestamps.  Keep both representations accepted, but reject malformed or
    timezone-invalid values instead of silently substituting a task clock.
    Naive values retain the repository's existing policy: interpret them as
    UTC, then normalize every accepted value to UTC for lineage equality.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"invalid {field_name}: non-finite timestamp")
        try:
            parsed = datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"invalid {field_name}: {value!r}") from exc
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"invalid {field_name}: empty timestamp")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field_name}: {value!r}") from exc
    else:
        raise ValueError(f"invalid {field_name}: expected datetime, ISO timestamp, or Unix timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if parsed.utcoffset() is None:
        raise ValueError(f"invalid {field_name}: timezone offset is unavailable")
    return parsed.astimezone(UTC)


class ResolveSourceStage:
    name = "resolve"
    # Resolution only yields metadata.  The source artifact is created after
    # download/fixture materialization has established the authoritative hash.
    output_types = ()

    def __init__(self, adapters: dict[str, SourceAdapter]) -> None:
        self._adapters = adapters

    def execute(self, context: PipelineContext) -> PipelineContext:
        fixture = context.options.get("metadata")
        context.state["metadata"] = fixture or self._adapters[context.source["type"]].resolve(context.source["ref"])
        return _stage_result(context)


def _source_version_id(source_identity_hash: str, raw_hash: str) -> str:
    return "source-version-" + hashlib.sha256(f"{source_identity_hash}:{raw_hash}".encode()).hexdigest()[:32]


def _durable_cache_dir(context: PipelineContext, category: str = "raw") -> Path:
    configured = context.options.get("raw_storage_dir") or os.getenv("CONTENT_RAW_STORAGE_DIR")
    root = Path(str(configured)) if configured else Path(tempfile.gettempdir()) / "stock-content-raw"
    directory = root / category
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _persist_durable_file(context: PipelineContext, source: Path, digest: str, category: str) -> Path:
    target = _durable_cache_dir(context, category) / f"{digest}{source.suffix.lower()}"
    if target.is_file():
        existing_digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                existing_digest.update(chunk)
        if existing_digest.hexdigest() != digest:
            raise RuntimeError(f"ARTIFACT_INTEGRITY_ERROR: durable cache hash mismatch for {target.name}")
        return target
    temporary = target.with_name(f".{target.name}.{context.task_id}.tmp")
    shutil.copyfile(source, temporary)
    try:
        temporary.replace(target)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
    return target


def _stable_fixture_media_hash(context: PipelineContext) -> tuple[str, int]:
    explicit = str(context.options.get("source_content_hash") or context.options.get("raw_content_hash") or "")
    if explicit:
        return explicit, len(explicit.encode("utf-8"))
    payload = {
        "transcript": context.options.get("transcript") or "",
        "segments": context.options.get("segments") or [],
        "frames": context.options.get("frames") or [],
        "ocr_evidence": context.options.get("ocr_evidence") or [],
        "frame_insights": context.options.get("frame_insights") or [],
    }
    raw = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _refresh_source_artifact(context: PipelineContext, raw_hash: str, length: int, uri: str | None = None) -> None:
    existing = context.artifacts.source
    source_type = existing.source_type if existing else str(context.source.get("type") or "")
    source_ref = existing.source_ref if existing else str(context.source.get("ref") or "")
    metadata = dict(existing.source_metadata if existing else context.state.metadata)
    identity_hash = hashlib.sha256(f"{source_type}:{source_ref}".encode()).hexdigest()
    source = SourceArtifact(
        artifact_id="source-pending",
        artifact_type="source",
        producer_stage="download",
        source_type=source_type,
        source_ref=source_ref,
        source_content_hash=raw_hash,
        raw_content_hash=raw_hash,
        raw_content_length=length,
        raw_storage_uri=uri,
        source_identity_hash=identity_hash,
        source_version_id=_source_version_id(identity_hash, raw_hash),
        source_metadata=metadata,
    )
    context.artifacts.source = SourceArtifact(**{**source.__dict__, "artifact_id": artifact_id_of(source)})


class DownloadStage:
    name = "download"
    required_inputs = ()
    output_types = ("source", "media")

    def __init__(self, adapters: dict[str, SourceAdapter], work_root: Path | None = None) -> None:
        self._adapters = adapters
        self._work_root = work_root

    def execute(self, context: PipelineContext) -> PipelineContext:
        fixture = bool(
            context.options.get("offline_fixture")
            or "transcript" in context.options
            or "segments" in context.options
        )
        if fixture:
            # Offline fixtures are synthetic, deterministic sources.  Give
            # them an explicit availability boundary so the public-strict
            # default remains meaningful without treating UNKNOWN as public.
            fixture_available = (
                context.options.get("source_available_at")
                or context.state.metadata.get("source_available_at")
                or context.state.metadata.get("available_at")
                or context.options.get("available_from")
                or context.options.get("as_of")
            )
            if fixture_available is None:
                fixture_available = datetime.now(UTC)
            context.options.setdefault("source_available_at", fixture_available)
            context.options.setdefault("source_availability_quality", "EXACT")
        if not fixture:
            # Capture once at immutable download completion.  Downstream
            # replay uses this value instead of deriving a new wall clock.
            context.options.setdefault("ingested_at", datetime.now(UTC))
            explicit_available = (
                context.options.get("source_available_at")
                or context.state.metadata.get("source_available_at")
                or context.state.metadata.get("available_at")
            )
            if explicit_available is not None:
                context.options.setdefault("source_available_at", explicit_available)
                context.options.setdefault(
                    "source_availability_quality",
                    context.state.metadata.get("source_availability_quality") or "EXACT",
                )
            else:
                # Download completion is an upper bound, not proof of public
                # availability.  A published timestamp is only a proxy when
                # the adapter explicitly supplies one.
                context.options.setdefault("source_available_at", context.options["ingested_at"])
                context.options.setdefault("source_availability_quality", "INGEST_TIME_UPPER_BOUND")
        replay_uri = context.options.get("replay_raw_storage_uri")
        if replay_uri:
            context.runtime.work_dir = Path(
                tempfile.mkdtemp(prefix=f"content-{context.task_id[:8]}-", dir=self._work_root)
            )
            path = Path(str(replay_uri).removeprefix("file://"))
            if not path.is_file():
                raise RuntimeError("REPLAY_INPUT_UNAVAILABLE: durable raw media is missing")
            context.runtime.video_path = path
            raw = hashlib.sha256()
            length = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    raw.update(chunk)
                    length += len(chunk)
            expected_hash = str(context.options.get("replay_expected_raw_hash") or "")
            if expected_hash and raw.hexdigest() != expected_hash:
                raise RuntimeError("REPLAY_INPUT_UNAVAILABLE: durable raw media hash mismatch")
            _refresh_source_artifact(context, raw.hexdigest(), length, str(path))
            source = context.artifacts.source
            if source is not None:
                media = MediaArtifact(
                    artifact_id="media-pending",
                    artifact_type="media",
                    source_artifact_id=source.artifact_id,
                    media_uri=str(path),
                    video_hash=raw.hexdigest(),
                    extractor_version="download.v1",
                    parent_artifact_ids=(source.artifact_id,),
                )
                context.artifacts.media = MediaArtifact(
                    **{**media.__dict__, "artifact_id": artifact_id_of(media)}
                )
            return _stage_result(context, "source", "media")
        if "transcript" in context.options or "segments" in context.options:
            raw_hash, length = _stable_fixture_media_hash(context)
            _refresh_source_artifact(context, raw_hash, length, "fixture://media")
            source = context.artifacts.source
            if source is not None:
                media = MediaArtifact(
                    artifact_id="media-pending",
                    artifact_type="media",
                    source_artifact_id=source.artifact_id,
                    media_uri="fixture://media",
                    video_hash=raw_hash,
                    extractor_version="fixture.v1",
                    parent_artifact_ids=(source.artifact_id,),
                )
                context.artifacts.media = MediaArtifact(**{**media.__dict__, "artifact_id": artifact_id_of(media)})
            return _stage_result(context, "source", "media")
        directory = Path(tempfile.mkdtemp(prefix=f"content-{context.task_id[:8]}-", dir=self._work_root))
        context.runtime.work_dir = directory
        context.runtime.video_path = self._adapters[context.source["type"]].download(context.source["ref"], directory)
        raw = hashlib.sha256()
        length = 0
        with Path(context.runtime.video_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                raw.update(chunk)
                length += len(chunk)
        durable_path = _persist_durable_file(context, Path(context.runtime.video_path), raw.hexdigest(), "raw")
        context.runtime.video_path = durable_path
        _refresh_source_artifact(context, raw.hexdigest(), length, str(durable_path))
        source = context.artifacts.source
        if source is not None:
            media = MediaArtifact(
                artifact_id="media-pending",
                artifact_type="media",
                source_artifact_id=source.artifact_id,
                media_uri=str(context.runtime.video_path),
                video_hash=raw.hexdigest(),
                extractor_version="download.v1",
                parent_artifact_ids=(source.artifact_id,),
            )
            context.artifacts.media = MediaArtifact(**{**media.__dict__, "artifact_id": artifact_id_of(media)})
        return _stage_result(context, "source", "media")


def cleanup_work_directory(context: PipelineContext) -> None:
    directory = context.runtime.work_dir
    if isinstance(directory, Path) and directory.name.startswith(f"content-{context.task_id[:8]}-"):
        shutil.rmtree(directory, ignore_errors=True)


class AudioStage:
    name = "audio"
    required_inputs = ("media",)
    output_types = ()

    def __init__(self, extractor: AudioExtractor) -> None:
        self._extractor = extractor

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.runtime.video_path is not None:
            context.runtime.audio_path = self._extractor.extract(context.runtime.video_path, context.runtime.work_dir)
        return _stage_result(context)


class FrameExtractionStage:
    name = "frame"
    required_inputs = ("media",)
    output_types = ("frame",)
    optional_output_types = ("frame",)

    def __init__(self, extractor) -> None:
        self._extractor = extractor

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.options.get("frames") is not None:
            context.state["frames"] = list(context.options["frames"])
        elif context.runtime.video_path is not None:
            context.state["frames"] = self._extractor.extract(context.runtime.video_path, context.runtime.work_dir)
        else:
            context.state["frames"] = []
        supplied_ids = {
            str(item.get("frame_id"))
            for key in ("ocr_evidence", "frame_insights")
            for item in (context.options.get(key) or [])
            if isinstance(item, dict) and item.get("frame_id")
        }
        existing_ids = {
            str(item.get("frame_id"))
            for item in context.state.frames
            if isinstance(item, dict) and item.get("frame_id")
        }
        for frame_id in sorted(supplied_ids - existing_ids):
            context.state.frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": 0,
                    "image_hash": hashlib.sha256(frame_id.encode()).hexdigest(),
                    "storage_ref": f"fixture://frame/{frame_id}",
                }
            )
        media = context.artifacts.media
        if media is not None:
            frame_artifacts = []
            for index, frame in enumerate(context.state.frames):
                item = frame if isinstance(frame, dict) else {"image_path": str(frame)}
                image_path = item.get("image_path")
                durable_image = None
                if image_path and Path(str(image_path)).is_file() and not str(image_path).startswith("fixture://"):
                    durable_image = _persist_durable_file(
                        context, Path(str(image_path)), self._image_hash(item), "frames"
                    )
                    item["image_path"] = str(durable_image)
                frame_artifact = FrameArtifact(
                    artifact_id="frame-pending",
                    artifact_type="frame",
                    media_artifact_id=media.artifact_id,
                    frame_id=str(item.get("frame_id") or f"frame-{index}"),
                    timestamp_ms=int(item.get("timestamp_ms") or 0),
                    image_hash=self._image_hash(item),
                    storage_ref=str(durable_image or item.get("image_path") or item.get("storage_ref") or ""),
                    extraction_reason=str(item.get("extraction_reason") or "fixture"),
                    parent_artifact_ids=(media.artifact_id,),
                )
                frame_artifacts.append(
                    FrameArtifact(
                        **{**frame_artifact.__dict__, "artifact_id": artifact_id_of(frame_artifact)}
                    )
                )
            context.artifacts.frames = frame_artifacts
        return _stage_result(context, "frames")

    @staticmethod
    def _image_hash(item: dict[str, Any]) -> str:
        image_path = item.get("image_path") or item.get("storage_ref")
        if image_path:
            path = Path(str(image_path))
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()
        return str(item.get("image_hash") or hashlib.sha256(canonical_json(item).encode()).hexdigest())


class ASRStage:
    name = "asr"
    required_inputs = ("media",)
    output_types = ("transcript",)

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
        if not segments and context.runtime.audio_path is not None:
            segments = self._recognizer.transcribe(context.runtime.audio_path, context.options.get("language"))
        if not segments:
            raise ValueError("ASR returned no transcript segments")
        context.state["segments"] = segments
        context.state["transcript"] = " ".join(segment.text for segment in segments)
        _register_transcript_artifact(context, producer_stage="asr")
        return _stage_result(context, "transcript")


def _register_transcript_artifact(
    context: PipelineContext, *, producer_stage: str, parent_artifact_id: str | None = None
) -> None:
    """P0 C-02：TranscriptArtifact 登记/升级（typed artifact 为权威，data 仅作 adapter）。"""
    segments = [
        TranscriptSegmentItem(
            segment_index=item.segment_index,
            start_seconds=item.start_seconds,
            end_seconds=item.end_seconds,
            text=item.text,
            confidence=item.confidence,
            speaker_id=item.speaker_id,
        )
        for item in context.state["segments"]
    ]
    source = context.artifacts.source
    transcript = TranscriptArtifact(
        artifact_id="transcript-pending",
        artifact_type="transcript",
        producer_stage=producer_stage,
        media_artifact_id=(
            context.artifacts.media.artifact_id
            if context.artifacts.media
            else (source.artifact_id if source else "")
        ),
        language=context.options.get("language"),
        segments=segments,
        # ASR model/version 进入 lineage（默认 faster-whisper，可被 options 覆盖）。
        asr_model=str(context.options.get("asr_model") or "faster-whisper"),
        asr_model_version=str(context.options.get("asr_model_version") or "1.0"),
        parent_artifact_ids=(parent_artifact_id,) if parent_artifact_id else (
            (context.artifacts.media.artifact_id,) if context.artifacts.media else ()
        ),
    )
    context.artifacts.transcript = TranscriptArtifact(
        **{**transcript.__dict__, "artifact_id": artifact_id_of(transcript)}
    )


class SpeakerDiarizationStage:
    name = "diarization"
    required_inputs = ("transcript",)
    output_types = ("transcript",)

    def __init__(self, diarizer) -> None:
        self._diarizer = diarizer

    def execute(self, context: PipelineContext) -> PipelineContext:
        previous_transcript = context.artifacts.transcript.artifact_id if context.artifacts.transcript else None
        context.state["segments"] = self._diarizer.annotate(
            str(context.runtime.audio_path or "") or None, context.state["segments"]
        )
        status = getattr(self._diarizer, "last_status", "UNKNOWN")
        context.state["diarization_status"] = status
        if status in {"UNAVAILABLE", "FAILED", "DEGRADED"}:
            context.state.setdefault("quality_warnings", []).append(f"DIARIZATION_{status}")
        _register_transcript_artifact(
            context, producer_stage="diarization", parent_artifact_id=previous_transcript
        )
        return _stage_result(context, "transcript")


class TranscriptPostprocessStage:
    name = "transcript_postprocess"
    required_inputs = ("transcript",)
    output_types = ("transcript",)

    def __init__(self, postprocessor: TranscriptPostprocessor) -> None:
        self._postprocessor = postprocessor

    def execute(self, context: PipelineContext) -> PipelineContext:
        previous_transcript = context.artifacts.transcript.artifact_id if context.artifacts.transcript else None
        context.state["segments"] = self._postprocessor.process(context.state["segments"])
        context.state["transcript"] = " ".join(segment.text for segment in context.state["segments"])
        # P0 C-02：后处理后的 transcript 升级为新的权威 TranscriptArtifact。
        _register_transcript_artifact(
            context, producer_stage="transcript_postprocess", parent_artifact_id=previous_transcript
        )
        return _stage_result(context, "transcript")


class OCRStage:
    name = "ocr"
    required_inputs = ("media",)
    output_types = ("ocr",)
    optional_output_types = ("ocr",)

    def __init__(self, engine) -> None:
        self._engine = engine

    def execute(self, context: PipelineContext) -> PipelineContext:
        supplied = context.options.get("ocr_evidence")
        evidence = list(supplied) if supplied is not None else []
        if supplied is None:
            for frame in context.state.get("frames", []):
                result = self._engine.recognize(str(frame["image_path"]))
                blocks = list(result.get("blocks") or [])
                item = {
                    **frame,
                    "ocr_text": str(result.get("text") or ""),
                    "ocr_evidence": {"blocks": blocks},
                    "ocr_engine": result.get("engine"),
                    "ocr_engine_version": result.get("engine_version"),
                }
                context.state.setdefault("frame_insights", []).append(item)
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
        context.state["ocr_evidence"] = evidence
        ocr_artifacts = []
        for index, item in enumerate(evidence):
            frame_id = str(item.get("frame_id") or "")
            parent = next((frame.artifact_id for frame in context.artifacts.frames if frame.frame_id == frame_id), "")
            ocr = OCRArtifact(
                artifact_id="ocr-pending",
                artifact_type="ocr",
                frame_artifact_id=parent,
                text=str(item.get("evidence_text") or item.get("text") or ""),
                blocks=[dict(item)],
                engine=str(item.get("ocr_engine") or "fixture"),
                engine_version=str(item.get("ocr_engine_version") or "1"),
                parent_artifact_ids=(parent,) if parent else (),
            )
            ocr_artifacts.append(OCRArtifact(**{**ocr.__dict__, "artifact_id": artifact_id_of(ocr)}))
        context.artifacts.ocr = ocr_artifacts
        return _stage_result(context, "ocr")


class VisionStage:
    name = "vision"
    required_inputs = ("media",)
    output_types = ("vision",)
    optional_output_types = ("vision",)

    def __init__(self, analyzer) -> None:
        self._analyzer = analyzer

    def execute(self, context: PipelineContext) -> PipelineContext:
        insights = list(context.state.get("frame_insights") or [])
        if context.options.get("frame_insights") is not None:
            insights.extend(context.options["frame_insights"])
        elif context.state.get("frames"):
            for frame in context.state["frames"]:
                result = self._analyzer.analyze(str(frame["image_path"]), context.state["transcript"])
                insights.append({**frame, **result, "model": getattr(self._analyzer, "_model", None)})
        context.state["frame_insights"] = insights
        vision_artifacts = []
        for index, item in enumerate(insights):
            frame_id = str(item.get("frame_id") or "")
            parent = next((frame.artifact_id for frame in context.artifacts.frames if frame.frame_id == frame_id), "")
            vision = VisionArtifact(
                artifact_id="vision-pending",
                artifact_type="vision",
                frame_artifact_id=parent,
                label=str(item.get("label") or item.get("description") or ""),
                payload=dict(item),
                model_name=str(item.get("model") or "fixture"),
                model_version=str(item.get("model_version") or "1"),
                parent_artifact_ids=(parent,) if parent else (),
            )
            vision_artifacts.append(VisionArtifact(**{**vision.__dict__, "artifact_id": artifact_id_of(vision)}))
        context.artifacts.vision = vision_artifacts
        return _stage_result(context, "vision")


class MultimodalContextStage:
    name = "multimodal_context"
    required_inputs = ("transcript",)
    output_types = ()

    def __init__(self, builder) -> None:
        self._builder = builder

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = {
            "segments": [
                {"text": item.text, "start_ms": int(item.start_seconds * 1000), "end_ms": int(item.end_seconds * 1000)}
                for item in context.state["segments"]
            ]
        }
        context.state["multimodal_context"] = self._builder.build(transcript, context.state.get("frame_insights") or [])
        return _stage_result(context)


class TemporalWindowStage:
    name = "temporal_window"
    required_inputs = ("transcript",)
    output_types = ()

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
                for item in context.state["segments"]
            ]
        }
        context.state["temporal_windows"] = self._builder.build(transcript, context.state.get("frame_insights") or [])
        return _stage_result(context)


class ChapterStage:
    name = "chapter"
    required_inputs = ("transcript",)
    output_types = ()

    def __init__(self, segmenter: ChapterSegmenter) -> None:
        self._segmenter = segmenter

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.state["chapters"] = self._segmenter.segment(context.state["segments"])
        return _stage_result(context)


class SemanticSegmentationStage:
    """Authoritative semantic boundary stage; chapters remain compatibility output."""

    name = "semantic_segmentation"
    required_inputs = ("transcript",)
    output_types = ("semantic_segments",)

    def __init__(self, segmenter: SemanticSegmenter | None = None, model_gateway=None, repository=None) -> None:
        self._segmenter = segmenter or SemanticSegmenter(model_gateway)
        self._repository = repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = context.artifacts.transcript
        if transcript is None:
            raise ValueError("semantic segmentation requires transcript artifact")
        try:
            result = self._segmenter.segment(
                transcript,
                offline_fixture=bool(
                    context.options.get("offline_fixture")
                    or "transcript" in context.options
                    or "segments" in context.options
                ),
            )
        except Exception:
            # Keep the metric fail-closed while preserving the stage error.
            context.runtime.metrics["segmentation_failure_rate"] = 1.0
            raise
        context.state["semantic_segments"] = list(result.segments)
        context.runtime.metrics.update(result.metrics)
        durations = sorted(max(0, item.end_ms - item.start_ms) for item in result.segments)
        count = len(durations)
        def _percentile(percent: float) -> float:
            if not durations:
                return 0.0
            index = max(0, min(count - 1, int((count - 1) * percent)))
            return float(durations[index])
        context.runtime.metrics.update({
            "semantic_segments_per_video": float(count),
            "semantic_segment_duration_p50": _percentile(0.50),
            "semantic_segment_duration_p95": _percentile(0.95),
            "segmentation_repair_rate": float(result.metrics.get("repair_count", 0.0)) / max(1.0, float(count)),
            "segmentation_failure_rate": float(result.metrics.get("failure_count", 0.0)) / max(1.0, float(count)),
        })
        context.artifacts.semantic_segments = result.artifact
        if self._repository is not None:
            video = context.state.get("video")
            video_id = getattr(video, "video_id", None)
            if not video_id:
                raise ValueError(
                    "semantic segment persistence requires the authoritative current context.state.video.video_id"
                )
            self._repository.save(result.artifact, video_id=video_id)
        return _stage_result(context, "semantic_segments")


class SemanticContextStage:
    name = "semantic_context"
    required_inputs = ("transcript", "semantic_segments")
    output_types = ()

    def __init__(self, builder: SemanticContextBuilder | None = None, padding_ms: int = 4000) -> None:
        self._builder = builder or SemanticContextBuilder(padding_ms=padding_ms)

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = context.artifacts.transcript
        semantic_artifact = context.artifacts.semantic_segments
        if transcript is None or semantic_artifact is None:
            raise ValueError("semantic context requires transcript and semantic segments")
        contexts = [
            self._builder.build(
                segment,
                transcript,
                context.artifacts.frames,
                context.artifacts.ocr,
                context.artifacts.vision,
                context.state.get("temporal_windows") or (),
            )
            for segment in context.state.get("semantic_segments") or ()
        ]
        context.state["semantic_contexts"] = contexts
        return _stage_result(context)


class AtomicClaimExtractionStage:
    name = "atomic_claim_extraction"
    required_inputs = ("semantic_segments",)
    output_types = ()

    def __init__(self, extractor: AtomicClaimExtractor | None = None, model_gateway=None) -> None:
        self._extractor = extractor or AtomicClaimExtractor(model_gateway)

    def execute(self, context: PipelineContext) -> PipelineContext:
        drafts: list[ClaimOccurrenceDraft] = []
        fixture = context.options.get("claim_drafts")
        fixture_by_segment: dict[str, list[Any]] = {}
        for item in fixture or ():
            value = item if isinstance(item, dict) else item.model_dump(mode="json")
            fixture_by_segment.setdefault(str(value.get("semantic_segment_id") or ""), []).append(value)
        for semantic_context in context.state.get("semantic_contexts") or ():
            drafts.extend(
                self._extractor.extract(
                    semantic_context,
                    metadata=context.state.get("metadata") or {},
                    fixture_drafts=fixture_by_segment.get(semantic_context.semantic_segment_id)
                    if fixture is not None
                    else None,
                    offline_fixture=bool(
                        context.options.get("offline_fixture")
                        or "transcript" in context.options
                        or "segments" in context.options
                    ),
                )
            )
        context.state["claim_drafts"] = drafts
        context.runtime.metrics["claim_count"] = float(len(drafts))
        context.runtime.metrics["zero_claim_context_count"] = float(
            sum(not any(item.semantic_segment_id == c.semantic_segment_id for item in drafts)
                for c in context.state.get("semantic_contexts") or ())
        )
        segment_count = len(context.state.get("semantic_segments") or ())
        zero_claim_count = sum(
            not any(item.semantic_segment_id == segment.semantic_segment_id for item in drafts)
            for segment in context.state.get("semantic_segments") or ()
        )
        context.runtime.metrics["claims_per_semantic_segment"] = len(drafts) / max(1.0, float(segment_count))
        context.runtime.metrics["zero_claim_segment_ratio"] = zero_claim_count / max(1.0, float(segment_count))
        return _stage_result(context)


class EvidenceGroundingStage:
    name = "evidence_grounding"
    required_inputs = ("transcript", "semantic_segments")
    output_types = ("evidence",)

    def __init__(self, grounder: ClaimDraftGrounder | None = None) -> None:
        self._grounder = grounder or ClaimDraftGrounder()

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = context.artifacts.transcript
        if transcript is None:
            raise ValueError("evidence grounding requires transcript")
        by_id = {item.semantic_segment_id: item for item in context.state.get("semantic_segments") or ()}
        drafts = list(context.state.get("claim_drafts") or ())
        try:
            grounded = [
                self._grounder.ground(draft, transcript, by_id[draft.semantic_segment_id])
                for draft in drafts
                if draft.semantic_segment_id in by_id
            ]
        except ValueError as exc:
            context.runtime.metrics["claim_grounding_reject_rate"] = 1.0
            if "temporal expression" in str(exc):
                context.runtime.metrics["temporal_expression_grounding_reject_rate"] = 1.0
            raise
        rejected = len(drafts) - len(grounded)
        context.runtime.metrics["claim_grounding_reject_rate"] = rejected / max(1.0, float(len(drafts)))
        context.runtime.metrics["temporal_expression_grounding_reject_rate"] = 0.0
        evidence_items = []
        for item in grounded:
            evidence_items.extend(item.evidences)
        unique = {item.evidence_id: item for item in evidence_items}
        evidence_artifact = EvidenceArtifact(
            artifact_id="evidence-pending",
            artifact_type="evidence",
            producer_stage=self.name,
            transcript_artifact_id=transcript.artifact_id,
            evidences=list(unique.values()),
            source_artifact_ids=(transcript.artifact_id,),
            # Semantic segmentation is the authoritative boundary producer;
            # transcript remains an explicit compatibility reference.
            parent_artifact_ids=tuple(
                item.artifact_id
                for item in (context.artifacts.semantic_segments, transcript)
                if item is not None
            ),
        )
        context.artifacts.evidence = EvidenceArtifact(
            **{**evidence_artifact.__dict__, "artifact_id": artifact_id_of(evidence_artifact)}
        )
        context.state["grounded_occurrences"] = grounded
        context.state.evidence = list(unique.values())
        context.runtime.metrics["grounding_reject_count"] = 0.0
        return _stage_result(context, "evidence")


class TemporalNormalizationStage:
    name = "temporal_normalization"
    required_inputs = ("evidence",)
    output_types = ()

    def __init__(
        self,
        normalizer: TemporalNormalizer | None = None,
        normalization_version: str = "temporal-normalization.final.v1",
    ) -> None:
        self._normalizer = normalizer or TemporalNormalizer(normalization_version=normalization_version)

    def execute(self, context: PipelineContext) -> PipelineContext:
        bindings_by_draft: dict[int, list[Any]] = {}
        anchor = context.options.get("temporal_anchor") or context.options.get("as_of")
        if isinstance(anchor, str):
            anchor = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
        as_of = context.options.get("as_of")
        if isinstance(as_of, str):
            as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        evidence_by_segment = {
            int(item.locator["segment_index"]): item.evidence_id
            for item in context.state.get("evidence") or ()
            if item.locator.get("segment_index") is not None
        }
        for draft_index, draft in enumerate(context.state.get("claim_drafts") or ()):
            draft_bindings = []
            for expression in draft.temporal_expressions:
                try:
                    role = TemporalRole(expression.role)
                except ValueError as exc:
                    raise ValueError(f"unknown temporal role: {expression.role}") from exc
                scope_hint = TemporalScope(expression.scope_hint) if expression.scope_hint else None
                expression_anchor = expression.anchor
                if isinstance(expression_anchor, str):
                    symbolic_anchor = expression_anchor.strip().upper()
                    if symbolic_anchor == "SOURCE_PUBLISH_TIME":
                        expression_anchor = (
                            _resolved_datetime(context.state.video.published_at, "video.published_at")
                            if context.state.video
                            else None
                        )
                        if expression_anchor is None:
                            expression_anchor = _resolved_datetime(
                                (context.state.get("metadata") or {}).get("published_at"),
                                "metadata.published_at",
                            )
                        if expression_anchor is None:
                            raise ValueError("SOURCE_PUBLISH_TIME anchor is unavailable")
                    else:
                        try:
                            expression_anchor = datetime.fromisoformat(expression_anchor.replace("Z", "+00:00"))
                        except ValueError:
                            try:
                                expression_anchor = date.fromisoformat(expression_anchor)
                            except ValueError as exc:
                                # Do not silently convert an unknown symbolic
                                # anchor into the task-level as_of timestamp.
                                raise ValueError(
                                    f"unknown temporal anchor: {expression.anchor}"
                                ) from exc
                draft_text = " ".join((draft.conclusion or "", draft.condition_text or ""))
                normalized_words = "".join(draft_text.split()).casefold()
                assertion_status = None
                if any(token in normalized_words for token in ("下修", "上修", "修订", "修正", "改到", "revised")):
                    assertion_status = TemporalAssertionStatus.REVISED
                elif any(token in normalized_words for token in ("计划", "拟", "planned")):
                    assertion_status = TemporalAssertionStatus.PLANNED
                elif any(token in normalized_words for token in ("预计", "预期", "expected", "estimate")):
                    assertion_status = TemporalAssertionStatus.EXPECTED
                metric_nature = None
                expression_text = expression.raw_expression.upper()
                if draft.claim_type == "FINANCIAL_METRIC":
                    if any(marker in expression_text for marker in ("YTD", "TTM", "LTM", "NTM")):
                        metric_nature = None  # let the normalizer's exact labels win
                    elif any(
                        marker in expression_text
                        for marker in ("期末", "余额", "截至", "END", "ENDING", "AS OF")
                    ):
                        metric_nature = MetricTemporalNature.INSTANT
                    elif (
                        scope_hint is TemporalScope.INTERVAL
                        or expression.scope_hint == "INTERVAL"
                        or any(marker in expression_text for marker in ("Q", "季度", "FY", "年", "月"))
                    ):
                        metric_nature = MetricTemporalNature.DURATION
                elif draft.claim_type in {"PRICE", "VALUATION"} and scope_hint in {
                    None, TemporalScope.POINT
                }:
                    metric_nature = MetricTemporalNature.SNAPSHOT
                draft_bindings.append(
                    self._normalizer.normalize(
                        expression.raw_expression,
                        role=role,
                        anchor=expression_anchor or anchor,
                        as_of=as_of,
                        subject_key=draft.subject_key,
                        scope_hint=scope_hint,
                        evidence_refs=[
                            evidence_by_segment[index]
                            for index in expression.evidence_segment_indices
                            if index in evidence_by_segment
                        ],
                        assertion_status=assertion_status,
                        metric_temporal_nature=metric_nature,
                    )
                )
            bindings_by_draft[draft_index] = draft_bindings
        context.state["temporal_bindings"] = [item for values in bindings_by_draft.values() for item in values]
        context.state["temporal_bindings_by_draft"] = bindings_by_draft
        context.runtime.metrics["temporal_normalized_count"] = float(
            sum(item.normalization_status == "NORMALIZED" for item in context.state["temporal_bindings"])
        )
        context.runtime.metrics["temporal_partial_count"] = float(
            sum(item.normalization_status == "PARTIAL" for item in context.state["temporal_bindings"])
        )
        context.runtime.metrics["temporal_unresolved_count"] = float(
            sum(item.normalization_status == "UNRESOLVED" for item in context.state["temporal_bindings"])
        )
        bindings = list(context.state["temporal_bindings"])
        binding_count = len(bindings)
        context.runtime.metrics.update({
            "temporal_binding_count": float(binding_count),
            "temporal_normalization_success_rate": sum(
                item.normalization_status == "NORMALIZED" for item in bindings
            ) / max(1.0, float(binding_count)),
            "temporal_normalization_partial_rate": sum(
                item.normalization_status == "PARTIAL" for item in bindings
            ) / max(1.0, float(binding_count)),
            "temporal_normalization_unresolved_rate": sum(
                item.normalization_status == "UNRESOLVED" for item in bindings
            ) / max(1.0, float(binding_count)),
            "temporal_partial_rate": sum(
                item.normalization_status == "PARTIAL" for item in bindings
            ) / max(1.0, float(binding_count)),
            "temporal_unresolved_rate": sum(
                item.normalization_status == "UNRESOLVED" for item in bindings
            ) / max(1.0, float(binding_count)),
            "temporal_role_distribution": {
                str(getattr(item.role, "value", item.role)): sum(
                    getattr(other.role, "value", other.role) == getattr(item.role, "value", item.role)
                    for other in bindings
                )
                for item in sorted(bindings, key=lambda value: str(getattr(value.role, "value", value.role)))
            },
        })
        unresolved = [item.expression_key for item in bindings if item.normalization_status == "UNRESOLVED"]
        context.runtime.metrics["unresolved_expression_collision_rate"] = (
            len(unresolved) - len(set(unresolved))
        ) / max(1.0, float(len(unresolved)))
        forecast_drafts = [item for item in context.state.get("claim_drafts") or () if item.claim_type == "FORECAST"]
        forecast_with_target = {
            index for index, values in bindings_by_draft.items()
            if any(getattr(binding, "role", None) is TemporalRole.FORECAST_TARGET for binding in values)
        }
        context.runtime.metrics["forecast_target_missing_rate"] = (
            sum(index not in forecast_with_target for index, item in enumerate(context.state.get("claim_drafts") or ())
                if item.claim_type == "FORECAST")
            / max(1.0, float(len(forecast_drafts)))
        )
        fiscal = [item for item in bindings if getattr(item.calendar_type, "value", item.calendar_type) == "FISCAL"]
        context.runtime.metrics["fiscal_period_unresolved_rate"] = sum(
            item.normalization_status in {"PARTIAL", "UNRESOLVED"} for item in fiscal
        ) / max(1.0, float(len(fiscal)))
        market = [
            item
            for item in bindings
            if item.market_session
            or getattr(item.calendar_type, "value", item.calendar_type) == "EXCHANGE"
        ]
        context.runtime.metrics["market_session_unresolved_rate"] = sum(
            not item.market_session for item in market
        ) / max(1.0, float(len(market)))
        metric_drafts = [
            item
            for item in context.state.get("claim_drafts") or ()
            if item.claim_type == "FINANCIAL_METRIC"
        ]
        metric_binding_items = [
            binding for index, values in bindings_by_draft.items()
            if index < len(context.state.get("claim_drafts") or ())
            and context.state["claim_drafts"][index].claim_type == "FINANCIAL_METRIC"
            for binding in values
        ]
        context.runtime.metrics["metric_temporal_nature_unknown_rate"] = sum(
            getattr(item.metric_temporal_nature, "value", item.metric_temporal_nature) in {None, "UNKNOWN"}
            for item in metric_binding_items
        ) / max(1.0, float(len(metric_binding_items) or len(metric_drafts)))
        planned = sum(
            getattr(item.assertion_status, "value", item.assertion_status) == "PLANNED" for item in bindings
        )
        actual = sum(
            getattr(item.assertion_status, "value", item.assertion_status) == "ACTUAL" for item in bindings
        )
        context.runtime.metrics["planned_vs_actual_ratio"] = planned / max(1.0, float(actual))
        return _stage_result(context)


class ClaimCanonicalizationStage:
    name = "claim_canonicalization"
    required_inputs = ("evidence",)
    output_types = ("claims",)

    def __init__(self, canonicalizer: ClaimCanonicalizer | None = None) -> None:
        self._canonicalizer = canonicalizer or ClaimCanonicalizer()

    def execute(self, context: PipelineContext) -> PipelineContext:
        config = dict(context.options.get("pipeline_config") or {})
        configured_normalization_version = config.get("temporal_normalization_version")
        claims = [
            self._canonicalizer.canonicalize(
                draft,
                temporal_bindings=(context.state.get("temporal_bindings_by_draft") or {}).get(index, []),
                evidence_refs=[],
                normalization_version=str(configured_normalization_version)
                if configured_normalization_version else None,
            )
            for index, draft in enumerate(context.state.get("claim_drafts") or ())
        ]
        context.state.claims = claims
        evidence_artifact = context.artifacts.evidence
        claim_artifact = ClaimArtifact(
            artifact_id="claims-pending",
            artifact_type="claims",
            producer_stage=self.name,
            evidence_artifact_id=evidence_artifact.artifact_id if evidence_artifact else "",
            claims=[claim.claim_id for claim in claims],
            parent_artifact_ids=(evidence_artifact.artifact_id,) if evidence_artifact else (),
        )
        context.artifacts.claims = ClaimArtifact(
            **{**claim_artifact.__dict__, "artifact_id": artifact_id_of(claim_artifact)}
        )
        return _stage_result(context, "claims")


def _stage_timestamp(context: PipelineContext) -> datetime:
    # Replay's fallback claim timestamp may be an old wall-clock value from a
    # legacy fixture.  Lifecycle projection must use the same deterministic
    # transcript boundary as the source run unless an explicit replay clock
    # was supplied.
    replay_without_explicit_clock = context.options.get("replay_lifecycle_timestamp") == "derive_transcript_boundary"
    value = None if replay_without_explicit_clock else (
        context.options.get("snapshot_commit_candidate")
        or context.options.get("available_from")
        or context.options.get("as_of")
    )
    if value is None and not (
        context.options.get("offline_fixture") or "transcript" in context.options or "segments" in context.options
    ):
        return datetime.now(UTC)
    if value is None:
        # A missing fixture clock is represented by the transcript boundary,
        # keeping this stage deterministic and avoiding a wall-clock identity.
        end_ms = max((item.end_ms for item in context.artifacts.transcript.segments), default=0)
        return datetime.fromtimestamp(end_ms / 1000, UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class ClaimOccurrencePersistenceStage:
    name = "claim_occurrence_persistence"
    required_inputs = ("semantic_segments", "evidence", "claims")
    output_types = ("occurrences", "claims")

    def __init__(self, repository=None) -> None:
        self._repository = repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = context.artifacts.transcript
        semantic_artifact = context.artifacts.semantic_segments
        evidence_artifact = context.artifacts.evidence
        if transcript is None or semantic_artifact is None:
            raise ValueError("occurrence persistence requires transcript and semantic segments")
        timestamp = _stage_timestamp(context)
        def _time(name: str, fallback: datetime | None = None) -> datetime | None:
            value = context.options.get(name, fallback)
            return _resolved_datetime(value, name)
        fixture_clock = bool(
            context.options.get("offline_fixture")
            or "transcript" in context.options
            or "segments" in context.options
        )
        # Production clocks describe actual processing events.  ``as_of`` is
        # a business/query clock and must never become a Unix-epoch-like
        # ingestion timestamp.  Fixture runs are the sole deterministic
        # exception, where the transcript boundary is an explicit clock.
        ingested = _time("ingested_at") or (timestamp if fixture_clock else datetime.now(UTC))
        extracted = _time("extraction_completed_at") or (timestamp if fixture_clock else datetime.now(UTC))
        source_available = _time("source_available_at")
        if source_available is None and not fixture_clock:
            source_available = ingested
        binding_available = [
            item.reference_available_at
            for item in context.state.get("temporal_bindings") or ()
            if getattr(item, "reference_available_at", None) is not None
        ]
        reference_available = max(
            [item for item in [_time("reference_available_at"), *binding_available] if item],
            default=None,
        )
        external_available = _time("external_available_at")
        candidate = choose_snapshot_commit_candidate(
            ingested_at=ingested,
            extraction_completed_at=extracted,
            source_available_at=source_available,
            reference_available_at=reference_available,
            external_available_at=external_available,
            candidate=_time("snapshot_commit_candidate"),
        )
        context.options["snapshot_commit_candidate"] = candidate
        claims = list(context.state.get("claims") or ())
        drafts = list(context.state.get("claim_drafts") or ())
        grounded = list(context.state.get("grounded_occurrences") or ())
        occurrences = []
        for draft_index, (claim, draft, relation) in enumerate(zip(claims, drafts, grounded)):
            refs = sorted(set(relation.primary_evidence_refs))
            source_published_at = _time("source_published_at")
            if source_published_at is None and context.state.video:
                source_published_at = _resolved_datetime(
                    context.state.video.published_at,
                    "video.published_at",
                )
            if source_published_at is None:
                source_published_at = _resolved_datetime(
                    (context.state.get("metadata") or {}).get("published_at"),
                    "metadata.published_at",
                )
            times = OccurrenceTimes(
                asserted_at=_time("asserted_at"),
                source_published_at=source_published_at,
                source_available_at=source_available,
                source_availability_quality=str(context.options.get("source_availability_quality", "UNKNOWN")),
                ingested_at=ingested,
                extraction_completed_at=extracted,
                snapshot_committed_at=candidate,
                available_from=candidate,
            )
            occurrences.append(
                ClaimOccurrence(
                    claim_id=claim.claim_id,
                    source_artifact_id=(
                        context.artifacts.source.artifact_id
                        if context.artifacts.source
                        else transcript.artifact_id
                    ),
                    transcript_artifact_id=transcript.artifact_id,
                    semantic_segment_id=draft.semantic_segment_id,
                    evidence_refs=refs,
                    condition_evidence_refs=list(relation.condition_evidence_refs),
                    invalidation_evidence_refs=list(relation.invalidation_evidence_refs),
                    temporal_evidence_refs=list(relation.temporal_evidence_refs),
                    times=times,
                    raw_temporal_expressions=[
                        {
                            "role": expression.role,
                            "raw_expression": expression.raw_expression,
                            "scope_hint": expression.scope_hint,
                            "anchor": expression.anchor,
                            "confidence": expression.confidence,
                            "evidence_segment_indices": list(expression.evidence_segment_indices),
                            "grounded_evidence_refs": list(next(
                                (
                                    binding.source_evidence_refs
                                    for binding in context.state.get("temporal_bindings_by_draft", {}).get(
                                        draft_index, []
                                    )
                                    if getattr(binding, "raw_expression", None) == expression.raw_expression
                                ),
                                [],
                            )),
                        }
                        for expression in draft.temporal_expressions
                    ],
                    provenance={
                        "model_id": draft.extraction_model_id,
                        "prompt_version": draft.extraction_prompt_version,
                    },
                )
            )
        occurrence_artifact = ClaimOccurrenceArtifact(
            artifact_id="occurrences-pending",
            artifact_type="occurrences",
            producer_stage=self.name,
            semantic_segment_artifact_id=semantic_artifact.artifact_id,
            evidence_artifact_id=evidence_artifact.artifact_id if evidence_artifact else "",
            occurrence_ids=[item.occurrence_id for item in occurrences],
            parent_artifact_ids=tuple(
                item.artifact_id for item in (semantic_artifact, evidence_artifact) if item is not None
            ),
        )
        context.artifacts.occurrences = ClaimOccurrenceArtifact(
            **{**occurrence_artifact.__dict__, "artifact_id": artifact_id_of(occurrence_artifact)}
        )
        # The final canonical claim artifact is downstream of the occurrence
        # artifact.  Keep the evidence field for compatibility, but make the
        # parent edge authoritative for the Evidence -> Occurrence -> Claim
        # lineage required by the final design.
        if context.artifacts.claims is not None:
            claim_artifact = ClaimArtifact(
                artifact_id="claims-final-pending",
                artifact_type="claims",
                producer_stage=self.name,
                evidence_artifact_id=evidence_artifact.artifact_id if evidence_artifact else "",
                claims=[item.claim_id for item in claims],
                parent_artifact_ids=(context.artifacts.occurrences.artifact_id,),
            )
            context.artifacts.claims = ClaimArtifact(
                **{**claim_artifact.__dict__, "artifact_id": artifact_id_of(claim_artifact)}
            )
        context.state["occurrences"] = occurrences
        # Durable occurrence rows are committed together with the snapshot at
        # the snapshot boundary; this stage only materializes immutable state.
        context.runtime.metrics["occurrence_count"] = float(len(occurrences))
        context.runtime.metrics["occurrences_per_claim"] = len(occurrences) / max(1.0, float(len(claims)))
        context.runtime.metrics["dependency_availability_delay_ms"] = max(
            0.0,
            (candidate - min(ingested, extracted)).total_seconds() * 1000.0,
        )
        return _stage_result(context, "occurrences", "claims")


class LifecycleProjectionStage:
    name = "lifecycle_projection"
    # Lifecycle is a projection of both immutable occurrence rows and the
    # verification decision available at this point in the graph.  Requiring
    # the verification artifact prevents publishing a historical lifecycle
    # closure that silently omits its verification lineage.
    required_inputs = ("occurrences", "verification")
    output_types = ("lifecycle", "knowledge")

    def __init__(self, repository=None) -> None:
        self._repository = repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        fixture_clock = bool(
            context.options.get("offline_fixture")
            or "transcript" in context.options
            or "segments" in context.options
        )
        # A fixture without an explicit business clock must be replayable.  The
        # download compatibility adapter may expose a wall-clock source
        # availability timestamp for PUBLIC_STRICT search, but that timestamp
        # is not a lifecycle business clock.  Use the immutable transcript
        # boundary for both the initial run and replay.
        if fixture_clock and not any(
            context.options.get(key)
            for key in ("as_of", "available_from", "replay_lifecycle_timestamp")
        ):
            end_ms = max(
                (item.end_ms for item in context.artifacts.transcript.segments),
                default=0,
            )
            timestamp = datetime.fromtimestamp(end_ms / 1000, UTC)
        else:
            timestamp = _stage_timestamp(context)
        claims = list(context.state.get("claims") or ())
        occurrences = list(context.state.get("occurrences") or ())
        events = []
        for target_type, target_id in [
            *(('CLAIM', item.claim_id) for item in claims),
            *(('OCCURRENCE', item.occurrence_id) for item in occurrences),
        ]:
            events.append(
                KnowledgeLifecycleEvent(
                    target_type=target_type,
                    target_id=target_id,
                    to_status="ACTIVE",
                    effective_at=timestamp,
                    recorded_at=timestamp,
                    reason_code="INITIAL_EXTRACTION",
                    policy_version="lifecycle.v1",
                )
            )
        occurrence_artifact = context.artifacts.occurrences
        lifecycle_artifact = LifecycleArtifact(
            artifact_id="lifecycle-pending",
            artifact_type="lifecycle",
            producer_stage=self.name,
            claim_lifecycle_event_ids=[
                item.lifecycle_event_id for item in events if item.target_type.value == "CLAIM"
            ],
            occurrence_lifecycle_event_ids=[
                item.lifecycle_event_id for item in events if item.target_type.value == "OCCURRENCE"
            ],
            lifecycle_business_as_of=timestamp,
            lifecycle_knowledge_as_of=timestamp,
            policy_version="lifecycle.v1",
            parent_artifact_ids=tuple(
                item.artifact_id
                for item in (occurrence_artifact, context.artifacts.verification)
                if item is not None
            ),
        )
        context.artifacts.lifecycle = LifecycleArtifact(
            **{**lifecycle_artifact.__dict__, "artifact_id": artifact_id_of(lifecycle_artifact)}
        )
        context.state["lifecycle_events"] = events
        # Rebuild the final knowledge projection after lifecycle assignment so
        # the authoritative chain is Verification -> Lifecycle -> Knowledge.
        # The verification id remains as an explicit compatibility field.
        lifecycle_id = context.artifacts.lifecycle.artifact_id
        for unit in context.state.get("knowledge") or ():
            attributes = dict(unit.attributes or {})
            attributes["lifecycle_status"] = "ACTIVE"
            attributes["lifecycle_artifact_id"] = lifecycle_id
            unit.attributes = attributes
        if context.artifacts.knowledge is not None:
            knowledge_artifact = KnowledgeArtifact(
                artifact_id="knowledge-final-pending",
                artifact_type="knowledge",
                producer_stage="knowledge_projection",
                verification_artifact_id=(
                    context.artifacts.verification.artifact_id
                    if context.artifacts.verification else ""
                ),
                knowledge_units=[unit.knowledge_uid for unit in context.state.get("knowledge") or ()],
                parent_artifact_ids=(lifecycle_id,),
            )
            context.artifacts.knowledge = KnowledgeArtifact(
                **{**knowledge_artifact.__dict__, "artifact_id": artifact_id_of(knowledge_artifact)}
            )
        # Durable event inserts are deferred to the snapshot commit boundary.
        context.runtime.metrics["lifecycle_event_count"] = float(len(events))
        context.runtime.metrics["lifecycle_transition_count"] = float(len(events))
        correction_count = float(sum(
            str(item.reason_code).upper() in {"CORRECTION", "CORRECTED", "REVISED"}
            for item in events
        ))
        context.runtime.metrics["correction_count"] = correction_count
        context.runtime.metrics["lifecycle_correction_count"] = correction_count
        return _stage_result(context, "lifecycle", "knowledge")


class KnowledgeExtractionStage:
    name = "knowledge"
    required_inputs = ("transcript",)
    output_types = ("evidence", "claims", "verification", "knowledge")
    optional_output_types = ("evidence", "claims")

    def __init__(
        self,
        model_client: ContentModelClient | None = None,
        fixture_extractor: KnowledgeExtractor | None = None,
        external_verifier: ExternalFactVerifier | None = None,
        authoritative_only: bool = False,
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
        self._authoritative_only = authoritative_only

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
        for chapter in context.state["chapters"]:
            segments = [
                segment
                for segment in context.state["segments"]
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
                "ocr_blocks": list(context.state.get("ocr_evidence") or []),
                "frame_refs": list(context.state.get("frame_insights") or []),
            }
            payload.append(
                {
                    "chapter_index": chapter.chapter_index,
                    "title": chapter.title,
                    "chapter_type": chapter.chapter_type,
                    "primary_domain": "GENERAL",
                    "windows": [
                        item
                        for item in context.state.get("temporal_windows", [])
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
            # 收尾文档 §63：外部验证状态随 attributes 落库，供 Evidence/Signal 链路传递。
            attributes["external_verification_status"] = record.get("external_verification_status") or "NOT_RUN"
            # Entity resolution (including OCR/LLM correction provenance) is
            # already produced before persistence. Keep it immutable here so
            # repositories never repeat heuristic resolution on writes.
            entities = list(record.get("entities") or [])
            if not entities and record.get("ticker"):
                entities.append(
                    {
                        "entity_name": record.get("subject_name") or record["ticker"],
                        "entity_key": record["ticker"],
                        "ticker": record["ticker"],
                        "entity_type": "EQUITY",
                        "resolution_source": "knowledge_subject",
                    }
                )
            attributes["entities"] = entities
            if record.get("entity_resolution"):
                attributes["entity_resolution"] = dict(record["entity_resolution"])
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
            context.state["video"].video_id, context.state["chapters"], timestamp
        ):
            chapter = next((item for item in context.state["chapters"] if item.chapter_id == unit.chapter_id), None)
            start_ms = int((chapter.start_seconds if chapter else 0) * 1000)
            end_ms = int((chapter.end_seconds if chapter else 0) * 1000)
            records.append(
                {
                    "knowledge_uid": unit.knowledge_uid,
                    "statement": unit.statement,
                    "claim_type": (
                        "FINANCIAL_METRIC"
                        if any(
                            term in unit.statement
                            for term in ("营收", "收入", "利润", "业绩", "毛利率")
                        )
                        else "INDUSTRY_RELATION"
                    ),
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

    def _register_claim_chain(self, context: PipelineContext, records: list[dict]) -> None:
        """Materialize Evidence -> canonical Claim -> Verification artifacts."""
        transcript = context.artifacts.transcript
        evidence_items: list[EvidenceItem] = []
        if transcript:
            for segment in transcript.segments:
                raw = segment.text
                locator = {
                    "segment_index": segment.segment_index,
                    "start_ms": int(segment.start_seconds * 1000),
                    "end_ms": int(segment.end_seconds * 1000),
                }
                evidence_id = "evidence-" + hashlib.sha256(
                    canonical_json(
                        {"source": transcript.artifact_id, "locator": locator, "raw": raw}
                    ).encode()
                ).hexdigest()[:32]
                evidence_items.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        source_type="ASR",
                        evidence_text=raw,
                        start_ms=locator["start_ms"],
                        end_ms=locator["end_ms"],
                        confidence_score=segment.confidence,
                        locator=locator,
                        raw_text=raw,
                        normalized_text=" ".join(raw.split()),
                        source_artifact_id=transcript.artifact_id,
                    )
                )
        ocr_sources: dict[tuple[str, str], list[str]] = {}
        for artifact in context.artifacts.ocr:
            frame_id = next((frame.frame_id for frame in context.artifacts.frames
                             if frame.artifact_id == artifact.frame_artifact_id), "")
            ocr_sources.setdefault((frame_id, artifact.text), []).append(artifact.artifact_id)
        for item in context.state.ocr_evidence:
            raw = str(item.get("evidence_text") or item.get("text") or "")
            if not raw.strip():
                continue
            source_queue = ocr_sources.get((str(item.get("frame_id") or ""), raw), [])
            source = source_queue.pop(0) if source_queue else ""
            if not source:
                raise ValueError("OCR evidence requires a resolvable OCR artifact source")
            locator = {
                "frame_id": item.get("frame_id"),
                "timestamp_ms": item.get("timestamp_ms"),
                "bbox": item.get("bbox"),
            }
            evidence_id = "evidence-" + hashlib.sha256(
                canonical_json({"source": source, "locator": locator, "raw": raw}).encode()
            ).hexdigest()[:32]
            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type="OCR",
                    evidence_text=raw,
                    start_ms=int(item.get("timestamp_ms") or 0),
                    end_ms=int(item.get("timestamp_ms") or 0),
                    confidence_score=item.get("confidence_score"),
                    locator=locator,
                    raw_text=raw,
                    normalized_text=" ".join(raw.split()),
                    source_artifact_id=source,
                )
            )
        vision_sources: dict[tuple[str, str], list[str]] = {}
        for artifact in context.artifacts.vision:
            frame_id = next((frame.frame_id for frame in context.artifacts.frames
                             if frame.artifact_id == artifact.frame_artifact_id), "")
            vision_sources.setdefault((frame_id, artifact.label), []).append(artifact.artifact_id)
        for item in context.state.frame_insights:
            raw = str(item.get("description") or item.get("label") or "")
            if not raw.strip():
                continue
            source_queue = vision_sources.get((str(item.get("frame_id") or ""), raw), [])
            source = source_queue.pop(0) if source_queue else ""
            if not source:
                raise ValueError("VISION evidence requires a resolvable Vision artifact source")
            locator = {
                "frame_id": item.get("frame_id"),
                "timestamp_ms": item.get("timestamp_ms"),
            }
            evidence_id = "evidence-" + hashlib.sha256(
                canonical_json({"source": source, "locator": locator, "raw": raw}).encode()
            ).hexdigest()[:32]
            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_type="VISION",
                    evidence_text=raw,
                    start_ms=int(item.get("timestamp_ms") or 0),
                    end_ms=int(item.get("timestamp_ms") or 0),
                    confidence_score=item.get("confidence_score"),
                    locator=locator,
                    raw_text=raw,
                    normalized_text=" ".join(raw.split()),
                    source_artifact_id=source,
                )
            )
        if not evidence_items:
            raise ValueError("claim extraction requires source evidence")
        source_ids = tuple(
            filter(
                None,
                [transcript.artifact_id if transcript else ""]
                + [item.artifact_id for item in context.artifacts.ocr]
                + [item.artifact_id for item in context.artifacts.vision],
            )
        )
        evidence_parent_ids = tuple(
            dict.fromkeys(
                [
                    *source_ids,
                    context.artifacts.semantic_segments.artifact_id
                    if context.artifacts.semantic_segments else "",
                ]
            )
        )
        evidence = EvidenceArtifact(
            artifact_id="evidence-pending",
            artifact_type="evidence",
            producer_stage="knowledge",
            transcript_artifact_id=transcript.artifact_id if transcript else "",
            evidences=evidence_items,
            source_artifact_ids=source_ids,
            parent_artifact_ids=tuple(item for item in evidence_parent_ids if item),
        )
        context.artifacts.evidence = EvidenceArtifact(
            **{**evidence.__dict__, "artifact_id": artifact_id_of(evidence)}
        )
        context.state.evidence = evidence_items
        def support_status(value: object) -> str:
            normalized = str(value or "").upper()
            return {
                "SOURCE_SUPPORTED": "SUPPORTED",
                "SUPPORTED": "SUPPORTED",
                "PARTIAL": "PARTIALLY_SUPPORTED",
                "PARTIALLY_SUPPORTED": "PARTIALLY_SUPPORTED",
                "SOURCE_PARTIAL": "PARTIALLY_SUPPORTED",
                "UNSUPPORTED": "UNSUPPORTED",
                "SOURCE_UNSUPPORTED": "UNSUPPORTED",
                "AMBIGUOUS": "AMBIGUOUS",
            }.get(normalized, "UNSUPPORTED")

        def evidence_refs_for(record: dict[str, Any]) -> list[str]:
            candidates = list(record.get("evidence") or [])
            refs_for_record: list[str] = []
            for candidate in candidates:
                text = str(candidate.get("evidence_text") or candidate.get("text") or "")
                source_type = str(candidate.get("source_type") or "")
                match = next(
                    (
                        item
                        for item in evidence_items
                        if item.evidence_text == text
                        and (not source_type or item.source_type == source_type)
                    ),
                    None,
                )
                if match and match.evidence_id not in refs_for_record:
                    refs_for_record.append(match.evidence_id)
            if not refs_for_record:
                statement = str(record.get("statement") or "")
                match = next(
                    (item for item in evidence_items if item.evidence_text == statement),
                    None,
                )
                if match:
                    refs_for_record.append(match.evidence_id)
            if not refs_for_record and evidence_items:
                refs_for_record.append(evidence_items[0].evidence_id)
            return refs_for_record

        claims: list[FinancialClaim] = []
        for record in records:
            claim_type = str(record.get("claim_type") or "FINANCIAL_METRIC")
            if claim_type not in {
                "PRICE",
                "RETURN",
                "VALUATION",
                "FINANCIAL_METRIC",
                "CORPORATE_EVENT",
                "INDUSTRY_RELATION",
                "FORECAST",
                "OPINION",
                "INFERENCE",
            }:
                claim_type = "FINANCIAL_METRIC"
            claim_refs = evidence_refs_for(record)
            claims.append(
                FinancialClaim(
                    claim_type=claim_type,
                    subject_type="EQUITY" if record.get("ticker") else "CONTENT",
                    subject_id=str(record.get("subject_key") or record.get("knowledge_uid")),
                    predicate=str(record.get("predicate_key") or "statement"),
                    value=(
                        record.get("value")
                        if record.get("value") is not None
                        else str(record.get("statement") or "")
                    ),
                    unit=record.get("unit"),
                    currency=record.get("currency"),
                    fact_time=record.get("as_of_time"),
                    published_at=(
                        context.state.video.published_at if context.state.video else None
                    ),
                    evidence_refs=claim_refs,
                    source_support_status=support_status(record.get("support_status")),
                    source_confidence=float(record.get("support_score") or 0.75),
                    extractor_confidence=float(record.get("extraction_confidence") or 0.75),
                    extraction_model_id=str(
                        (record.get("attributes") or {}).get("model") or "fixture"
                    ),
                    extraction_prompt_version=str(
                        (record.get("attributes") or {}).get("prompt_version") or "fixture.v1"
                    ),
                )
            )
        context.state.claims = claims
        claim_artifact = ClaimArtifact(
            artifact_id="claims-pending",
            artifact_type="claims",
            producer_stage="knowledge",
            evidence_artifact_id=context.artifacts.evidence.artifact_id,
            claims=[claim.claim_id for claim in claims],
            parent_artifact_ids=(context.artifacts.evidence.artifact_id,),
        )
        context.artifacts.claims = ClaimArtifact(
            **{**claim_artifact.__dict__, "artifact_id": artifact_id_of(claim_artifact)}
        )
        results = [
            VerificationResult(
                claim_id=claim.claim_id,
                status=(
                    "VERIFICATION_PENDING" if claim.fact_category == "FACT" else "NOT_REQUIRED"
                ),
            )
            for claim in claims
        ]
        verification = VerificationArtifact(
            artifact_id="verification-pending",
            artifact_type="verification",
            producer_stage="verification",
            claim_artifact_id=context.artifacts.claims.artifact_id,
            results=results,
            parent_artifact_ids=(context.artifacts.claims.artifact_id,),
        )
        context.artifacts.verification = VerificationArtifact(
            **{**verification.__dict__, "artifact_id": artifact_id_of(verification)}
        )
        if context.artifacts.lifecycle is not None:
            lifecycle = LifecycleArtifact(
                artifact_id="lifecycle-chain-pending",
                artifact_type="lifecycle",
                producer_stage="lifecycle_projection",
                claim_lifecycle_event_ids=list(context.artifacts.lifecycle.claim_lifecycle_event_ids or ()),
                occurrence_lifecycle_event_ids=list(
                    context.artifacts.lifecycle.occurrence_lifecycle_event_ids or ()
                ),
                lifecycle_business_as_of=context.artifacts.lifecycle.lifecycle_business_as_of,
                lifecycle_knowledge_as_of=context.artifacts.lifecycle.lifecycle_knowledge_as_of,
                policy_version=context.artifacts.lifecycle.policy_version,
                parent_artifact_ids=tuple(
                    item.artifact_id
                    for item in (context.artifacts.occurrences, context.artifacts.verification)
                    if item is not None
                ),
            )
            context.artifacts.lifecycle = LifecycleArtifact(
                **{**lifecycle.__dict__, "artifact_id": artifact_id_of(lifecycle)}
            )

    @staticmethod
    def _overlay_fixture_lineage(
        records: list[dict],
        claims: list[FinancialClaim],
        occurrences: list[ClaimOccurrence],
    ) -> bool:
        """Attach canonical IDs to offline read projections without replacing them.

        The semantic stages are the source of truth for claims and occurrences.
        Fixture extraction only supplies the legacy read-model shape (including
        stable fixture knowledge UIDs), so a cardinality mismatch is rejected
        by returning ``False`` to the authoritative projection fallback.
        """
        if not (len(records) == len(claims) == len(occurrences)):
            return False
        for record, claim, occurrence in zip(records, claims, occurrences):
            record["claim_type"] = claim.claim_type
            attributes = dict(record.get("attributes") or {})
            attributes.update({
                "claim_id": claim.claim_id,
                "occurrence_id": occurrence.occurrence_id,
                "semantic_segment_id": occurrence.semantic_segment_id,
                "asserted_at": occurrence.times.asserted_at.isoformat()
                if occurrence.times.asserted_at else None,
                "source_published_at": occurrence.times.source_published_at.isoformat()
                if occurrence.times.source_published_at else None,
                "source_available_at": occurrence.times.source_available_at.isoformat()
                if occurrence.times.source_available_at else None,
                "source_availability_quality": occurrence.times.source_availability_quality.value,
                "ingested_at": occurrence.times.ingested_at.isoformat(),
                "extraction_completed_at": occurrence.times.extraction_completed_at.isoformat(),
                "available_from": occurrence.times.available_from.isoformat(),
            })
            record["attributes"] = attributes
            record["_claim_id"] = claim.claim_id
        return True

    def execute(self, context: PipelineContext) -> PipelineContext:
        available_from = self._timestamp(context.options.get("available_from") or context.options.get("as_of"))
        fixture = bool(
            context.options.get("offline_fixture")
            or "transcript" in context.options
            or "segments" in context.options
        )
        if self._authoritative_only and not fixture:
            return self._project_authoritative(context, available_from)
        if fixture and self._authoritative_only:
            records = self._fixture_records(context, available_from)
            # Offline fixtures are a read-model compatibility overlay.  Never
            # synthesize or replace the canonical semantic claim/occurrence
            # objects already produced by the preceding stages.
            if not self._overlay_fixture_lineage(
                records,
                list(context.state.get("claims") or ()),
                list(context.state.get("occurrences") or ()),
            ):
                return self._project_authoritative(context, available_from)
            claim_artifact = context.artifacts.claims
            results = [
                VerificationResult(
                    claim_id=claim.claim_id,
                    status=(
                        "VERIFICATION_PENDING" if claim.fact_category == "FACT" else "NOT_REQUIRED"
                    ),
                )
                for claim in context.state.get("claims") or ()
            ]
            verification = VerificationArtifact(
                artifact_id="verification-pending",
                artifact_type="verification",
                producer_stage="verification",
                claim_artifact_id=claim_artifact.artifact_id if claim_artifact else "",
                results=results,
                parent_artifact_ids=(claim_artifact.artifact_id,) if claim_artifact else (),
            )
            context.artifacts.verification = VerificationArtifact(
                **{**verification.__dict__, "artifact_id": artifact_id_of(verification)}
            )
        elif fixture:
            # Legacy/chapter-only configurations still rely on the fixture
            # extractor to materialize their compatibility claim chain.
            records = self._fixture_records(context, available_from)
            self._register_claim_chain(context, records)
        else:
            metadata = dict(context.state["metadata"])
            metadata.setdefault("platform", context.source["type"])
            metadata.setdefault("platform_video_id", context.source["ref"])
            metadata.setdefault("publish_time", available_from.isoformat())
            records = self._structured_extractor.extract(metadata, self._chapter_payload(context))
            records = self._normalizer.normalize(records, metadata)
            records = self._cross_modal.verify_many(records, list(context.state.get("ocr_evidence") or []))
            records = self._external.verify_many(records)
            records = self._temporal.apply(records, available_from)
            records = self._deduplicator.deduplicate(records)
        if not fixture:
            self._register_claim_chain(context, records)
        # Close the active semantic -> evidence -> occurrence -> claim chain
        # after the legacy-compatible extractor has produced final claims.
        if not fixture and context.artifacts.occurrences is not None:
            occurrence = ClaimOccurrenceArtifact(
                artifact_id="occurrences-chain-pending",
                artifact_type="occurrences",
                producer_stage="claim_occurrence_persistence",
                semantic_segment_artifact_id=(
                    context.artifacts.semantic_segments.artifact_id
                    if context.artifacts.semantic_segments
                    else ""
                ),
                evidence_artifact_id=(
                    context.artifacts.evidence.artifact_id if context.artifacts.evidence else ""
                ),
                occurrence_ids=list(context.artifacts.occurrences.occurrence_ids or ()),
                parent_artifact_ids=tuple(
                    item.artifact_id
                    for item in (context.artifacts.semantic_segments, context.artifacts.evidence)
                    if item is not None
                ),
            )
            context.artifacts.occurrences = ClaimOccurrenceArtifact(
                **{**occurrence.__dict__, "artifact_id": artifact_id_of(occurrence)}
            )
            claim_artifact = ClaimArtifact(
                artifact_id="claims-fixture-chain-pending",
                artifact_type="claims",
                producer_stage="claim_occurrence_persistence",
                evidence_artifact_id=(
                    context.artifacts.evidence.artifact_id if context.artifacts.evidence else ""
                ),
                claims=[claim.claim_id for claim in context.state.claims],
                parent_artifact_ids=(context.artifacts.occurrences.artifact_id,),
            )
            context.artifacts.claims = ClaimArtifact(
                **{**claim_artifact.__dict__, "artifact_id": artifact_id_of(claim_artifact)}
            )
        context.state["knowledge"] = self._to_domain(context.state["video"].video_id, records, available_from)
        claims_by_id = {claim.claim_id: claim for claim in context.state.claims}
        claims_by_uid = {
            str(record.get("knowledge_uid")): (
                claims_by_id.get(str(record.get("_claim_id")))
                or claim
            )
            for record, claim in zip(records, context.state.claims)
        }
        for unit in context.state.knowledge:
            claim = claims_by_uid.get(unit.knowledge_uid)
            claim_ids = [claim.claim_id] if claim else []
            unit.attributes = {
                **unit.attributes,
                "claim_ids": claim_ids,
                "source_support_status": claim.source_support_status if claim else "UNSUPPORTED",
            }
            unit.provenance = {**unit.provenance, "claim_ids": claim_ids}
        # P0 C-02：知识/claim 输出立即登记为 KnowledgeArtifact（claim/evidence ref 进入 lineage，
        # Fact/Forecast/Opinion 保留在 attributes，不压扁成普通字符串）。
        knowledge_units = context.state["knowledge"]
        knowledge = KnowledgeArtifact(
            artifact_id="knowledge-pending",
            artifact_type="knowledge",
            producer_stage="knowledge",
            verification_artifact_id=(
                context.artifacts.verification.artifact_id
                if context.artifacts.verification
                else ""
            ),
            knowledge_units=[unit.knowledge_uid for unit in knowledge_units],
            parent_artifact_ids=(
                (context.artifacts.verification.artifact_id,)
                if context.artifacts.verification
                else ()
            ),
        )
        context.artifacts.knowledge = KnowledgeArtifact(
            **{**knowledge.__dict__, "artifact_id": artifact_id_of(knowledge)}
        )
        return _stage_result(context, "evidence", "claims", "verification", "knowledge")

    def _project_authoritative(self, context: PipelineContext, available_from: datetime) -> PipelineContext:
        """Build the read model from the semantic canonical chain only.

        This branch deliberately does not call the legacy extractor or replace
        evidence/claims/occurrence/lifecycle slots.  It is a projection for
        search/read consumers, not another source of truth.
        """
        claims = list(context.state.get("claims") or ())
        occurrences = list(context.state.get("occurrences") or ())
        verification_results = [
            VerificationResult(
                claim_id=claim.claim_id,
                status="VERIFICATION_PENDING" if claim.fact_category == "FACT" else "NOT_REQUIRED",
            )
            for claim in claims
        ]
        claim_artifact = context.artifacts.claims
        verification = VerificationArtifact(
            artifact_id="verification-pending",
            artifact_type="verification",
            producer_stage="verification",
            claim_artifact_id=claim_artifact.artifact_id if claim_artifact else "",
            results=verification_results,
            parent_artifact_ids=(claim_artifact.artifact_id,) if claim_artifact else (),
        )
        context.artifacts.verification = VerificationArtifact(
            **{**verification.__dict__, "artifact_id": artifact_id_of(verification)}
        )
        if context.artifacts.lifecycle is not None:
            lifecycle = LifecycleArtifact(
                artifact_id="lifecycle-chain-pending",
                artifact_type="lifecycle",
                producer_stage="lifecycle_projection",
                claim_lifecycle_event_ids=list(context.artifacts.lifecycle.claim_lifecycle_event_ids),
                occurrence_lifecycle_event_ids=list(context.artifacts.lifecycle.occurrence_lifecycle_event_ids),
                lifecycle_business_as_of=context.artifacts.lifecycle.lifecycle_business_as_of,
                lifecycle_knowledge_as_of=context.artifacts.lifecycle.lifecycle_knowledge_as_of,
                policy_version=context.artifacts.lifecycle.policy_version,
                parent_artifact_ids=tuple(
                    item.artifact_id for item in (context.artifacts.occurrences, context.artifacts.verification) if item
                ),
            )
            context.artifacts.lifecycle = LifecycleArtifact(
                **{**lifecycle.__dict__, "artifact_id": artifact_id_of(lifecycle)}
            )
        occurrences_by_claim: dict[str, list[ClaimOccurrence]] = {}
        for item in occurrences:
            occurrences_by_claim.setdefault(item.claim_id, []).append(item)
        records: list[dict[str, Any]] = []
        for claim in claims:
            occurrence = (occurrences_by_claim.get(claim.claim_id) or [None]).pop(0)
            projection = KnowledgeProjectionBuilder().build(claim, occurrence, verification_results[len(records)])
            projected_attributes = dict(projection.get("attributes", {}))
            if occurrence is not None:
                projected_attributes.update({
                    "source_available_at": (
                        occurrence.times.source_available_at.isoformat()
                        if occurrence.times.source_available_at else None
                    ),
                    "source_availability_quality": occurrence.times.source_availability_quality.value,
                    "ingested_at": occurrence.times.ingested_at.isoformat(),
                    "extraction_completed_at": occurrence.times.extraction_completed_at.isoformat(),
                    "available_from": occurrence.times.available_from.isoformat(),
                })
            records.append({
                "knowledge_uid": projection["knowledge_uid"],
                "statement": str(claim.value if isinstance(claim.value, str) else claim.predicate),
                "knowledge_kind": claim.fact_category,
                "subject_key": claim.subject_id,
                "ticker": claim.ticker,
                "sentiment": "NEUTRAL",
                "support_status": {
                    "SUPPORTED": "SOURCE_SUPPORTED",
                    "PARTIALLY_SUPPORTED": "SOURCE_PARTIAL",
                }.get(claim.source_support_status, "SOURCE_UNSUPPORTED"),
                "truth_status": "NOT_CHECKED",
                "lifecycle_status": "ACTIVE",
                "support_score": claim.source_confidence,
                "extraction_confidence": claim.extractor_confidence,
                "as_of_time": available_from,
                "attributes": projected_attributes,
                "extractor_version": claim.extraction_prompt_version,
                "schema_version": claim.claim_schema_version,
                "semantic_hash": claim.claim_id,
            })
        context.state["knowledge"] = self._to_domain(context.state["video"].video_id, records, available_from)
        knowledge = KnowledgeArtifact(
            artifact_id="knowledge-pending",
            artifact_type="knowledge",
            producer_stage="knowledge_projection",
            verification_artifact_id=context.artifacts.verification.artifact_id,
            knowledge_units=[unit.knowledge_uid for unit in context.state["knowledge"]],
            parent_artifact_ids=(context.artifacts.verification.artifact_id,),
        )
        context.artifacts.knowledge = KnowledgeArtifact(
            **{**knowledge.__dict__, "artifact_id": artifact_id_of(knowledge)}
        )
        return _stage_result(context, "verification", "knowledge")


class VerificationStage:
    name = "verification"
    required_inputs = ("claims",)
    output_types = ()

    def execute(self, context: PipelineContext) -> PipelineContext:
        # Source, semantic and cross-modal verification are performed before
        # conversion to the persistence model.  This stage is retained as a
        # named checkpoint for worker compatibility and deliberately does not
        # reintroduce substring-based verification.
        return _stage_result(context)


class FinancialEnrichmentStage:
    """Materialise numeric facts and events once for every downstream consumer."""

    name = "financial_enrichment"
    required_inputs = ("knowledge",)
    output_types = ()

    def __init__(self, extractor: FinancialEventExtractor | None = None) -> None:
        self._extractor = extractor or FinancialEventExtractor()

    def execute(self, context: PipelineContext) -> PipelineContext:
        records: list[dict] = []
        numeric_facts: list[dict] = []
        for unit in context.state["knowledge"]:
            # A numeric fact is observable only when the knowledge item is
            # observable.  Do not let a re-parser silently drop the temporal
            # boundary that protects downstream research from look-ahead.
            numerics = [asdict(item) for item in parse_financial_numerics(unit.statement)]
            evidence = list((unit.attributes or {}).get("evidence") or [])
            evidence_refs = [
                str(item.get("source_id") or item.get("frame_id") or "")
                for item in evidence
                if item.get("source_id") or item.get("frame_id")
            ]
            numeric_ids: list[str] = []
            for index, item in enumerate(numerics):
                digest = hashlib.sha256(
                    f"{unit.video_id}:{unit.knowledge_uid}:{index}:{item.get('raw_expression', '')}".encode()
                ).hexdigest()[:32]
                numeric_id = f"num_{digest}"
                numeric_ids.append(numeric_id)
                item.update(
                    {
                        "numeric_id": numeric_id,
                        "as_of_time": unit.as_of.isoformat(),
                        "available_from": unit.available_from.isoformat(),
                        "evidence_ref": evidence_refs[0] if evidence_refs else None,
                    }
                )
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
                    "numeric_ids": numeric_ids,
                    "evidence_ids": evidence_refs,
                }
            )
            numeric_facts.extend([{"knowledge_uid": unit.knowledge_uid, **item} for item in numerics])
        context.state["financial_numeric_facts"] = numeric_facts
        context.state["financial_events"] = self._extractor.extract(records)
        return _stage_result(context)


class SummaryStage:
    name = "summary"
    required_inputs = ("knowledge",)
    output_types = ("summary",)

    def __init__(self, generator: SummaryGenerator) -> None:
        self._generator = generator

    def execute(self, context: PipelineContext) -> PipelineContext:
        context.state["summary"] = self._generator.generate(
            context.state["video"], context.state["chapters"], context.state["knowledge"]
        )
        # P0 C-02：SummaryArtifact 登记，knowledge_artifact_id 指向本次已登记的 Knowledge Artifact。
        knowledge_artifact = context.artifacts.knowledge
        summary = context.state["summary"]
        summary_artifact = SummaryArtifact(
            artifact_id="summary-pending",
            artifact_type="summary",
            producer_stage="summary",
            knowledge_artifact_id=knowledge_artifact.artifact_id if knowledge_artifact else "",
            core_summary=summary.core_summary,
            parent_artifact_ids=(knowledge_artifact.artifact_id,) if knowledge_artifact else (),
        )
        context.artifacts.summary = SummaryArtifact(
            **{**summary_artifact.__dict__, "artifact_id": artifact_id_of(summary_artifact)}
        )
        return _stage_result(context, "summary")


class ContentSnapshotPersistError(RuntimeError):
    """P0 C-03：ContentSnapshot 创建失败必须使 task 失败，不得静默成功。"""


class SnapshotRecordingStage:
    """P0 C-03/C-04：在 persist 之前基于 pipeline 已生成的 typed Artifact 记录 ContentSnapshot。

    - 直接读取 context.artifacts，不再从 context.state 二次拼装；
    - mandatory artifact（source/transcript/knowledge/summary）缺失即失败；
    - 失败抛 ContentSnapshotPersistError → task FAILED（CONTENT_SNAPSHOT_PERSIST_FAILED）；
    - 成功后 content_snapshot_id 写入 knowledge attributes，signal v3 透传。
    """

    name = "content_snapshot"
    required_inputs = (
        "source", "media", "transcript", "semantic_segments", "evidence", "claims", "occurrences",
        "verification", "lifecycle", "knowledge", "summary",
    )
    output_types = ()

    def __init__(
        self,
        snapshot_service: SnapshotService,
        artifact_repository=None,
        occurrence_repository=None,
        lifecycle_repository=None,
    ) -> None:
        self._snapshots = snapshot_service
        self._artifact_repository = artifact_repository
        self._occurrence_repository = occurrence_repository
        self._lifecycle_repository = lifecycle_repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        registry = context.artifacts
        mandatory = {
            "source": registry.source, "media": registry.media, "transcript": registry.transcript,
            "evidence": registry.evidence, "claims": registry.claims, "verification": registry.verification,
            "knowledge": registry.knowledge, "summary": registry.summary,
        }
        semantic_enabled = bool((context.options.get("pipeline_config") or {}).get(
            "semantic_segmentation_enabled", True
        ))
        if semantic_enabled:
            mandatory.update({
                "semantic_segments": registry.semantic_segments,
                "occurrences": registry.occurrences,
                "lifecycle": registry.lifecycle,
            })
        missing = [slot for slot, artifact in mandatory.items() if artifact is None]
        if missing:
            raise ContentSnapshotPersistError(
                f"CONTENT_SNAPSHOT_PERSIST_FAILED: mandatory artifact missing: {sorted(missing)}"
            )
        if self._artifact_repository is not None:
            for artifact in registry.artifacts():
                self._artifact_repository.put(artifact)
        source_content_hash = str(registry.source.raw_content_hash or registry.source.source_content_hash or "")
        if not source_content_hash:
            raise ContentSnapshotPersistError("CONTENT_SNAPSHOT_PERSIST_FAILED: source raw hash missing")
        try:
            if self._occurrence_repository is not None:
                for occurrence in context.state.get("occurrences") or ():
                    validator = getattr(self._occurrence_repository, "validate_immutable", None)
                    if validator is not None:
                        validator(occurrence)
            producer_manifest = _producer_manifest(context)
            reference_records = []
            reference_snapshot_ids = set()
            for binding in context.state.get("temporal_bindings") or ():
                snapshot_id = getattr(binding, "reference_snapshot_id", None)
                if not snapshot_id:
                    continue
                reference_snapshot_ids.add(str(snapshot_id))
                reference_records.append({
                    "snapshot_id": str(snapshot_id),
                    "data_version": getattr(binding, "reference_data_version", None),
                    "available_at": (
                        binding.reference_available_at.isoformat()
                        if getattr(binding, "reference_available_at", None) is not None else None
                    ),
                })
            reference_records.sort(key=lambda item: (
                item["snapshot_id"], item.get("data_version") or "", item.get("available_at") or ""
            ))
            if reference_records:
                producer_manifest["reference_data"] = reference_records
            manifest_models = dict(producer_manifest.get("models") or {})
            snapshot = self._snapshots.record_bundle_from_artifacts(
                source_type=context.source["type"],
                source_ref=context.source["ref"],
                source_content_hash=source_content_hash,
                artifact_ids=registry.artifact_ids(),
                source_artifact_id=registry.source.artifact_id,
                model_versions={
                    "asr_model": str(manifest_models.get("asr") or "unknown"),
                    "asr_model_version": str(manifest_models.get("asr_version") or "unknown"),
                    "ocr_model": str(manifest_models.get("ocr") or "unknown"),
                    "ocr_model_version": str(manifest_models.get("ocr_version") or "unknown"),
                    "llm_model": str(manifest_models.get("llm") or "unknown"),
                    "vision_model": str(manifest_models.get("vision") or "unknown"),
                    "embedding_model": str(manifest_models.get("embedding") or "unknown"),
                },
                producer_manifest=producer_manifest,
                code_sha=str(producer_manifest["code_sha"]),
                prompt_versions={
                    "extraction": context.options.get("extraction_prompt_version", "extraction.v1"),
                    "normalization": context.options.get(
                        "normalization_prompt_version", "normalization.v1"
                    ),
                    "verification": context.options.get(
                        "verification_prompt_version", "verification.v1"
                    ),
                    "summary": context.options.get("summary_prompt_version", "summary.v1"),
                },
                configuration={
                    **dict(context.options.get("pipeline_config") or {}),
                },
                external_snapshots=tuple(
                    sorted(
                        reference_snapshot_ids
                        | {
                            str(item)
                            for item in (
                                context.options.get("external_snapshot_ids")
                                or context.options.get("quant_market_snapshot_ids")
                                or ()
                            )
                        }
                    )
                ),
                policy_versions={
                    "claim": context.options.get("claim_policy_version", "claim_policy.v1"),
                    "verification": context.options.get(
                        "verification_policy_version", "verification_policy.v1"
                    ),
                    "signal": context.options.get("signal_policy_version", "signal_policy.v1"),
                },
                quant_market_snapshot_ids=sorted({
                    str(item) for item in (context.options.get("quant_market_snapshot_ids") or ())
                } | reference_snapshot_ids),
                config_hash=str(producer_manifest["configs"]["config_hash"]),
                snapshot_kind=str(context.options.get("replay_snapshot_kind") or "INITIAL"),
                parent_snapshot_id=context.options.get("replay_parent_snapshot_id"),
                supersedes_snapshot_id=context.options.get("replay_supersedes_snapshot_id"),
                pipeline_version=str(context.options.get("replay_pipeline_version") or "pipeline.v3"),
                created_at=(
                    context.state.occurrences[0].times.snapshot_committed_at
                    if context.state.get("occurrences") else _stage_timestamp(context)
                ),
                occurrences=tuple(context.state.get("occurrences") or ()),
                lifecycle_events=tuple(context.state.get("lifecycle_events") or ()),
            )
        except Exception as exc:  # noqa: BLE001 - 显式失败，绝不静默
            raise ContentSnapshotPersistError(f"CONTENT_SNAPSHOT_PERSIST_FAILED: {exc}") from exc
        context.state["content_snapshot_id"] = snapshot.content_snapshot_id
        # ``record_bundle_from_artifacts`` commits occurrence/lifecycle rows
        # with the snapshot for SQL stores.  Keep the explicit repositories as
        # a compatibility fallback for custom stores without that capability.
        if not hasattr(getattr(self._snapshots, "_store", None), "save_bundle"):
            if self._occurrence_repository is not None:
                for occurrence in context.state.get("occurrences") or ():
                    self._occurrence_repository.save(occurrence)
            if self._lifecycle_repository is not None:
                for event in context.state.get("lifecycle_events") or ():
                    self._lifecycle_repository.append(event)
        # snapshot identity 回填知识 attributes，供 signal v3 透传 content_snapshot_id。
        for unit in context.state.get("knowledge") or []:
            attributes = dict(unit.attributes or {})
            attributes["content_snapshot_id"] = snapshot.content_snapshot_id
            unit.attributes = attributes
        return _stage_result(context)


def _producer_manifest(context: PipelineContext) -> dict[str, Any]:
    """Normalize immutable release provenance without task metadata."""
    manifest = dict(context.options.get("producer_manifest") or {})
    dependency = context.options.get("dependency_lock_hash", manifest.get("python_lock_hash", "unknown"))
    # The explicit top-level option is authoritative when it conflicts with a
    # nested manifest value.  Otherwise preserve an explicitly supplied
    # manifest value, then fall back to the deployment/default release SHA.
    manifest_code_sha = manifest.get("code_sha")
    effective_code_sha = (
        context.options.get("code_sha")
        or manifest_code_sha
        or default_code_sha()
    )
    manifest["code_sha"] = str(effective_code_sha)
    manifest.setdefault(
        "container_digest",
        context.options.get("container_digest", manifest.get("container_image", "unknown")),
    )
    manifest.setdefault("dependency_lock_hash", dependency)
    manifest.setdefault("python_lock_hash", manifest.get("dependency_lock_hash", dependency))
    # Keep the complete release manifest in the immutable snapshot identity.
    # Explicit nested values always win, while the defaults mirror the
    # separately persisted snapshot provenance fields.
    transcript = context.artifacts.transcript
    models = dict(manifest.get("models") or {})
    models.setdefault(
        "asr",
        (transcript.asr_model if transcript else None)
        or context.options.get("asr_model")
        or "unknown",
    )
    models.setdefault(
        "asr_version",
        (transcript.asr_model_version if transcript else None)
        or context.options.get("asr_model_version")
        or "unknown",
    )
    models.setdefault("ocr", context.options.get("ocr_model") or "fixture")
    models.setdefault("ocr_version", context.options.get("ocr_model_version") or "1")
    models.setdefault("vision", context.options.get("vision_model") or "unknown")
    pipeline_config = dict(context.options.get("pipeline_config") or {})
    models.setdefault(
        "segmentation",
        context.options.get("segmentation_model") or pipeline_config.get("segmentation_model") or "unknown",
    )
    models.setdefault(
        "extraction",
        context.options.get("extraction_model") or pipeline_config.get("extraction_model") or "unknown",
    )
    models.setdefault("llm", context.options.get("llm_model") or models.get("extraction") or "unknown")
    models.setdefault("embedding", context.options.get("embedding_model") or "unknown")
    manifest["models"] = models
    semantic_artifact = context.artifacts.semantic_segments
    manifest.setdefault(
        "semantic_segmentation",
        {
            "model": getattr(semantic_artifact, "model_id", None) or models.get("segmentation", "unknown"),
            "prompt": getattr(semantic_artifact, "prompt_version", None)
            or (context.options.get("pipeline_config") or {}).get(
                "segmentation_prompt_version", "semantic-segmentation.prompt.v1"
            ),
            "schema": getattr(semantic_artifact, "segmentation_schema_version", None)
            or "semantic-segment.v1",
        },
    )
    manifest.setdefault(
        "atomic_claim_extraction",
        {
            "model": pipeline_config.get("extraction_model")
            or context.options.get("llm_model") or models.get("llm", "unknown"),
            "prompt": context.options.get("atomic_claim_prompt_version")
            or (context.options.get("pipeline_config") or {}).get(
                "extraction_prompt_version", "atomic-claim-extraction.prompt.v1"
            ),
            "schema": "claim-occurrence-draft.v1",
        },
    )
    manifest.setdefault(
        "temporal_normalization",
        {
            "version": (context.options.get("pipeline_config") or {}).get(
                "temporal_normalization_version", "temporal-normalization.final.v1"
            ),
            "deterministic": True,
        },
    )
    prompts = dict(manifest.get("prompts") or {})
    prompts.setdefault("extraction", context.options.get("extraction_prompt_version", "extraction.v1"))
    prompts.setdefault("normalization", context.options.get("normalization_prompt_version", "normalization.v1"))
    prompts.setdefault("verification", context.options.get("verification_prompt_version", "verification.v1"))
    prompts.setdefault("summary", context.options.get("summary_prompt_version", "summary.v1"))
    manifest["prompts"] = prompts
    configs = dict(manifest.get("configs") or {})
    # As with code_sha, an explicit option wins.  A nested manifest value is
    # retained when present; otherwise derive the hash from pipeline_config.
    effective_config_hash = (
        context.options.get("config_hash")
        or configs.get("config_hash")
        or _config_hash_of(context.options.get("pipeline_config"))
    )
    configs["config_hash"] = str(effective_config_hash)
    configs.setdefault("entity_alias_version", context.options.get("entity_alias_version", "entity_alias.v1"))
    manifest["configs"] = configs
    return manifest


class ClaimPersistenceStage:
    """Persist canonical evidence/claims before creating a snapshot row."""

    name = "claim_persistence"
    required_inputs = ("evidence", "claims")
    output_types = ()

    def __init__(self, claims: Any, artifacts: Any) -> None:
        self._claims = claims
        self._artifacts = artifacts

    def execute(self, context: PipelineContext) -> PipelineContext:
        if self._artifacts is not None and context.artifacts.evidence is not None:
            self._artifacts.put(context.artifacts.evidence)
        occurrence_refs = {
            item.claim_id: list(item.evidence_refs)
            for item in context.state.get("occurrences") or ()
        }
        for claim in context.state.claims:
            self._claims.save(
                claim,
                compatibility_evidence_refs=occurrence_refs.get(claim.claim_id),
            )
        if self._artifacts is not None and context.artifacts.claims is not None:
            self._artifacts.put(context.artifacts.claims)
            if hasattr(self._artifacts, "put_claim_members"):
                self._artifacts.put_claim_members(context.artifacts.claims)
        context.state.claims_persisted = True
        return _stage_result(context)


def _config_hash_of(config: dict | None) -> str:
    import hashlib
    import json

    if not config:
        return ""
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


class PersistStage:
    name = "persist"
    required_inputs = ("summary",)
    output_types = ()

    def __init__(
        self,
        videos: VideoRepository,
        chapters: ChapterRepository,
        knowledge: KnowledgeRepository,
        summaries: SummaryRepository,
        multimodal: MultimodalRepository | None = None,
        financial=None,
        entities=None,
        verifications=None,
        artifacts=None,
        claims=None,
        snapshot_service=None,
        signal_service=None,
        signal_outbox=None,
    ) -> None:
        self._videos = videos
        self._chapters = chapters
        self._knowledge = knowledge
        self._summaries = summaries
        self._multimodal = multimodal
        self._financial = financial
        self._entities = entities
        self._verifications = verifications
        self._artifacts = artifacts
        self._claims = claims
        self._snapshot_service = snapshot_service
        self._signal_service = signal_service
        self._signal_outbox = signal_outbox

    def execute(self, context: PipelineContext) -> PipelineContext:
        if self._artifacts:
            for artifact in context.artifacts.artifacts():
                self._artifacts.put(artifact)
        if self._claims and not context.state.claims_persisted:
            for claim in context.state.claims:
                self._claims.save(claim)
        # Jobs are enqueued only after SnapshotRecordingStage has succeeded.
        if self._claims and hasattr(self._claims, "enqueue_verification_jobs"):
            self._claims.enqueue_verification_jobs(context.state.claims, context.trace.get("trace_id"))
        if self._snapshot_service and self._signal_service and self._signal_outbox and context.artifacts.verification:
            snapshot = self._snapshot_service.get(context.state.get("content_snapshot_id", ""))
            if snapshot is not None:
                for result in context.artifacts.verification.results:
                    claim = next((item for item in context.state.claims if item.claim_id == result.claim_id), None)
                    if claim is None or claim.claim_type in {"PRICE", "RETURN", "VALUATION", "FINANCIAL_METRIC"}:
                        continue
                    self._signal_service.enqueue_initial(
                        self._signal_outbox,
                        snapshot,
                        claim,
                        result.model_dump(mode="json") | {"provider": "none"},
                        verification_artifact_id=context.artifacts.verification.artifact_id,
                        trace_id=context.trace.get("trace_id"),
                        decision_id=context.trace.get("decision_id"),
                    )
        if self._artifacts and context.artifacts.claims is not None and hasattr(
            self._artifacts, "put_claim_members"
        ):
            self._artifacts.put_claim_members(context.artifacts.claims)
        video = context.state["video"]
        self._videos.upsert(video, context.state["segments"])
        self._chapters.replace_for_video(video.video_id, context.state["chapters"])
        self._knowledge.replace_for_video(video.video_id, context.state["knowledge"])
        if self._verifications:
            # Verification ledger trace must identify the request lineage, not
            # the task UUID.  The latter is an operational identifier and is
            # already persisted on the task/checkpoint rows.
            self._verifications.append(context.state["knowledge"], context.trace.get("trace_id"))
        if self._multimodal:
            self._multimodal.replace(
                video.video_id,
                list(context.state.get("frames") or []),
                list(context.state.get("ocr_evidence") or []),
                list(context.state.get("frame_insights") or []),
                list(context.state.get("temporal_windows") or []),
            )
        if self._financial:
            self._financial.replace(
                video.video_id,
                list(context.state.get("financial_numeric_facts") or []),
                list(context.state.get("financial_events") or []),
            )
        if self._entities:
            self._entities.replace(video.video_id, context.state["knowledge"])
        self._summaries.upsert(context.state["summary"])
        return _stage_result(context)


class IndexStage:
    name = "index"
    required_inputs = ("knowledge",)
    output_types = ()

    def __init__(self, index: KnowledgeIndex) -> None:
        self._index = index

    def execute(self, context: PipelineContext) -> PipelineContext:
        self._index.index(context.state["knowledge"])
        return _stage_result(context)


class BuildVideoStage:
    name = "transcript"
    required_inputs = ("transcript",)
    output_types = ()

    def execute(self, context: PipelineContext) -> PipelineContext:
        metadata = context.state.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("source metadata must be an object")
        def _resolved_value(name: str) -> Any:
            # Explicit options are the caller-visible resolution override;
            # otherwise use the authoritative adapter metadata unchanged.
            if context.options.get(name) is not None:
                return context.options[name]
            return metadata.get(name)

        published_at = _resolved_datetime(_resolved_value("published_at"), "published_at")
        resolved_at = _resolved_datetime(_resolved_value("resolved_at"), "resolved_at")
        canonical_url = _resolved_value("canonical_url")
        if canonical_url is None:
            candidate_url = metadata.get("source_ref")
            if isinstance(candidate_url, str) and candidate_url.startswith(("http://", "https://")):
                canonical_url = candidate_url
        if canonical_url is not None and not isinstance(canonical_url, str):
            raise ValueError("invalid canonical_url: expected string")
        source_version = _resolved_value("source_version")
        if source_version is not None and not isinstance(source_version, str):
            raise ValueError("invalid source_version: expected string")
        source_key = f"{context.source['type']}:{context.source['ref']}"
        video_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:32]
        transcript = context.state["transcript"]
        context.state["video"] = VideoAsset(
            video_id=video_id,
            source_type=context.source["type"],
            source_ref=context.source["ref"],
            title=metadata.get("title") or context.source["ref"],
            author=metadata.get("author"),
            duration_seconds=metadata.get("duration_seconds"),
            transcript_text=transcript,
            source_hash=(context.artifacts.source.raw_content_hash if context.artifacts.source else ""),
            canonical_url=canonical_url,
            published_at=published_at,
            source_version=source_version,
            metadata=dict(metadata),
            resolved_at=resolved_at,
        )
        return _stage_result(context)

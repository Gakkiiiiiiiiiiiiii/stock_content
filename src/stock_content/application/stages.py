from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from stock_content.adapters.http.model_client import ContentModelClient
from stock_content.application.fenced_effects import EffectIntent
from stock_content.application.pipeline import PipelineContext
from stock_content.application.sealed_media import SealedMediaValidationError, validate_sealed_media
from stock_content.application.snapshot_service import SnapshotService, choose_snapshot_commit_candidate
from stock_content.application.stage_runner import StageResult
from stock_content.application.transcript_quality_service import TranscriptQualityService
from stock_content.application.transcript_selection_service import TranscriptSelectionError, TranscriptSelectionService
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
    TranscriptVisualCrosscheckArtifact,
    TranscriptVisualCrosscheckRecord,
    VerificationArtifact,
    VisionArtifact,
    artifact_id_of,
    canonical_json,
)
from stock_content.domain.atomic_claim_extractor import AtomicClaimExtractor
from stock_content.domain.atomic_claim_validator import AtomicClaimDraftValidator
from stock_content.domain.chapter import ChapterSegmenter
from stock_content.domain.claim_canonicalizer import ClaimCanonicalizer
from stock_content.domain.claim_draft import ClaimOccurrenceDraft, TemporalExpressionDraft, VisualEvidenceAnchor
from stock_content.domain.claim_draft_grounder import ClaimDraftGrounder
from stock_content.domain.claim_evidence_verifier import ClaimEvidenceVerifier
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claim_state_event import ClaimStateEvent, event_logical_identity
from stock_content.domain.claims import FinancialClaim, VerificationResult, normalized_ticker
from stock_content.domain.cross_modal_evidence_verifier import CrossModalEvidenceVerifier
from stock_content.domain.external_fact_verifier import ExternalFactVerifier
from stock_content.domain.financial_event_extractor import FinancialEventExtractor
from stock_content.domain.financial_numeric import parse_financial_numerics
from stock_content.domain.governance_evidence import governance_evidence_for, redact_pii
from stock_content.domain.initial_verification import build_initial_verification_plan
from stock_content.domain.knowledge import KnowledgeExtractor
from stock_content.domain.knowledge_deduplicator import KnowledgeDeduplicator
from stock_content.domain.knowledge_evidence_window import KnowledgeEvidenceWindowPlanner
from stock_content.domain.knowledge_frame_plan import (
    KnowledgeFramePlanner,
    evidence_window_id,
    frame_id_for,
    request_id_for,
)
from stock_content.domain.knowledge_projection_builder import KnowledgeProjectionBuilder
from stock_content.domain.knowledge_semantics import atomic_statement, bundle_v2_semantics
from stock_content.domain.knowledge_temporal_policy import KnowledgeTemporalPolicy
from stock_content.domain.knowledge_unit_extractor import KnowledgeUnitExtractor
from stock_content.domain.knowledge_unit_normalizer import KnowledgeUnitNormalizer
from stock_content.domain.lifecycle_event import KnowledgeLifecycleEvent
from stock_content.domain.lineage import default_code_sha
from stock_content.domain.models import KnowledgeUnit, TranscriptSegment, VideoAsset
from stock_content.domain.semantic_context_builder import SemanticContextBuilder
from stock_content.domain.semantic_entailment_judge import SemanticEntailmentJudge
from stock_content.domain.semantic_segmenter import SemanticSegmenter
from stock_content.domain.source_policy import policy_for_source
from stock_content.domain.source_url import canonical_public_source_url
from stock_content.domain.summary import SummaryGenerator
from stock_content.domain.temporal_normalizer import TemporalNormalizer
from stock_content.domain.temporal_semantics import (
    MetricTemporalNature,
    OccurrenceTimes,
    TemporalAssertionStatus,
    TemporalRole,
    TemporalScope,
)
from stock_content.domain.transcript_candidate import (
    AlignmentStatus,
    TranscriptCandidate,
    TranscriptCandidateSegment,
    TranscriptSource,
)
from stock_content.domain.transcript_postprocessor import TranscriptPostprocessor
from stock_content.domain.transcript_visual_crosscheck import TranscriptVisualCrossChecker
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

    @staticmethod
    def _resolve_materialization(adapter: Any, context: PipelineContext):
        kwargs: dict[str, Any] = {"part": context.options.get("part")}
        if context.source["type"] in {"bilibili", "xiaoe", "xiaoe_hls"} and context.options.get("credential_ref_hash"):
            # This is a one-way reference recovered only against the worker's
            # configured allowlist.  It is safe in a checkpoint but never
            # reveals a credential name, storage state, cookie, or locator.
            kwargs["credential_ref_hash"] = context.options.get("credential_ref_hash")
        return adapter.resolve_materialization(context.source["ref"], **kwargs)

    def execute(self, context: PipelineContext) -> PipelineContext:
        fixture = context.options.get("metadata")
        adapter = self._adapters[context.source["type"]]
        sealed_source_id = str(context.options.get("replay_sealed_source_artifact_id") or "")
        if sealed_source_id:
            metadata = context.options.get("replay_sealed_source_metadata")
            if not isinstance(metadata, dict) or not context.options.get("replay_raw_storage_uri"):
                raise RuntimeError("REPLAY_INPUT_UNAVAILABLE: sealed replay source metadata is unavailable")
            context.state["metadata"] = dict(metadata)
        elif fixture:
            context.state["metadata"] = fixture
        elif hasattr(adapter, "resolve_materialization"):
            materialization = self._resolve_materialization(adapter, context)
            # The Pydantic JSON projection has no SecretStr value; ephemeral
            # stream/subtitle URLs remain in RuntimeWorkspace only.
            context.runtime.source_materialization = materialization
            context.state["metadata"] = materialization.public.model_dump(mode="json")
        else:
            context.state["metadata"] = adapter.resolve(context.source["ref"])
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
    # Materializers expose only a public projection. Normalize the finite
    # provenance fields while runtime-only stream URLs remain out of it.
    # ``canonical_source_ref`` is intentionally a compact Xiaoe
    # product/lesson identity, not a public browser URL.  Prefer the explicit
    # resolver-provided page projection when it exists.
    canonical_ref = metadata.get("canonical_url") or metadata.get("canonical_source_ref")
    if not canonical_ref and source_type == "bilibili" and str(source_ref).upper().startswith("BV"):
        canonical_ref = f"https://www.bilibili.com/video/{source_ref.upper()}"
    canonical_ref = canonical_public_source_url(source_type, canonical_ref)
    metadata.update(
        {
            "canonical_url": canonical_ref,
            "source_id": metadata.get("platform_id") or metadata.get("source_id") or source_ref,
            "source_part": metadata.get("part_id") or metadata.get("source_part") or context.options.get("part"),
            "source_available_from": (
                context.options.get("replay_source_available_at")
                or context.options.get("source_available_at")
                or metadata.get("source_available_at")
                or metadata.get("available_at")
            ),
            "business_as_of": (
                context.options.get("replay_source_business_as_of")
                or context.options.get("business_as_of")
                or metadata.get("business_as_of")
            ),
            "pipeline_version": (
                context.options.get("replay_source_pipeline_version")
                or context.options.get("replay_pipeline_version")
                or "pipeline.v3"
            ),
        }
    )
    if context.options.get("source_artifact_metadata_required"):
        policy = policy_for_source(source_type)
        required = {
            "source_policy_version": context.options.get("source_policy_version"),
            "retention_class": context.options.get("retention_class"),
            "access_classification": context.options.get("access_classification"),
        }
        if any(not value for value in required.values()):
            raise ValueError("source artifact policy metadata is required for new ingestion")
        metadata.update(required)
        metadata.update(
            {
                "source_content_hash": raw_hash,
                "content_size": length,
                "mime_type": context.options.get("mime_type") or "application/octet-stream",
                "encryption_key_id": context.options.get("encryption_key_id"),
                "governance_evidence": governance_evidence_for(policy),
            }
        )
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

    @staticmethod
    def _resolve_materialization(adapter: Any, context: PipelineContext):
        kwargs: dict[str, Any] = {"part": context.options.get("part")}
        if context.source["type"] in {"bilibili", "xiaoe", "xiaoe_hls"} and context.options.get("credential_ref_hash"):
            kwargs["credential_ref_hash"] = context.options.get("credential_ref_hash")
        return adapter.resolve_materialization(context.source["ref"], **kwargs)

    def execute(self, context: PipelineContext) -> PipelineContext:
        fixture = bool(
            context.options.get("offline_fixture")
            or "transcript" in context.options
            or "segments" in context.options
            or "test_subtitle_candidates" in context.options
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
            sealed_source_id = str(context.options.get("replay_sealed_source_artifact_id") or "")
            if not sealed_source_id or not context.options.get("replay_sealed_snapshot_id"):
                raise RuntimeError("REPLAY_INPUT_UNAVAILABLE: unsealed replay media is not allowed")
            context.runtime.work_dir = Path(
                tempfile.mkdtemp(prefix=f"content-{context.task_id[:8]}-", dir=self._work_root)
            )
            try:
                sealed_media = validate_sealed_media(
                    str(replay_uri),
                    private_root=str(context.options.get("replay_sealed_media_root") or ""),
                    expected_hash=str(context.options.get("replay_expected_raw_hash") or ""),
                    expected_length=None,
                )
            except SealedMediaValidationError as exc:
                raise RuntimeError("REPLAY_INPUT_UNAVAILABLE: sealed raw media is unavailable") from exc
            path = sealed_media.path
            context.runtime.video_path = path
            _refresh_source_artifact(context, sealed_media.content_hash, sealed_media.content_length, str(path))
            source = context.artifacts.source
            if source is not None:
                media = MediaArtifact(
                    artifact_id="media-pending",
                    artifact_type="media",
                    source_artifact_id=source.artifact_id,
                    media_uri=str(path),
                    video_hash=sealed_media.content_hash,
                    extractor_version="download.v1",
                    parent_artifact_ids=(source.artifact_id,),
                )
                context.artifacts.media = MediaArtifact(**{**media.__dict__, "artifact_id": artifact_id_of(media)})
            return _stage_result(context, "source", "media")
        if (
            "transcript" in context.options
            or "segments" in context.options
            or "test_subtitle_candidates" in context.options
        ):
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
        adapter = self._adapters[context.source["type"]]
        materialization = context.runtime.source_materialization
        if hasattr(adapter, "resolve_materialization"):
            if materialization is None:
                materialization = self._resolve_materialization(adapter, context)
            materialized = adapter.materialize(
                materialization,
                directory,
                expected_duration=materialization.public.duration_seconds,
                reresolve=lambda: self._resolve_materialization(adapter, context),
            )
            context.runtime.video_path = materialized.path
            context.state.metadata["subtitle"] = materialized.subtitle_metadata
            context.runtime.subtitle_tracks = materialized.subtitle_tracks
            # The probe is the authoritative local-media duration.  It also
            # bounds parsed subtitle cues before they reach candidate stages.
            context.options.setdefault("duration_ms", round(materialized.duration_seconds * 1000))
        else:
            context.runtime.video_path = adapter.download(context.source["ref"], directory)
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
                    semantic_segment_ids=tuple(str(value) for value in item.get("semantic_segment_ids") or ()),
                    evidence_window_ids=tuple(str(value) for value in item.get("evidence_window_ids") or ()),
                    planner_version=str(item.get("planner_version") or ""),
                    parent_artifact_ids=(media.artifact_id,),
                )
                frame_artifacts.append(
                    FrameArtifact(**{**frame_artifact.__dict__, "artifact_id": artifact_id_of(frame_artifact)})
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


class FixtureFrameRegistrationStage:
    """Register deterministic, test-only visual fixtures without ffmpeg.

    Live ingest may only receive frames from ``KnowledgeDirectedFrameExtractionStage``.
    Keeping this isolated preserves existing offline pipeline fixtures without
    allowing a request option to bypass transcript-derived frame planning.
    """

    name = "frame_fixture"
    required_inputs = ("media",)
    output_types = ("frame",)
    optional_output_types = ("frame",)

    @staticmethod
    def _is_fixture(context: PipelineContext) -> bool:
        return bool(
            context.options.get("offline_fixture")
            or "transcript" in context.options
            or "segments" in context.options
            or "test_subtitle_candidates" in context.options
        )

    def execute(self, context: PipelineContext) -> StageResult:
        supplied_frames = context.options.get("frames")
        supplied_ids = {
            str(item.get("frame_id"))
            for key in ("ocr_evidence", "frame_insights")
            for item in (context.options.get(key) or [])
            if isinstance(item, dict) and item.get("frame_id")
        }
        if supplied_frames is None and not supplied_ids:
            return StageResult(context=context)
        if not self._is_fixture(context):
            raise ValueError("live media cannot register fixture frames; use knowledge-directed extraction")
        frames = list(supplied_frames or [])
        existing_ids = {str(item.get("frame_id")) for item in frames if isinstance(item, dict) and item.get("frame_id")}
        for frame_id in sorted(supplied_ids - existing_ids):
            frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": 0,
                    "image_hash": hashlib.sha256(frame_id.encode()).hexdigest(),
                    "storage_ref": f"fixture://frame/{frame_id}",
                }
            )
        media = context.artifacts.media
        if media is None:
            raise ValueError("fixture frame registration requires media")
        artifacts: list[FrameArtifact] = []
        for index, frame in enumerate(frames):
            item = frame if isinstance(frame, dict) else {"image_path": str(frame)}
            frame_artifact = FrameArtifact(
                artifact_id="frame-pending",
                artifact_type="frame",
                producer_stage=self.name,
                media_artifact_id=media.artifact_id,
                frame_id=str(item.get("frame_id") or f"fixture-frame-{index}"),
                timestamp_ms=int(item.get("timestamp_ms") or 0),
                image_hash=FrameExtractionStage._image_hash(item),
                storage_ref=str(item.get("image_path") or item.get("storage_ref") or ""),
                extraction_reason="fixture",
                parent_artifact_ids=(media.artifact_id,),
            )
            artifact = FrameArtifact(**{**frame_artifact.__dict__, "artifact_id": artifact_id_of(frame_artifact)})
            context.artifacts.add("frames", artifact)
            artifacts.append(artifact)
            context.state.frames.append({**item, "frame_id": artifact.frame_id})
        return StageResult(context=context, produced_artifacts=tuple(artifacts))


class VisualEvidencePolicyStage:
    """Fail closed when a live video cannot enter semantic visual planning."""

    name = "visual_evidence_policy"
    required_inputs = ("media", "transcript")
    output_types = ()

    def __init__(self, *, semantic_segmentation_enabled: bool) -> None:
        self._semantic_segmentation_enabled = semantic_segmentation_enabled

    def execute(self, context: PipelineContext) -> StageResult:
        if context.runtime.video_path is not None and not self._semantic_segmentation_enabled:
            raise ValueError("live media visual extraction requires semantic segmentation")
        return StageResult(context=context)


class ASRStage:
    name = "asr"
    required_inputs = ("media",)
    output_types = ("transcript",)
    optional_output_types = ("transcript",)

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
        # SubtitleCandidateStage records whether speech is needed.  A direct
        # ASRStage invocation keeps its historical behaviour for callers/tests.
        if context.options.get("_transcript_candidate_mode") and not context.options.get("_asr_required"):
            return _stage_result(context)
        segments = self._fixture(context.options)
        if not segments and context.runtime.audio_path is not None:
            segments = self._recognizer.transcribe(context.runtime.audio_path, context.options.get("language"))
        if not segments:
            raise ValueError("ASR returned no transcript segments")
        detected_types: set[str] = set()
        for segment in segments:
            text = redact_pii(segment.text)
            segment.text = text.text
            detected_types.update(text.detected_types)
            if segment.raw_text is not None:
                raw = redact_pii(segment.raw_text)
                segment.raw_text = raw.text
                detected_types.update(raw.detected_types)
            if segment.normalized_text is not None:
                normalized = redact_pii(segment.normalized_text)
                segment.normalized_text = normalized.text
                detected_types.update(normalized.detected_types)
        if detected_types:
            context.state.quality_warnings.extend(f"PII_REDACTED:{category}" for category in sorted(detected_types))
        context.state["segments"] = segments
        context.state["transcript"] = " ".join(segment.text for segment in segments)
        if context.options.get("_transcript_candidate_mode"):
            candidate = _asr_candidate(context, segments)
            context.state.transcript_candidates.append(candidate)
            return StageResult(
                context=context,
                produced_artifacts=(context.state.transcript_candidate_artifacts[-1],),
            )
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
            source=item.source,
            source_artifact_id=item.source_artifact_id,
            alignment_status=item.alignment_status,
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
            context.artifacts.media.artifact_id if context.artifacts.media else (source.artifact_id if source else "")
        ),
        language=context.options.get("language"),
        segments=segments,
        # ASR model/version 进入 lineage（默认 faster-whisper，可被 options 覆盖）。
        asr_model=str(context.options.get("asr_model") or "faster-whisper"),
        asr_model_version=str(context.options.get("asr_model_version") or "1.0"),
        parent_artifact_ids=(parent_artifact_id,)
        if parent_artifact_id
        else ((context.artifacts.media.artifact_id,) if context.artifacts.media else ()),
    )
    context.artifacts.transcript = TranscriptArtifact(
        **{**transcript.__dict__, "artifact_id": artifact_id_of(transcript)}
    )


def _duration_ms(context: PipelineContext) -> int:
    value = context.options.get("duration_ms")
    if value is None:
        value = context.state.metadata.get("duration_ms")
    if value is None:
        seconds = context.state.metadata.get("duration_seconds")
        value = float(seconds) * 1000 if seconds is not None else None
    if value is None and context.artifacts.media is not None:
        value = context.artifacts.media.duration_ms
    if value is None:
        value = max((round(item.end_seconds * 1000) for item in context.state.segments), default=0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError("TRANSCRIPT_DURATION_INVALID")
    rounded = round(float(value))
    if rounded <= 0:
        raise ValueError("TRANSCRIPT_DURATION_INVALID")
    return rounded


def _candidate_artifact(
    context: PipelineContext, candidate: TranscriptCandidate, producer_stage: str
) -> TranscriptArtifact:
    media_id = context.artifacts.media.artifact_id if context.artifacts.media else ""
    transcript = TranscriptArtifact(
        artifact_id="transcript-pending",
        artifact_type="transcript",
        producer_stage=producer_stage,
        media_artifact_id=media_id,
        language=candidate.language,
        segments=[
            TranscriptSegmentItem(
                segment_index=index,
                start_seconds=item.start_ms / 1000,
                end_seconds=item.end_ms / 1000,
                text=item.normalized_text,
                raw_text=item.raw_text,
                normalized_text=item.normalized_text,
                confidence=item.confidence,
                source=item.source.value,
                source_artifact_id=item.source_artifact_id,
                alignment_status=item.alignment_status.value,
            )
            for index, item in enumerate(candidate.ordered_segments)
        ],
        asr_model=str(context.options.get("asr_model") or "candidate"),
        asr_model_version=str(context.options.get("asr_model_version") or "1.0"),
        parent_artifact_ids=(context.artifacts.media.artifact_id,) if context.artifacts.media else (),
    )
    return TranscriptArtifact(**{**transcript.__dict__, "artifact_id": artifact_id_of(transcript)})


def _asr_candidate(context: PipelineContext, segments: list[TranscriptSegment]) -> TranscriptCandidate:
    source_id = (
        "asr-"
        + hashlib.sha256(
            canonical_json(
                [(item.start_seconds, item.end_seconds, item.raw_text or item.text) for item in segments]
            ).encode()
        ).hexdigest()[:24]
    )
    converted = tuple(
        TranscriptCandidateSegment(
            source=TranscriptSource.ASR,
            source_artifact_id=source_id,
            start_ms=round(item.start_seconds * 1000),
            end_ms=round(item.end_seconds * 1000),
            raw_text=item.raw_text or item.text,
            normalized_text=item.normalized_text or item.text,
            confidence=0.0 if item.confidence is None else float(item.confidence),
            alignment_status=AlignmentStatus(item.alignment_status),
        )
        for item in segments
    )
    candidate = TranscriptCandidate(
        "asr-" + source_id,
        TranscriptSource.ASR,
        str(context.options.get("language") or "zh"),
        source_id,
        converted,
    )
    context.state.transcript_candidate_artifacts = getattr(context.state, "transcript_candidate_artifacts", [])
    context.state.transcript_candidate_artifacts.append(_candidate_artifact(context, candidate, "asr"))
    return candidate


class TranscriptCandidateStage:
    """Convert worker-materialized subtitle cues into immutable candidates."""

    name = "transcript_candidate"
    required_inputs = ("media",)
    output_types = ("transcript",)
    optional_output_types = ("transcript",)

    @staticmethod
    def _source(value: object) -> TranscriptSource:
        normalized = str(value or "").upper()
        aliases = {"OFFICIAL": "OFFICIAL_SUBTITLE", "MANUAL": "OFFICIAL_SUBTITLE", "AUTOMATIC": "AUTO_SUBTITLE"}
        return TranscriptSource(aliases.get(normalized, normalized))

    @staticmethod
    def _runtime_candidates(context: PipelineContext) -> list[dict[str, object]]:
        """Adapt the secret-free materializer output without serialising it.

        The source materializer is the sole production authority for subtitle
        bytes.  The small test adapter below exists only for deterministic
        unit fixtures and is not admitted by the canonical ingestion request.
        """
        result: list[dict[str, object]] = []
        for track in context.runtime.subtitle_tracks:
            origin = getattr(track, "source", None)
            if origin not in {"official", "automatic"}:
                raise ValueError("TRANSCRIPT_CANDIDATES_INVALID")
            result.append(
                {
                    "candidate_id": getattr(track, "artifact_id"),
                    "source": "OFFICIAL_SUBTITLE" if origin == "official" else "AUTO_SUBTITLE",
                    "source_artifact_id": getattr(track, "artifact_id"),
                    "language": getattr(track, "language"),
                    "segments": [
                        {
                            "start_ms": getattr(cue, "start_ms"),
                            "end_ms": getattr(cue, "end_ms"),
                            "raw_text": getattr(cue, "raw_text"),
                            "normalized_text": getattr(cue, "normalized_text"),
                            "confidence": 1.0 if origin == "official" else 0.8,
                        }
                        for cue in getattr(track, "cues", ())
                    ],
                }
            )
        return result

    def execute(self, context: PipelineContext) -> PipelineContext:
        raw_candidates = self._runtime_candidates(context)
        if not raw_candidates and context.options.get("_test_subtitle_candidate_adapter") is True:
            raw_candidates = context.options.get("test_subtitle_candidates") or []
        if not isinstance(raw_candidates, list):
            raise ValueError("TRANSCRIPT_CANDIDATES_INVALID")
        candidates: list[TranscriptCandidate] = []
        artifacts: list[TranscriptArtifact] = []
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                raise ValueError("TRANSCRIPT_CANDIDATES_INVALID")
            source = self._source(raw.get("source"))
            if source is TranscriptSource.ASR:
                raise ValueError("ASR must be provided by the ASR port")
            source_id = str(raw.get("source_artifact_id") or f"subtitle-{index}")
            converted = []
            for segment in raw.get("segments") or []:
                if not isinstance(segment, dict):
                    raise ValueError("TRANSCRIPT_SEGMENT_INVALID")
                raw_text = str(segment.get("raw_text") or segment.get("text") or "")
                normalized = str(segment.get("normalized_text") or raw_text)
                converted.append(
                    TranscriptCandidateSegment(
                        source=source,
                        source_artifact_id=source_id,
                        start_ms=int(segment.get("start_ms", round(float(segment.get("start_seconds", 0)) * 1000))),
                        end_ms=int(segment.get("end_ms", round(float(segment.get("end_seconds", 0)) * 1000))),
                        raw_text=raw_text,
                        normalized_text=normalized,
                        confidence=float(segment.get("confidence", 1.0)),
                        alignment_status=AlignmentStatus(str(segment.get("alignment_status") or "ALIGNED").upper()),
                    )
                )
            candidate = TranscriptCandidate(
                str(raw.get("candidate_id") or f"subtitle-{index}"),
                source,
                str(raw.get("language") or "zh"),
                source_id,
                tuple(converted),
            )
            candidates.append(candidate)
            artifacts.append(_candidate_artifact(context, candidate, "transcript_candidate"))
        context.state.transcript_candidates = candidates
        context.state.transcript_candidate_artifacts = artifacts
        context.options["_transcript_candidate_mode"] = True
        # ASR is only invoked when no candidate has already met the formal gate.
        if not candidates:
            context.options["_asr_required"] = True
        else:
            quality = TranscriptQualityService()
            duration = _duration_ms(context)
            context.options["_asr_required"] = not any(
                candidate.is_chinese
                and quality.evaluate(
                    candidate.ordered_segments, duration_ms=duration, language=candidate.language
                ).quality_status.value
                == "PASS"
                for candidate in candidates
                if candidate.source in {TranscriptSource.OFFICIAL_SUBTITLE, TranscriptSource.AUTO_SUBTITLE}
            )
        return StageResult(context=context, produced_artifacts=tuple(artifacts))


class TranscriptSelectionStage:
    name = "transcript_selection"
    required_inputs = ("media",)
    output_types = ("transcript",)

    def __init__(self, service: TranscriptSelectionService | None = None) -> None:
        self._service = service or TranscriptSelectionService()

    def execute(self, context: PipelineContext) -> PipelineContext:
        try:
            selection = self._service.select(
                tuple(context.state.transcript_candidates), duration_ms=_duration_ms(context)
            )
        except TranscriptSelectionError as exc:
            context.state.transcript_quality_report = exc.report
            context.state.quality_warnings.append("TRANSCRIPT_NEEDS_REVIEW")
            raise
        context.state.segments = [
            TranscriptSegment(
                segment_index=index,
                start_seconds=item.start_ms / 1000,
                end_seconds=item.end_ms / 1000,
                text=item.normalized_text,
                raw_text=item.raw_text,
                normalized_text=item.normalized_text,
                confidence=item.confidence,
                source=item.source.value,
                source_artifact_id=item.source_artifact_id,
                alignment_status=item.alignment_status.value,
            )
            for index, item in enumerate(selection.segments)
        ]
        context.state.transcript = " ".join(item.text for item in context.state.segments)
        context.state.transcript_quality_report = selection.report
        _register_transcript_artifact(context, producer_stage="transcript_selection")
        return _stage_result(context, "transcript")


class TranscriptQualityStage:
    name = "transcript_quality"
    required_inputs = ("transcript",)
    output_types = ()

    def execute(self, context: PipelineContext) -> PipelineContext:
        report = context.state.transcript_quality_report
        if report is None or report.quality_status.value != "PASS":
            context.state.quality_warnings.append("TRANSCRIPT_NEEDS_REVIEW")
            raise RuntimeError("NEEDS_REVIEW: transcript quality gate")
        return _stage_result(context)


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
        _register_transcript_artifact(context, producer_stage="diarization", parent_artifact_id=previous_transcript)
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


def _finite_confidence(value: Any, field: str = "confidence_score") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def _normalise_bbox(value: Any) -> list[Any]:
    """Accept only a rectangular, JSON-safe OCR coordinate payload.

    PaddleOCR has emitted both ``[left, top, right, bottom]`` and four
    corner-points over supported releases.  Preserve either exact shape while
    rejecting lossy/coerced coordinates and non-finite values.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("OCR bbox must contain four coordinates or corner points")
    if all(not isinstance(item, (list, tuple)) for item in value):
        return [_finite_coordinate(item) for item in value]
    if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
        raise ValueError("OCR bbox corner points must each contain two coordinates")
    return [[_finite_coordinate(coordinate) for coordinate in item] for item in value]


def _finite_coordinate(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("OCR bbox coordinates must be finite numbers")
    return value


def _frame_metadata(context: PipelineContext, frame_id: str, supplied: dict[str, Any]) -> dict[str, Any]:
    """Resolve immutable frame provenance without synthesising model evidence."""
    frame = next((item for item in context.artifacts.frames if item.frame_id == frame_id), None)
    if frame is not None:
        return {
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms,
            "image_hash": frame.image_hash,
            "semantic_segment_ids": list(frame.semantic_segment_ids),
            "evidence_window_ids": list(frame.evidence_window_ids),
            "frame_artifact_id": frame.artifact_id,
        }
    # Explicit test fixtures may intentionally have no materialised frame.
    # Preserve their supplied coordinate rather than creating a fake model
    # output; FrameExtractionStage creates deterministic parents in production.
    return {
        "frame_id": frame_id,
        "timestamp_ms": int(supplied.get("timestamp_ms") or 0),
        "image_hash": str(supplied.get("image_hash") or ""),
        "semantic_segment_ids": list(supplied.get("semantic_segment_ids") or ()),
        "evidence_window_ids": list(supplied.get("evidence_window_ids") or ()),
        "frame_artifact_id": "",
    }


def _require_model_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


class OCRStage:
    name = "ocr"
    required_inputs = ("media",)
    output_types = ("ocr",)
    optional_output_types = ("ocr",)

    def __init__(self, engine) -> None:
        self._engine = engine

    def execute(self, context: PipelineContext) -> PipelineContext:
        supplied = context.options.get("ocr_evidence")
        evidence: list[dict[str, Any]] = []
        if supplied is None:
            for frame in context.state.get("frames", []):
                result = self._engine.recognize(str(frame["image_path"]), str(frame.get("image_hash") or ""))
                if not isinstance(result, dict):
                    raise ValueError("OCR engine returned a non-object response")
                engine = _require_model_text(result.get("engine"), "OCR engine")
                engine_version = _require_model_text(result.get("engine_version"), "OCR engine_version")
                runtime_identity = result.get("runtime_identity") or {}
                if not isinstance(runtime_identity, dict):
                    raise ValueError("OCR runtime_identity must be an object")
                requested_device = str(result.get("requested_device") or "")
                actual_device = str(result.get("actual_device") or "")
                if requested_device == "gpu:0" and not actual_device.lower().startswith("gpu:0"):
                    raise ValueError("OCR actual device does not satisfy requested gpu:0")
                if runtime_identity:
                    frozen_identity = {str(key): str(value) for key, value in runtime_identity.items()}
                    existing = context.options.get("ocr_runtime_identity")
                    if existing and existing != frozen_identity:
                        raise ValueError("OCR runtime identity changed during task")
                    context.options["ocr_runtime_identity"] = frozen_identity
                raw_blocks = result.get("blocks")
                if not isinstance(raw_blocks, list):
                    raise ValueError("OCR blocks must be a list")
                blocks = []
                metadata = _frame_metadata(context, str(frame.get("frame_id") or ""), frame)
                for raw_block in raw_blocks:
                    if not isinstance(raw_block, dict):
                        raise ValueError("OCR block must be an object")
                    # Paddle may return an otherwise well-formed detection
                    # with an empty recognition string for a blank/transition
                    # frame.  That is *absence of evidence*, not an OCR
                    # artifact with fabricated text.  Keep the frame's
                    # runtime provenance, but produce zero blocks.  Missing
                    # or non-string text remains a malformed model response;
                    # non-empty blocks still pass the full strict schema
                    # checks below.
                    raw_text = raw_block.get("text")
                    if not isinstance(raw_text, str):
                        raise ValueError("OCR text must be a non-empty string")
                    text = raw_text.strip()
                    if not text:
                        continue
                    block = {
                        **metadata,
                        "source_type": "OCR",
                        "evidence_text": text,
                        "text": text,
                        "bbox": _normalise_bbox(raw_block.get("bbox")),
                        "confidence_score": _finite_confidence(raw_block.get("score"), "OCR score"),
                        "ocr_engine": engine,
                        "ocr_engine_version": engine_version,
                        "ocr_requested_device": requested_device,
                        "ocr_actual_device": actual_device,
                        "ocr_runtime_identity": runtime_identity,
                    }
                    blocks.append(block)
                    evidence.append(block)
                item = {
                    **frame,
                    "ocr_text": "\n".join(block["text"] for block in blocks),
                    "ocr_evidence": {"blocks": blocks},
                    "ocr_engine": engine,
                    "ocr_engine_version": engine_version,
                    "source_type": "OCR",
                }
                context.state.setdefault("frame_insights", []).append(item)
        else:
            # Explicit, caller-supplied offline fixtures stay deterministic.
            # They are not accepted as a substitute for a malformed engine
            # response and therefore retain their historical minimal shape.
            for raw in supplied:
                if not isinstance(raw, dict):
                    raise ValueError("OCR fixture evidence must be an object")
                frame_id = str(raw.get("frame_id") or "")
                if not frame_id:
                    raise ValueError("OCR fixture evidence requires frame_id")
                metadata = _frame_metadata(context, frame_id, raw)
                evidence.append({**metadata, **raw, "source_type": "OCR"})
        context.state["ocr_evidence"] = evidence
        ocr_artifacts = []
        for item in evidence:
            frame_id = str(item.get("frame_id") or "")
            metadata = _frame_metadata(context, frame_id, item)
            parent = metadata["frame_artifact_id"]
            is_fixture = supplied is not None
            confidence = item.get("confidence_score", item.get("score"))
            ocr = OCRArtifact(
                artifact_id="ocr-pending",
                artifact_type="ocr",
                frame_artifact_id=parent,
                frame_id=frame_id,
                timestamp_ms=metadata["timestamp_ms"],
                image_hash=metadata["image_hash"],
                semantic_segment_ids=tuple(metadata["semantic_segment_ids"]),
                evidence_window_ids=tuple(metadata["evidence_window_ids"]),
                text=str(item.get("evidence_text") or item.get("text") or ""),
                bbox=(item.get("bbox") if is_fixture else _normalise_bbox(item.get("bbox"))),
                confidence_score=(None if is_fixture and confidence is None else _finite_confidence(confidence)),
                blocks=[dict(item)],
                engine=str(item.get("ocr_engine") or "fixture"),
                engine_version=str(item.get("ocr_engine_version") or "fixture.v1"),
                requested_device=str(item.get("ocr_requested_device") or ""),
                actual_device=str(item.get("ocr_actual_device") or ""),
                runtime_identity={
                    str(key): str(value) for key, value in dict(item.get("ocr_runtime_identity") or {}).items()
                },
                parent_artifact_ids=(parent,) if parent else (),
            )
            ocr_artifacts.append(OCRArtifact(**{**ocr.__dict__, "artifact_id": artifact_id_of(ocr)}))
        context.artifacts.ocr = ocr_artifacts
        return _stage_result(context, "ocr")


def _transcript_context_for_frame(context: PipelineContext, frame: dict[str, Any]) -> str:
    """Keep a targeted visual request anchored to its semantic evidence window."""
    segment_ids = {str(value) for value in frame.get("semantic_segment_ids") or ()}
    if not segment_ids:
        return str(context.state.transcript)
    ranges = [
        (int(segment.start_ms), int(segment.end_ms))
        for segment in context.state.semantic_segments
        if str(segment.semantic_segment_id) in segment_ids
    ]
    if not ranges:
        return str(context.state.transcript)
    selected = []
    for item in context.state.segments:
        start_ms, end_ms = int(item.start_seconds * 1000), int(item.end_seconds * 1000)
        if any(start_ms <= high and end_ms >= low for low, high in ranges):
            selected.append(item.text)
    return " ".join(selected) or str(context.state.transcript)


def _normalise_string_list(value: Any, field: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    return list(value)


def _normalise_vision_item(context: PipelineContext, frame: dict[str, Any], result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("vision analyzer returned a non-object response")
    metadata = _frame_metadata(context, str(frame.get("frame_id") or ""), frame)
    visual_summary = _require_model_text(result.get("visual_summary"), "vision visual_summary")
    labels = _normalise_string_list(result.get("labels"), "vision labels", allow_empty=False)
    themes = _normalise_string_list(result.get("themes"), "vision themes")
    symbols = _normalise_string_list(result.get("symbols"), "vision symbols")
    narration_aligned = result.get("narration_aligned")
    if not isinstance(narration_aligned, bool):
        raise ValueError("vision narration_aligned must be boolean")
    model = _require_model_text(result.get("model"), "vision model")
    model_version = _require_model_text(result.get("model_version"), "vision model_version")
    normalized = {
        **frame,
        **metadata,
        "visual_summary": visual_summary,
        "label": labels[0],
        "labels": labels,
        "themes": themes,
        "symbols": symbols,
        "confidence_score": _finite_confidence(result.get("confidence_score"), "vision confidence_score"),
        "narration_aligned": narration_aligned,
        "model": model,
        "model_version": model_version,
        "source_type": "VISION",
    }
    # The only test-only adapter supplies these fields.  Keep its non-secret
    # provenance inside the normal vision artifact so C4/C5 replay/audit sees
    # the precise fixture identity without exposing it in Bundle v1.
    if "fixture_content_hash" in result:
        normalized.update(
            {
                "bbox": _normalise_bbox(result.get("bbox")),
                "environment": _require_model_text(result.get("environment"), "vision environment"),
                "fixture_content_hash": _require_model_text(
                    result.get("fixture_content_hash"), "vision fixture_content_hash"
                ),
                "frame_content_hash": _require_model_text(
                    result.get("frame_content_hash"), "vision frame_content_hash"
                ),
                "adapter_version": _require_model_text(result.get("adapter_version"), "vision adapter_version"),
            }
        )
    return normalized


class VisionStage:
    name = "vision"
    required_inputs = ("media",)
    output_types = ("vision",)
    optional_output_types = ("vision",)

    def __init__(self, analyzer) -> None:
        self._analyzer = analyzer

    def execute(self, context: PipelineContext) -> PipelineContext:
        insights = list(context.state.get("frame_insights") or [])
        vision_items: list[dict[str, Any]] = []
        if context.options.get("frame_insights") is not None:
            for raw in context.options["frame_insights"]:
                if not isinstance(raw, dict) or not str(raw.get("frame_id") or ""):
                    raise ValueError("vision fixture insight requires frame_id")
                metadata = _frame_metadata(context, str(raw["frame_id"]), raw)
                vision_items.append(
                    {
                        **metadata,
                        **raw,
                        "model": raw.get("model") or "fixture",
                        "model_version": raw.get("model_version") or "fixture.v1",
                    }
                )
        elif context.state.get("frames"):
            bind_context = getattr(self._analyzer, "bind_context", None)
            if callable(bind_context):
                # An adapter which opts into this hook is explicit.  No
                # dependency wiring calls it, so it cannot provide a runtime
                # fallback for an unconfigured production vision model.
                identity = bind_context(context)
                if not isinstance(identity, dict):
                    raise ValueError("vision fixture adapter identity must be an object")
                context.options["vision_fixture_identity"] = {str(key): str(value) for key, value in identity.items()}
            for frame in context.state["frames"]:
                transcript_context = _transcript_context_for_frame(context, frame)
                analyze_for_frame = getattr(self._analyzer, "analyze_for_frame", None)
                if callable(analyze_for_frame):
                    result = analyze_for_frame(frame, transcript_context)
                else:
                    result = self._analyzer.analyze(str(frame["image_path"]), transcript_context)
                vision_items.append(_normalise_vision_item(context, frame, result))
        # OCR contributes one context item per frame. Merge the corresponding
        # visual observation into that same item so bounded multimodal context
        # cannot crowd vision out behind duplicate OCR-only entries.
        insight_indexes = {
            str(item.get("frame_id") or ""): index
            for index, item in enumerate(insights)
            if isinstance(item, dict) and item.get("frame_id")
        }
        for item in vision_items:
            frame_id = str(item["frame_id"])
            if frame_id in insight_indexes:
                index = insight_indexes[frame_id]
                insights[index] = {**insights[index], **item}
            else:
                insight_indexes[frame_id] = len(insights)
                insights.append(item)
        context.state["frame_insights"] = insights
        vision_artifacts = []
        for item in vision_items:
            frame_id = str(item.get("frame_id") or "")
            metadata = _frame_metadata(context, frame_id, item)
            parent = metadata["frame_artifact_id"]
            is_fixture = context.options.get("frame_insights") is not None
            confidence = item.get("confidence_score")
            vision = VisionArtifact(
                artifact_id="vision-pending",
                artifact_type="vision",
                frame_artifact_id=parent,
                frame_id=frame_id,
                timestamp_ms=metadata["timestamp_ms"],
                image_hash=metadata["image_hash"],
                semantic_segment_ids=tuple(metadata["semantic_segment_ids"]),
                evidence_window_ids=tuple(metadata["evidence_window_ids"]),
                label=str(item.get("label") or item.get("description") or ""),
                labels=list(item.get("labels") or ()),
                confidence_score=(None if is_fixture and confidence is None else _finite_confidence(confidence)),
                payload=dict(item),
                model_name=str(item.get("model") or "fixture"),
                model_version=str(item.get("model_version") or "fixture.v1"),
                parent_artifact_ids=(parent,) if parent else (),
            )
            vision_artifacts.append(VisionArtifact(**{**vision.__dict__, "artifact_id": artifact_id_of(vision)}))
        context.artifacts.vision = vision_artifacts
        return _stage_result(context, "vision")


class TranscriptVisualCrosscheckStage:
    """Admit visual context only after deterministic transcript cross-check."""

    name = "transcript_visual_crosscheck"
    required_inputs = ("transcript", "semantic_segments")
    output_types = ("transcript_visual_crosscheck",)

    def __init__(self, checker: TranscriptVisualCrossChecker | None = None) -> None:
        self._checker = checker or TranscriptVisualCrossChecker()

    def execute(self, context: PipelineContext) -> StageResult:
        transcript = context.artifacts.transcript
        if transcript is None:
            raise ValueError("transcript visual crosscheck requires transcript")
        artifact_by_frame = {item.frame_id: item for item in context.artifacts.frames}
        ocr_by_frame: dict[str, list[dict[str, Any]]] = {}
        for item in context.artifacts.ocr:
            ocr_by_frame.setdefault(item.frame_id, []).append(
                {
                    "frame_id": item.frame_id,
                    "evidence_text": item.text,
                    "text": item.text,
                    "confidence_score": item.confidence_score,
                    "ocr_engine": item.engine,
                    "ocr_engine_version": item.engine_version,
                }
            )
        vision_by_frame = {item.frame_id: dict(item.payload) for item in context.artifacts.vision}
        segments_by_semantic: dict[str, list[Any]] = {}
        for semantic in context.state.get("semantic_segments") or ():
            segments_by_semantic[str(semantic.semantic_segment_id)] = [
                item
                for item in transcript.segments
                if semantic.start_segment_index <= item.segment_index <= semantic.end_segment_index
            ]
        checks: list[dict[str, Any]] = []
        eligible_ids: set[str] = set()
        for raw_frame in context.state.get("frames") or ():
            if not isinstance(raw_frame, dict):
                continue
            frame_id = str(raw_frame.get("frame_id") or "")
            artifact = artifact_by_frame.get(frame_id)
            frame = {
                **raw_frame,
                "frame_artifact_id": artifact.artifact_id if artifact else "",
                "semantic_segment_ids": list(
                    artifact.semantic_segment_ids if artifact else raw_frame.get("semantic_segment_ids") or ()
                ),
                "evidence_window_ids": list(
                    artifact.evidence_window_ids if artifact else raw_frame.get("evidence_window_ids") or ()
                ),
            }
            owned, seen = [], set()
            for semantic_id in frame["semantic_segment_ids"]:
                for segment in segments_by_semantic.get(str(semantic_id), ()):
                    if segment.segment_id not in seen:
                        seen.add(segment.segment_id)
                        owned.append(segment)
            check = self._checker.check(
                frame=frame,
                ocr_items=ocr_by_frame.get(frame_id, ()),
                vision_item=vision_by_frame.get(frame_id),
                transcript_segments=owned,
            )
            checks.append(check)
            if check["relation"] in {"SUPPORTS", "CONTRADICTS"}:
                eligible_ids.add(frame_id)
        context.state["transcript_visual_crosschecks"] = sorted(
            checks, key=lambda item: (item["timestamp_ms"], item["frame_id"])
        )
        context.state["eligible_frame_insights"] = [
            item
            for item in context.state.get("frame_insights") or []
            if str(item.get("frame_id") or "") in eligible_ids
        ]
        semantic = context.artifacts.semantic_segments
        identity = _visual_identity(context, self._checker.version)
        relation_records = tuple(TranscriptVisualCrosscheckRecord.from_dict(item) for item in checks)
        artifact = TranscriptVisualCrosscheckArtifact(
            artifact_id="transcript-visual-crosscheck-pending",
            artifact_type="transcript_visual_crosscheck",
            producer_stage=self.name,
            producer_version=self._checker.version,
            transcript_artifact_id=transcript.artifact_id,
            semantic_segment_artifact_id=semantic.artifact_id if semantic else "",
            crosscheck_version=self._checker.version,
            visual_identity=identity,
            relations=relation_records,
            eligible_frame_ids=tuple(sorted(eligible_ids)),
            parent_artifact_ids=tuple(
                sorted(
                    {
                        transcript.artifact_id,
                        *(item.artifact_id for item in context.artifacts.frames),
                        *(item.artifact_id for item in context.artifacts.ocr),
                        *(item.artifact_id for item in context.artifacts.vision),
                        *((semantic.artifact_id,) if semantic else ()),
                    }
                )
            ),
        )
        context.artifacts.set(
            "transcript_visual_crosscheck",
            TranscriptVisualCrosscheckArtifact(**{**artifact.__dict__, "artifact_id": artifact_id_of(artifact)}),
        )
        return _stage_result(context, "transcript_visual_crosscheck")


def _visual_identity(context: PipelineContext, crosscheck_version: str) -> dict[str, str]:
    """Return only replay-relevant, non-secret visual component identities."""
    config = dict(context.options.get("pipeline_config") or {})
    planner_versions = sorted(
        {str(item.planner_version) for item in context.artifacts.frames if str(item.planner_version or "")}
    )
    ocr_versions = sorted(
        {f"{item.engine}@{item.engine_version}" for item in context.artifacts.ocr if item.engine or item.engine_version}
    )
    vision_versions = sorted(
        {
            f"{item.model_name}@{item.model_version}"
            for item in context.artifacts.vision
            if item.model_name or item.model_version
        }
    )
    identity = {
        "crosscheck_version": str(crosscheck_version),
        "knowledge_evidence_window_planner_version": str(
            config.get("knowledge_evidence_window_planner_version") or "knowledge-evidence-window.v1"
        ),
        "knowledge_frame_planner_version": ",".join(
            planner_versions or [str(config.get("knowledge_frame_planner_version") or "knowledge-frame-plan.v1")]
        ),
        "ocr_engine_versions": ",".join(
            ocr_versions or [f"{config.get('ocr_engine') or 'paddleocr'}@{config.get('ocr_engine_version') or '3'}"]
        ),
        "vision_model_versions": ",".join(
            vision_versions
            or [
                f"{config.get('vision_model') or context.options.get('vision_model') or 'unconfigured'}@"
                f"{config.get('vision_model_version') or context.options.get('vision_model_version') or 'unconfigured'}"
            ]
        ),
        "vision_prompt_version": str(
            config.get("vision_prompt_version")
            or context.options.get("vision_prompt_version")
            or "vision-context.prompt.v1"
        ),
        "vision_adapter_version": str(config.get("vision_adapter_version") or "http-vision-adapter.v1"),
    }
    fixture = context.options.get("vision_fixture_identity")
    if isinstance(fixture, dict):
        identity.update(
            {
                "vision_fixture_adapter_version": str(fixture.get("adapter_version") or ""),
                "vision_fixture_environment": str(fixture.get("environment") or ""),
                "vision_fixture_content_hash": str(fixture.get("fixture_content_hash") or ""),
            }
        )
    return identity


def _eligible_visual_ids(context: PipelineContext) -> set[str]:
    artifact = context.artifacts.transcript_visual_crosscheck
    if artifact is not None:
        return set(artifact.eligible_frame_ids)
    return {
        str(item.get("frame_id") or "")
        for item in context.state.get("eligible_frame_insights") or ()
        if isinstance(item, dict) and str(item.get("frame_id") or "")
    }


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
        context.state["multimodal_context"] = self._builder.build(
            transcript, context.state.get("eligible_frame_insights") or []
        )
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
        context.state["temporal_windows"] = self._builder.build(
            transcript, context.state.get("eligible_frame_insights") or []
        )
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

        context.runtime.metrics.update(
            {
                "semantic_segments_per_video": float(count),
                "semantic_segment_duration_p50": _percentile(0.50),
                "semantic_segment_duration_p95": _percentile(0.95),
                "segmentation_repair_rate": float(result.metrics.get("repair_count", 0.0)) / max(1.0, float(count)),
                "segmentation_failure_rate": float(result.metrics.get("failure_count", 0.0)) / max(1.0, float(count)),
            }
        )
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


class KnowledgeDirectedFrameExtractionStage:
    """Materialize transcript-planned frames after semantic segmentation.

    OCR and vision deliberately remain untouched in this packet: they execute
    earlier in the existing graph.  A following packet can move or rerun those
    consumers after this stage without changing this stage's deterministic
    request/identity boundary.
    """

    name = "knowledge_frame"
    required_inputs = ("media", "transcript", "semantic_segments")
    output_types = ("frame",)
    optional_output_types = ("frame",)

    def __init__(
        self,
        extractor,
        window_planner: KnowledgeEvidenceWindowPlanner | None = None,
        frame_planner: KnowledgeFramePlanner | None = None,
    ) -> None:
        self._extractor = extractor
        self._window_planner = window_planner or KnowledgeEvidenceWindowPlanner()
        self._frame_planner = frame_planner or KnowledgeFramePlanner()

    def execute(self, context: PipelineContext) -> StageResult:
        transcript = context.artifacts.transcript
        media = context.artifacts.media
        if transcript is None or media is None:
            raise ValueError("knowledge-directed frame extraction requires media and transcript artifacts")
        drafts = list(context.state.get("claim_drafts") or ())
        windows = (
            self._window_planner.plan_claim_drafts(transcript, drafts, media_duration_ms=_duration_ms(context))
            if drafts
            else self._window_planner.plan(
                transcript, context.state.semantic_segments, media_duration_ms=_duration_ms(context)
            )
        )
        context.state.knowledge_evidence_windows = list(windows)
        draft_window_ids: dict[int, tuple[str, ...]] = {}
        for index, draft in enumerate(drafts):
            indices = tuple(sorted({int(value) for value in draft.evidence_segment_indices}))
            matching = [
                window
                for window in windows
                if window.semantic_segment_id == draft.semantic_segment_id
                and tuple(
                    item.segment_index
                    for item in transcript.segments
                    if item.segment_id in window.transcript_segment_ids
                )
                == indices
            ]
            if matching:
                draft_window_ids[index] = tuple(evidence_window_id(item) for item in matching)
        context.state.claim_evidence_window_ids = draft_window_ids
        requests = self._frame_planner.plan(windows, media_duration_ms=_duration_ms(context))
        if context.runtime.video_path is None:
            return StageResult(context=context)
        if not windows or not requests:
            raise ValueError("live media requires transcript-derived semantic evidence windows for visual extraction")
        existing_hashes = {
            str(item.get("image_hash") or "")
            for item in context.state.frames
            if isinstance(item, dict) and item.get("image_hash")
        }
        extracted = self._extractor.extract_targeted(
            context.runtime.video_path,
            context.runtime.work_dir,
            requests,
            existing_image_hashes=existing_hashes,
        )
        request_by_timestamp = {item.timestamp_ms: item for item in requests}
        artifacts: list[FrameArtifact] = []
        for item in extracted:
            timestamp_ms = int(item["timestamp_ms"])
            request = request_by_timestamp.get(timestamp_ms)
            if request is None:
                raise RuntimeError("knowledge frame extractor returned an unplanned timestamp")
            image_path = Path(str(item["image_path"]))
            digest = self._image_hash(item)
            durable_image = _persist_durable_file(context, image_path, digest, "frames")
            frame_id = frame_id_for(media_artifact_id=media.artifact_id, request=request)
            planned = {
                **item,
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "image_path": str(durable_image),
                "storage_ref": str(durable_image),
                "extraction_reason": request.extraction_reason,
                "semantic_segment_ids": list(request.semantic_segment_ids),
                "evidence_window_ids": list(request.evidence_window_ids),
                "planner_version": request.planner_version,
                "planner_request_id": request_id_for(request),
            }
            frame = FrameArtifact(
                artifact_id="frame-pending",
                artifact_type="frame",
                producer_stage=self.name,
                producer_version=request.planner_version,
                media_artifact_id=media.artifact_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                image_hash=digest,
                storage_ref=str(durable_image),
                extraction_reason=request.extraction_reason,
                semantic_segment_ids=request.semantic_segment_ids,
                evidence_window_ids=request.evidence_window_ids,
                planner_version=request.planner_version,
                planner_request_id=request_id_for(request),
                parent_artifact_ids=(media.artifact_id,),
            )
            artifact = FrameArtifact(**{**frame.__dict__, "artifact_id": artifact_id_of(frame)})
            context.artifacts.add("frames", artifact)
            artifacts.append(artifact)
            context.state.frames.append(planned)
        context.state.frames.sort(
            key=lambda item: (int(item.get("timestamp_ms") or 0), str(item.get("frame_id") or ""))
        )
        return StageResult(context=context, produced_artifacts=tuple(artifacts))

    @staticmethod
    def _image_hash(item: dict[str, Any]) -> str:
        return FrameExtractionStage._image_hash(item)


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
        eligible_ids = _eligible_visual_ids(context)
        frames = [item for item in context.artifacts.frames if item.frame_id in eligible_ids]
        ocr = [item for item in context.artifacts.ocr if item.frame_id in eligible_ids]
        vision = [item for item in context.artifacts.vision if item.frame_id in eligible_ids]
        contexts = [
            self._builder.build(
                segment,
                transcript,
                frames,
                ocr,
                vision,
                context.state.get("temporal_windows") or (),
            )
            for segment in context.state.get("semantic_segments") or ()
        ]
        context.state["semantic_contexts"] = contexts
        return _stage_result(context)


class ClaimVisualBindingStage:
    """Bind only cross-check-admitted targeted visual artifacts to each draft.

    The draft was validated before visual processing.  This stage cannot add a
    proposition or alter transcript coordinates; it simply records the exact
    Frame/OCR/Vision artifact evidence that supports or contradicts that
    already-grounded occurrence window.
    """

    name = "claim_visual_binding"
    required_inputs = ("semantic_segments", "transcript_visual_crosscheck")
    output_types = ()

    def execute(self, context: PipelineContext) -> PipelineContext:
        admitted = {
            str(item.get("frame_id") or "")
            for item in context.state.get("transcript_visual_crosschecks") or ()
            if item.get("relation") in {"SUPPORTS", "CONTRADICTS"}
        }
        frames = {item.frame_id: item for item in context.artifacts.frames if item.frame_id in admitted}
        displayed_secondary = {
            str(item.get("frame_id") or "")
            for item in context.state.get("transcript_visual_crosschecks") or ()
            if item.get("relation") == "SUPPORTS_DISPLAYED_SECONDARY"
        }
        # Keep this material separate from normally admitted multimodal
        # evidence.  It is evidence that a secondary page was displayed, not
        # independent confirmation of its policy or macro assertion.
        frames.update(
            {
                item.frame_id: item
                for item in context.artifacts.frames
                if item.frame_id in displayed_secondary
            }
        )
        ocr_by_frame: dict[str, list[OCRArtifact]] = {}
        for item in context.artifacts.ocr:
            if item.frame_id in frames and item.text:
                ocr_by_frame.setdefault(item.frame_id, []).append(item)
        vision_by_frame: dict[str, list[VisionArtifact]] = {}
        for item in context.artifacts.vision:
            if item.frame_id in frames and (item.label or item.labels):
                vision_by_frame.setdefault(item.frame_id, []).append(item)
        bound: list[ClaimOccurrenceDraft] = []
        for index, draft in enumerate(context.state.get("claim_drafts") or ()):
            window_ids = set(context.state.claim_evidence_window_ids.get(index, ()))
            permit_displayed_secondary = _is_attributed_displayed_secondary_report(draft)
            anchors: list[VisualEvidenceAnchor] = []
            for frame in sorted(frames.values(), key=lambda item: (item.timestamp_ms, item.frame_id)):
                if not window_ids.intersection(frame.evidence_window_ids):
                    continue
                if frame.frame_id in displayed_secondary and not permit_displayed_secondary:
                    continue
                ocr = next(iter(ocr_by_frame.get(frame.frame_id, ())), None)
                if ocr is not None:
                    anchors.append(
                        VisualEvidenceAnchor(
                            frame_id=frame.frame_id,
                            timestamp_ms=frame.timestamp_ms,
                            bbox=tuple(float(value) for value in (ocr.bbox or (0, 0, 0, 0))),
                            ocr_text=ocr.text,
                            model_id=ocr.engine,
                            model_version=ocr.engine_version,
                            confidence=float(ocr.confidence_score or 0.0),
                            support_type="OCR",
                        )
                    )
                    continue
                vision = next(iter(vision_by_frame.get(frame.frame_id, ())), None)
                if vision is not None:
                    anchors.append(
                        VisualEvidenceAnchor(
                            frame_id=frame.frame_id,
                            timestamp_ms=frame.timestamp_ms,
                            bbox=(0.0, 0.0, 0.0, 0.0),
                            visual_label=vision.label or vision.labels[0],
                            model_id=vision.model_name,
                            model_version=vision.model_version,
                            confidence=float(vision.confidence_score or 0.0),
                            support_type="LABEL",
                        )
                    )
            bound.append(draft.model_copy(update={"visual_anchors": anchors}))
        context.state.claim_drafts = bound
        return _stage_result(context)


def _is_attributed_displayed_secondary_report(draft: ClaimOccurrenceDraft) -> bool:
    """Permit a page-display citation without upgrading it into a fact.

    This is intentionally narrower than ``source_grade == SECONDARY``.  A
    speaker thesis or forecast may have a related page on screen (KU09/KU10),
    but that page is not evidence for the conclusion.  Only the two explicit
    Bundle-v2 report natures describe the proposition as *what a displayed
    secondary page says*.
    """
    semantic = dict(draft.bundle_v2 or {})
    if semantic.get("claim_nature") not in {
        "ATTRIBUTED_SECONDARY_POLICY_REPORT",
        "ATTRIBUTED_SECONDARY_MACRO_FACT_REPORT",
    } or semantic.get("source_grade") != "SECONDARY":
        return False
    attribution = dict(semantic.get("attribution") or {})
    return bool(attribution.get("attributed")) and "displayed" in str(
        attribution.get("source_label") or ""
    ).lower()


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
            sum(
                not any(item.semantic_segment_id == c.semantic_segment_id for item in drafts)
                for c in context.state.get("semantic_contexts") or ()
            )
        )
        segment_count = len(context.state.get("semantic_segments") or ())
        zero_claim_count = sum(
            not any(item.semantic_segment_id == segment.semantic_segment_id for item in drafts)
            for segment in context.state.get("semantic_segments") or ()
        )
        context.runtime.metrics["claims_per_semantic_segment"] = len(drafts) / max(1.0, float(segment_count))
        context.runtime.metrics["zero_claim_segment_ratio"] = zero_claim_count / max(1.0, float(segment_count))
        return _stage_result(context)


class AtomicClaimValidationStage:
    """Validate the structured atomic-claim seam before evidence grounding.

    Legacy ``ClaimOccurrenceDraft`` extraction remains an explicit compatibility
    path until the projection packet adopts AtomicClaimDraft as its input.  A
    structured model payload is never allowed to bypass the transcript-quality
    or semantic-coordinate gate.
    """

    name = "atomic_claim_validation"
    required_inputs = ("transcript", "semantic_segments")
    output_types = ()

    def __init__(self, validator: AtomicClaimDraftValidator | None = None) -> None:
        self._validator = validator or AtomicClaimDraftValidator()

    def execute(self, context: PipelineContext) -> PipelineContext:
        transcript = context.artifacts.transcript
        if transcript is None:
            raise ValueError("atomic claim validation requires transcript")
        # Canonical ingestion never accepts ``atomic_claim_payload`` as a
        # request option.  The compatibility seam is retained for isolated
        # stage tests, but production always validates the extractor's actual
        # ClaimOccurrenceDraft output.  In particular, an empty extractor
        # result is a valid, empty formal projection rather than a legacy
        # bypass.
        extracted = list(context.state.get("claim_drafts") or ())
        payload = (
            {"claims": [_extractor_draft_to_atomic_payload(item, transcript) for item in extracted]}
            if extracted
            else context.options.get("atomic_claim_payload")
        )
        if payload is None:
            context.state["validated_atomic_claims"] = []
            context.state["atomic_claim_rejections"] = []
            context.state["claim_drafts"] = []
            context.runtime.metrics["atomic_claim_accept_count"] = 0.0
            context.runtime.metrics["atomic_claim_reject_count"] = 0.0
            return _stage_result(context)
        report = context.state.transcript_quality_report
        result = self._validator.validate_payloads(
            payload,
            transcript,
            context.state.get("semantic_segments") or (),
            transcript_quality_status=(getattr(report, "quality_status", "NOT_PASS")),
        )
        context.state["validated_atomic_claims"] = list(result.accepted)
        context.state["atomic_claim_rejections"] = list(result.rejected)
        # An explicit structured payload is usable only after SC-07A accepts
        # it; replace any parallel raw draft list at this boundary.
        context.state["claim_drafts"] = [_accepted_atomic_to_claim_draft(item) for item in result.accepted]
        context.runtime.metrics["atomic_claim_accept_count"] = float(len(result.accepted))
        context.runtime.metrics["atomic_claim_reject_count"] = float(len(result.rejected))
        return _stage_result(context)


def _extractor_draft_to_atomic_payload(draft: ClaimOccurrenceDraft, transcript: TranscriptArtifact) -> dict[str, Any]:
    """Build validator input from an untrusted extractor DTO and authority text.

    This adapter deliberately has no path for extractor-supplied acceptance
    flags.  Coordinates, statement, subject and hard-fact candidates remain
    untrusted and the validator checks them against the selected transcript.
    When a legacy-shaped model response omitted a quote, the quote is a direct
    projection of its selected authority coordinates, not model-generated
    evidence or a new fact.
    """
    text_by_index = {
        item.segment_index: str(getattr(item, "raw_text", None) or getattr(item, "text", ""))
        for item in transcript.segments
    }
    evidence_indices = list(draft.evidence_segment_indices)
    authority_quote = " ".join(text_by_index[index] for index in evidence_indices if index in text_by_index)
    temporal = []
    for expression in draft.temporal_expressions:
        role = str(expression.role).upper()
        pit_meaning = {
            "REPORTING_PERIOD": "REPORTING_PERIOD",
            "FORECAST_TARGET": "FORECAST_TARGET",
        }.get(role, "UNKNOWN")
        temporal.append(
            {
                "raw_expression": expression.raw_expression,
                # ``scope_hint`` is not necessarily a textual period (for
                # example INTERVAL), so never claim it as one.
                "target_period": None,
                "pit_meaning": pit_meaning,
                "evidence_segment_indices": list(expression.evidence_segment_indices),
                "confidence": expression.confidence,
            }
        )
    value = draft.value
    object_value = None
    if value is not None:
        object_value = {
            "text": str(value) if isinstance(value, str) else "",
            "value": value if isinstance(value, (str, int, float)) else None,
            "unit": draft.unit,
            "currency": draft.currency,
        }
    return {
        "semantic_segment_id": draft.semantic_segment_id,
        "claim_type": draft.claim_type,
        "knowledge_kind": draft.knowledge_kind,
        "verbatim_quote": draft.verbatim_quote or authority_quote,
        "normalized_statement": atomic_statement(draft.normalized_statement or draft.conclusion),
        "subject": {
            "subject_type": draft.subject_type or "UNKNOWN",
            "subject_key": draft.subject_key,
            "subject_name": draft.subject_name,
        },
        "predicate": draft.predicate_key,
        "object": object_value,
        "condition_text": draft.condition_text,
        "invalidation_text": draft.invalidation_text,
        "sentiment": draft.sentiment,
        # These are not persisted model assertions in ClaimOccurrenceDraft.
        # The validator independently checks statement polarity/tense against
        # the evidence; leaving tense UNKNOWN avoids inventing a temporal fact.
        "polarity": "ASSERTS",
        "assertion_tense": "UNKNOWN",
        "evidence_segment_indices": evidence_indices,
        "condition_evidence_segment_indices": list(draft.condition_evidence_segment_indices),
        "invalidation_evidence_segment_indices": list(draft.invalidation_evidence_segment_indices),
        "temporal_expressions": temporal,
        "visual_anchors": [item.model_dump(mode="json") for item in draft.visual_anchors],
        "bundle_v2": dict(draft.bundle_v2),
        "extraction_confidence": draft.extraction_confidence,
    }


def _accepted_atomic_to_claim_draft(item) -> ClaimOccurrenceDraft:
    """Adapt only a validator-accepted atomic draft into the legacy stage DTO."""
    draft = item.draft
    return ClaimOccurrenceDraft(
        semantic_segment_id=draft.semantic_segment_id,
        knowledge_kind=draft.knowledge_kind,
        claim_type=draft.claim_type,
        subject_type=draft.subject.subject_type,
        subject_key=draft.subject.subject_key or (draft.subject.subject_name or ""),
        subject_name=draft.subject.subject_name,
        predicate_key=draft.predicate,
        conclusion=atomic_statement(draft.normalized_statement),
        value=(
            draft.object.value
            if draft.object and draft.object.value is not None
            else (draft.object.text if draft.object else draft.normalized_statement)
        ),
        unit=draft.object.unit if draft.object else None,
        currency=draft.object.currency if draft.object else None,
        sentiment=draft.sentiment,
        condition_text=draft.condition_text,
        invalidation_text=draft.invalidation_text,
        evidence_segment_indices=list(draft.evidence_segment_indices),
        condition_evidence_segment_indices=list(draft.condition_evidence_segment_indices),
        invalidation_evidence_segment_indices=list(draft.invalidation_evidence_segment_indices),
        temporal_expressions=[
            TemporalExpressionDraft(
                role=(
                    "REPORTING_PERIOD"
                    if expression.pit_meaning == "REPORTING_PERIOD"
                    else "FORECAST_TARGET"
                    if expression.pit_meaning == "FORECAST_TARGET"
                    else "VALID_AT"
                ),
                raw_expression=expression.raw_expression,
                scope_hint=None,
                evidence_segment_indices=list(expression.evidence_segment_indices),
                confidence=expression.confidence,
            )
            for expression in draft.temporal_expressions
        ],
        extraction_confidence=draft.extraction_confidence,
        extraction_model_id="atomic-claim-validator",
        extraction_prompt_version="atomic-claim-validator.v1",
        visual_anchors=list(draft.visual_anchors),
        bundle_v2=dict(draft.bundle_v2),
        verbatim_quote=draft.verbatim_quote,
        normalized_statement=draft.normalized_statement,
        grounding_status="GROUNDED",
        grounding_reason_codes=[],
        contradiction_group_id=item.contradiction_group_id,
        claim_schema_version="claim.atomic.v1",
        legacy_grounding_incomplete=False,
    )


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
        enriched = []
        for item in grounded:
            visual_items = _occurrence_visual_evidence(context, item.draft)
            if visual_items:
                item = replace(
                    item,
                    evidences=tuple([*item.evidences, *visual_items]),
                    secondary_evidence_refs=tuple(entry.evidence_id for entry in visual_items),
                )
            enriched.append(item)
        grounded = enriched
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
                item.artifact_id for item in (context.artifacts.semantic_segments, transcript) if item is not None
            ),
        )
        context.artifacts.evidence = EvidenceArtifact(
            **{**evidence_artifact.__dict__, "artifact_id": artifact_id_of(evidence_artifact)}
        )
        context.state["grounded_occurrences"] = grounded
        context.state.evidence = list(unique.values())
        context.runtime.metrics["grounding_reject_count"] = 0.0
        return _stage_result(context, "evidence")


def _evidence_item_for_visual(*, artifact, source_type: str, frame_id: str, timestamp_ms: int, content: str, bbox):
    locator = {"frame_id": frame_id, "timestamp_ms": timestamp_ms, "bbox": list(bbox) if bbox else None}
    evidence_id = (
        "ev_"
        + hashlib.sha256(
            canonical_json(
                {"source_artifact_id": artifact.artifact_id, "locator": locator, "content": content}
            ).encode()
        ).hexdigest()
    )
    return EvidenceItem(
        evidence_id=evidence_id,
        source_type=source_type,
        source_artifact_id=artifact.artifact_id,
        evidence_text=content,
        raw_text=content,
        normalized_text=content,
        start_ms=timestamp_ms,
        end_ms=timestamp_ms,
        confidence_score=getattr(artifact, "confidence_score", None),
        locator=locator,
    )


def _occurrence_visual_evidence(context: PipelineContext, draft: ClaimOccurrenceDraft) -> list[EvidenceItem]:
    """Materialise only model-selected, crosscheck-admitted visual evidence.

    Evidence IDs are occurrence-owned through the later SECONDARY relation;
    a frame/OCR/Vision artifact never becomes a transcript substitute.
    """
    if not draft.visual_anchors:
        return []
    eligible = _eligible_visual_ids(context)
    frames = {item.frame_id: item for item in context.artifacts.frames if item.frame_id in eligible}
    ocr_by_frame: dict[str, list[Any]] = {}
    vision_by_frame: dict[str, list[Any]] = {}
    for item in context.artifacts.ocr:
        ocr_by_frame.setdefault(item.frame_id, []).append(item)
    for item in context.artifacts.vision:
        vision_by_frame.setdefault(item.frame_id, []).append(item)
    selected: dict[str, EvidenceItem] = {}
    for anchor in draft.visual_anchors:
        frame = frames.get(anchor.frame_id)
        if frame is None:
            raise ValueError("VISUAL_ANCHOR_ARTIFACT_MISSING_OR_NOT_ADMITTED")
        timestamp = int(anchor.timestamp_ms)
        selected_item = _evidence_item_for_visual(
            artifact=frame,
            source_type="FRAME",
            frame_id=frame.frame_id,
            timestamp_ms=timestamp,
            content=f"frame:{frame.frame_id}",
            bbox=anchor.bbox,
        )
        selected[selected_item.evidence_id] = selected_item
        if anchor.support_type == "OCR":
            matches = [
                item
                for item in ocr_by_frame.get(anchor.frame_id, [])
                if item.engine == anchor.model_id
                and item.engine_version == anchor.model_version
                and (not anchor.ocr_text or anchor.ocr_text in item.text)
            ]
            if not matches:
                raise ValueError("VISUAL_OCR_ANCHOR_NOT_BACKED_BY_ARTIFACT")
            for item in matches:
                entry = _evidence_item_for_visual(
                    artifact=item,
                    source_type="OCR",
                    frame_id=item.frame_id,
                    timestamp_ms=item.timestamp_ms,
                    content=item.text,
                    bbox=anchor.bbox,
                )
                selected[entry.evidence_id] = entry
        else:
            matches = [
                item
                for item in vision_by_frame.get(anchor.frame_id, [])
                if item.model_name == anchor.model_id
                and item.model_version == anchor.model_version
                and (not anchor.visual_label or anchor.visual_label in {item.label, *item.labels})
            ]
            if not matches:
                raise ValueError("VISUAL_VISION_ANCHOR_NOT_BACKED_BY_ARTIFACT")
            for item in matches:
                content = item.label or " ".join(item.labels)
                entry = _evidence_item_for_visual(
                    artifact=item,
                    source_type="VISION",
                    frame_id=item.frame_id,
                    timestamp_ms=item.timestamp_ms,
                    content=content,
                    bbox=anchor.bbox,
                )
                selected[entry.evidence_id] = entry
    return list(selected.values())


def _occurrence_review_reason_codes(context: PipelineContext, draft: ClaimOccurrenceDraft) -> list[str]:
    reasons: set[str] = set()
    checks = list(context.state.get("transcript_visual_crosschecks") or ())
    if not checks and context.artifacts.transcript_visual_crosscheck is not None:
        checks = [
            {
                "relation": item.relation,
                "semantic_segment_ids": list(item.semantic_segment_ids),
                "mismatches": dict(item.mismatches),
            }
            for item in context.artifacts.transcript_visual_crosscheck.relations
        ]
    for check in checks:
        if not isinstance(check, dict) or check.get("relation") != "CONTRADICTS":
            continue
        if draft.semantic_segment_id not in set(check.get("semantic_segment_ids") or []):
            continue
        mismatches = dict(check.get("mismatches") or {})
        if "NUMBER" in mismatches:
            reasons.add("ASR_OCR_NUMERIC_CONFLICT")
        else:
            reasons.add("ASR_VISUAL_CONFLICT")
    return sorted(reasons)


class TemporalNormalizationStage:
    name = "temporal_normalization"
    required_inputs = ("evidence",)
    output_types = ()

    def __init__(
        self,
        normalizer: TemporalNormalizer | None = None,
        normalization_version: str = "temporal-normalization.final.v1",
        reference_provider: Any | None = None,
    ) -> None:
        self._normalizer = normalizer or TemporalNormalizer(
            reference_provider=reference_provider, normalization_version=normalization_version
        )
        self._normalization_version = normalization_version
        self._reference_provider = reference_provider

    def execute(self, context: PipelineContext) -> PipelineContext:
        # Replay can provide a snapshot-pinned provider per context.  This is
        # intentionally an explicit adapter option; the production stage's
        # default provider remains unchanged for offline fixtures.
        normalizer = self._normalizer
        context_provider = context.options.get("temporal_reference_provider")
        if context_provider is not None and context_provider is not getattr(normalizer, "reference_provider", None):
            normalizer = TemporalNormalizer(
                reference_provider=context_provider,
                normalization_version=getattr(normalizer, "normalization_version", self._normalization_version),
            )
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
                                raise ValueError(f"unknown temporal anchor: {expression.anchor}") from exc
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
                        marker in expression_text for marker in ("期末", "余额", "截至", "END", "ENDING", "AS OF")
                    ):
                        metric_nature = MetricTemporalNature.INSTANT
                    elif (
                        scope_hint is TemporalScope.INTERVAL
                        or expression.scope_hint == "INTERVAL"
                        or any(marker in expression_text for marker in ("Q", "季度", "FY", "年", "月"))
                    ):
                        metric_nature = MetricTemporalNature.DURATION
                elif draft.claim_type in {"PRICE", "VALUATION"} and scope_hint in {None, TemporalScope.POINT}:
                    metric_nature = MetricTemporalNature.SNAPSHOT
                draft_bindings.append(
                    normalizer.normalize(
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
        # If a caller supplies the immutable snapshot candidate explicitly,
        # references published after it are not visible and must fail closed.
        candidate = context.options.get("snapshot_commit_candidate") or context.options.get(
            "normalization_available_at"
        )
        if isinstance(candidate, str):
            candidate = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        if candidate is not None:
            for binding in (item for values in bindings_by_draft.values() for item in values):
                available = getattr(binding, "reference_available_at", None)
                if available is not None and available > candidate:
                    raise ValueError("REFERENCE_AS_OF_VIOLATION: reference available_at is after snapshot candidate")
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
        context.runtime.metrics.update(
            {
                "temporal_binding_count": float(binding_count),
                "temporal_normalization_success_rate": sum(
                    item.normalization_status == "NORMALIZED" for item in bindings
                )
                / max(1.0, float(binding_count)),
                "temporal_normalization_partial_rate": sum(item.normalization_status == "PARTIAL" for item in bindings)
                / max(1.0, float(binding_count)),
                "temporal_normalization_unresolved_rate": sum(
                    item.normalization_status == "UNRESOLVED" for item in bindings
                )
                / max(1.0, float(binding_count)),
                "temporal_partial_rate": sum(item.normalization_status == "PARTIAL" for item in bindings)
                / max(1.0, float(binding_count)),
                "temporal_unresolved_rate": sum(item.normalization_status == "UNRESOLVED" for item in bindings)
                / max(1.0, float(binding_count)),
                "temporal_role_distribution": {
                    str(getattr(item.role, "value", item.role)): sum(
                        getattr(other.role, "value", other.role) == getattr(item.role, "value", item.role)
                        for other in bindings
                    )
                    for item in sorted(bindings, key=lambda value: str(getattr(value.role, "value", value.role)))
                },
            }
        )
        unresolved = [item.expression_key for item in bindings if item.normalization_status == "UNRESOLVED"]
        context.runtime.metrics["unresolved_expression_collision_rate"] = (
            len(unresolved) - len(set(unresolved))
        ) / max(1.0, float(len(unresolved)))
        forecast_drafts = [item for item in context.state.get("claim_drafts") or () if item.claim_type == "FORECAST"]
        forecast_with_target = {
            index
            for index, values in bindings_by_draft.items()
            if any(getattr(binding, "role", None) is TemporalRole.FORECAST_TARGET for binding in values)
        }
        context.runtime.metrics["forecast_target_missing_rate"] = sum(
            index not in forecast_with_target
            for index, item in enumerate(context.state.get("claim_drafts") or ())
            if item.claim_type == "FORECAST"
        ) / max(1.0, float(len(forecast_drafts)))
        fiscal = [item for item in bindings if getattr(item.calendar_type, "value", item.calendar_type) == "FISCAL"]
        context.runtime.metrics["fiscal_period_unresolved_rate"] = sum(
            item.normalization_status in {"PARTIAL", "UNRESOLVED"} for item in fiscal
        ) / max(1.0, float(len(fiscal)))
        market = [
            item
            for item in bindings
            if item.market_session or getattr(item.calendar_type, "value", item.calendar_type) == "EXCHANGE"
        ]
        context.runtime.metrics["market_session_unresolved_rate"] = sum(
            not item.market_session for item in market
        ) / max(1.0, float(len(market)))
        metric_drafts = [
            item for item in context.state.get("claim_drafts") or () if item.claim_type == "FINANCIAL_METRIC"
        ]
        metric_binding_items = [
            binding
            for index, values in bindings_by_draft.items()
            if index < len(context.state.get("claim_drafts") or ())
            and context.state["claim_drafts"][index].claim_type == "FINANCIAL_METRIC"
            for binding in values
        ]
        context.runtime.metrics["metric_temporal_nature_unknown_rate"] = sum(
            getattr(item.metric_temporal_nature, "value", item.metric_temporal_nature) in {None, "UNKNOWN"}
            for item in metric_binding_items
        ) / max(1.0, float(len(metric_binding_items) or len(metric_drafts)))
        planned = sum(getattr(item.assertion_status, "value", item.assertion_status) == "PLANNED" for item in bindings)
        actual = sum(getattr(item.assertion_status, "value", item.assertion_status) == "ACTUAL" for item in bindings)
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
                if configured_normalization_version
                else None,
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
    value = (
        None
        if replay_without_explicit_clock
        else (
            context.options.get("snapshot_commit_candidate")
            or context.options.get("available_from")
            or context.options.get("as_of")
        )
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
            context.options.get("offline_fixture") or "transcript" in context.options or "segments" in context.options
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
            review_codes = _occurrence_review_reason_codes(context, draft)
            semantic_envelope = bundle_v2_semantics(
                statement=claim.normalized_statement or draft.conclusion,
                claim_type=claim.claim_type,
                supplied={**dict(claim.bundle_v2), **dict(draft.bundle_v2)},
                temporal_expressions=[item.model_dump(mode="json") for item in draft.temporal_expressions],
                review_reason_codes=review_codes,
            )
            occurrences.append(
                ClaimOccurrence(
                    claim_id=claim.claim_id,
                    source_artifact_id=(
                        context.artifacts.source.artifact_id if context.artifacts.source else transcript.artifact_id
                    ),
                    transcript_artifact_id=transcript.artifact_id,
                    semantic_segment_id=draft.semantic_segment_id,
                    evidence_refs=refs,
                    secondary_evidence_refs=list(relation.secondary_evidence_refs),
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
                            "grounded_evidence_refs": list(
                                next(
                                    (
                                        binding.source_evidence_refs
                                        for binding in context.state.get("temporal_bindings_by_draft", {}).get(
                                            draft_index, []
                                        )
                                        if getattr(binding, "raw_expression", None) == expression.raw_expression
                                    ),
                                    [],
                                )
                            ),
                        }
                        for expression in draft.temporal_expressions
                    ],
                    provenance={
                        "model_id": draft.extraction_model_id,
                        "prompt_version": draft.extraction_prompt_version,
                        "bundle_v2": semantic_envelope,
                    },
                    primary_quote=draft.verbatim_quote,
                    normalized_statement=draft.normalized_statement,
                    grounding_status=draft.grounding_status,
                    grounding_reason_codes=list(draft.grounding_reason_codes),
                    contradiction_group_id=draft.contradiction_group_id,
                    claim_schema_version=draft.claim_schema_version,
                    legacy_grounding_incomplete=draft.legacy_grounding_incomplete,
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
            context.options.get("offline_fixture") or "transcript" in context.options or "segments" in context.options
        )
        # A fixture without an explicit business clock must be replayable.  The
        # download compatibility adapter may expose a wall-clock source
        # availability timestamp for PUBLIC_STRICT search, but that timestamp
        # is not a lifecycle business clock.  Use the immutable transcript
        # boundary for both the initial run and replay.
        if fixture_clock and not any(
            context.options.get(key) for key in ("as_of", "available_from", "replay_lifecycle_timestamp")
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
        # A transcript/OCR contradiction is occurrence-local and does not
        # mean that the original evidence should be discarded.  It does mean
        # that publishing the projection as ACTIVE would overstate what the
        # pipeline knows before a person has reviewed it.  Keep the complete
        # evidence and semantic envelope, but make both the occurrence ledger
        # and its knowledge projection explicitly pre-publication.
        review_by_occurrence = {
            item.occurrence_id: _occurrence_review(item)
            for item in occurrences
        }
        review_required_ids = {
            occurrence_id
            for occurrence_id, review in review_by_occurrence.items()
            if review["status"] == "HUMAN_REVIEW_REQUIRED"
        }
        review_required_claim_ids = {
            item.claim_id for item in occurrences if item.occurrence_id in review_required_ids
        }
        events = []
        for target_type, target_id, review_required in [
            *(("CLAIM", item.claim_id, item.claim_id in review_required_claim_ids) for item in claims),
            *(("OCCURRENCE", item.occurrence_id, item.occurrence_id in review_required_ids) for item in occurrences),
        ]:
            events.append(
                KnowledgeLifecycleEvent(
                    target_type=target_type,
                    target_id=target_id,
                    to_status="EXTRACTED" if review_required else "ACTIVE",
                    effective_at=timestamp,
                    recorded_at=timestamp,
                    reason_code=(
                        "INITIAL_EXTRACTION_HUMAN_REVIEW_REQUIRED"
                        if review_required else "INITIAL_EXTRACTION"
                    ),
                    policy_version="lifecycle.v1",
                )
            )
        occurrence_artifact = context.artifacts.occurrences
        lifecycle_artifact = LifecycleArtifact(
            artifact_id="lifecycle-pending",
            artifact_type="lifecycle",
            producer_stage=self.name,
            claim_lifecycle_event_ids=[item.lifecycle_event_id for item in events if item.target_type.value == "CLAIM"],
            occurrence_lifecycle_event_ids=[
                item.lifecycle_event_id for item in events if item.target_type.value == "OCCURRENCE"
            ],
            lifecycle_business_as_of=timestamp,
            lifecycle_knowledge_as_of=timestamp,
            policy_version="lifecycle.v1",
            parent_artifact_ids=tuple(
                item.artifact_id for item in (occurrence_artifact, context.artifacts.verification) if item is not None
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
            occurrence_id = str(attributes.get("occurrence_id") or "")
            review = review_by_occurrence.get(occurrence_id)
            review_required = occurrence_id in review_required_ids
            lifecycle_status = "EXTRACTED" if review_required else "ACTIVE"
            # ReviewStatus has no "required" member: UNREVIEWED is the
            # truthful Axis-3 core value until a human decision is recorded.
            # A contradictory item remains source-located, rather than being
            # represented as source-supported before that review.
            if review_required:
                unit.support_status = "SOURCE_LOCATED"
                unit.review_status = "UNREVIEWED"
            unit.lifecycle_status = lifecycle_status
            attributes["lifecycle_status"] = lifecycle_status
            attributes["support_status"] = unit.support_status
            attributes["review_status"] = unit.review_status
            if review is not None:
                attributes["occurrence_review"] = review
            attributes["lifecycle_artifact_id"] = lifecycle_id
            unit.attributes = attributes
        if context.artifacts.knowledge is not None:
            knowledge_artifact = KnowledgeArtifact(
                artifact_id="knowledge-final-pending",
                artifact_type="knowledge",
                producer_stage="knowledge_projection",
                verification_artifact_id=(
                    context.artifacts.verification.artifact_id if context.artifacts.verification else ""
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
        correction_count = float(
            sum(str(item.reason_code).upper() in {"CORRECTION", "CORRECTED", "REVISED"} for item in events)
        )
        context.runtime.metrics["correction_count"] = correction_count
        context.runtime.metrics["lifecycle_correction_count"] = correction_count
        return _stage_result(context, "lifecycle", "knowledge")


def _occurrence_review(occurrence: ClaimOccurrence) -> dict[str, Any]:
    """Read the immutable occurrence-scoped manual-review boundary.

    The review envelope is intentionally not promoted to ``ReviewStatus``:
    ``HUMAN_REVIEW_REQUIRED`` is a request for a future decision, whereas
    ``ReviewStatus`` records a completed human decision.  Keeping that
    distinction prevents an extraction conflict from masquerading as either
    approval or rejection.
    """
    provenance = dict(occurrence.provenance or {})
    envelope = dict(provenance.get("bundle_v2") or {})
    review = dict(envelope.get("occurrence_review") or {})
    reasons = sorted({str(item) for item in review.get("reason_codes") or [] if str(item)})
    return {
        "status": "HUMAN_REVIEW_REQUIRED" if review.get("status") == "HUMAN_REVIEW_REQUIRED" else "NOT_REQUIRED",
        "reason_codes": reasons,
    }


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
        eligible_ids = _eligible_visual_ids(context)
        eligible_ocr = [
            item for item in context.state.get("ocr_evidence") or () if str(item.get("frame_id") or "") in eligible_ids
        ]
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
                "ocr_blocks": eligible_ocr,
                "frame_refs": list(context.state.get("eligible_frame_insights") or []),
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
            ticker = normalized_ticker(record.get("ticker"))
            if not entities and ticker:
                entities.append(
                    {
                        "entity_name": record.get("subject_name") or ticker,
                        "entity_key": ticker,
                        "ticker": ticker,
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
                    ticker=ticker,
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
                        if any(term in unit.statement for term in ("营收", "收入", "利润", "业绩", "毛利率"))
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
                evidence_id = (
                    "evidence-"
                    + hashlib.sha256(
                        canonical_json({"source": transcript.artifact_id, "locator": locator, "raw": raw}).encode()
                    ).hexdigest()[:32]
                )
                evidence_items.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        source_type=segment.source,
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
        eligible_ids = _eligible_visual_ids(context)
        ocr_sources: dict[tuple[str, str], list[str]] = {}
        for artifact in context.artifacts.ocr:
            if artifact.frame_id not in eligible_ids:
                continue
            frame_id = next(
                (
                    frame.frame_id
                    for frame in context.artifacts.frames
                    if frame.artifact_id == artifact.frame_artifact_id
                ),
                "",
            )
            ocr_sources.setdefault((frame_id, artifact.text), []).append(artifact.artifact_id)
        for item in context.state.ocr_evidence:
            if str(item.get("frame_id") or "") not in eligible_ids:
                continue
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
            evidence_id = (
                "evidence-"
                + hashlib.sha256(
                    canonical_json({"source": source, "locator": locator, "raw": raw}).encode()
                ).hexdigest()[:32]
            )
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
            if artifact.frame_id not in eligible_ids:
                continue
            frame_id = next(
                (
                    frame.frame_id
                    for frame in context.artifacts.frames
                    if frame.artifact_id == artifact.frame_artifact_id
                ),
                "",
            )
            vision_sources.setdefault((frame_id, artifact.label), []).append(artifact.artifact_id)
        for item in context.state.eligible_frame_insights:
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
            evidence_id = (
                "evidence-"
                + hashlib.sha256(
                    canonical_json({"source": source, "locator": locator, "raw": raw}).encode()
                ).hexdigest()[:32]
            )
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
                + [item.artifact_id for item in context.artifacts.ocr if item.frame_id in eligible_ids]
                + [item.artifact_id for item in context.artifacts.vision if item.frame_id in eligible_ids],
            )
        )
        evidence_parent_ids = tuple(
            dict.fromkeys(
                [
                    *source_ids,
                    context.artifacts.semantic_segments.artifact_id if context.artifacts.semantic_segments else "",
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
        context.artifacts.evidence = EvidenceArtifact(**{**evidence.__dict__, "artifact_id": artifact_id_of(evidence)})
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
                        if item.evidence_text == text and (not source_type or item.source_type == source_type)
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
            ticker = normalized_ticker(record.get("ticker"))
            claims.append(
                FinancialClaim(
                    claim_type=claim_type,
                    subject_type="EQUITY" if ticker else "CONTENT",
                    subject_id=str(record.get("subject_key") or record.get("knowledge_uid")),
                    ticker=ticker,
                    predicate=str(record.get("predicate_key") or "statement"),
                    value=(
                        record.get("value") if record.get("value") is not None else str(record.get("statement") or "")
                    ),
                    unit=record.get("unit"),
                    currency=record.get("currency"),
                    fact_time=record.get("as_of_time"),
                    published_at=(context.state.video.published_at if context.state.video else None),
                    evidence_refs=claim_refs,
                    source_support_status=support_status(record.get("support_status")),
                    source_confidence=float(record.get("support_score") or 0.75),
                    extractor_confidence=float(record.get("extraction_confidence") or 0.75),
                    extraction_model_id=str((record.get("attributes") or {}).get("model") or "fixture"),
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
                status=("VERIFICATION_PENDING" if claim.fact_category == "FACT" else "NOT_REQUIRED"),
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
                occurrence_lifecycle_event_ids=list(context.artifacts.lifecycle.occurrence_lifecycle_event_ids or ()),
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
            attributes.update(
                {
                    "claim_id": claim.claim_id,
                    "occurrence_id": occurrence.occurrence_id,
                    "semantic_segment_id": occurrence.semantic_segment_id,
                    "asserted_at": occurrence.times.asserted_at.isoformat() if occurrence.times.asserted_at else None,
                    "source_published_at": occurrence.times.source_published_at.isoformat()
                    if occurrence.times.source_published_at
                    else None,
                    "source_available_at": occurrence.times.source_available_at.isoformat()
                    if occurrence.times.source_available_at
                    else None,
                    "source_availability_quality": occurrence.times.source_availability_quality.value,
                    "ingested_at": occurrence.times.ingested_at.isoformat(),
                    "extraction_completed_at": occurrence.times.extraction_completed_at.isoformat(),
                    "available_from": occurrence.times.available_from.isoformat(),
                }
            )
            record["attributes"] = attributes
            record["_claim_id"] = claim.claim_id
        return True

    def execute(self, context: PipelineContext) -> PipelineContext:
        available_from = self._timestamp(context.options.get("available_from") or context.options.get("as_of"))
        fixture = bool(
            context.options.get("offline_fixture") or "transcript" in context.options or "segments" in context.options
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
                    status=("VERIFICATION_PENDING" if claim.fact_category == "FACT" else "NOT_REQUIRED"),
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
            records = self._cross_modal.verify_many(
                records,
                [
                    item
                    for item in context.state.get("ocr_evidence") or ()
                    if str(item.get("frame_id") or "") in _eligible_visual_ids(context)
                ],
            )
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
                    context.artifacts.semantic_segments.artifact_id if context.artifacts.semantic_segments else ""
                ),
                evidence_artifact_id=(context.artifacts.evidence.artifact_id if context.artifacts.evidence else ""),
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
                evidence_artifact_id=(context.artifacts.evidence.artifact_id if context.artifacts.evidence else ""),
                claims=[claim.claim_id for claim in context.state.claims],
                parent_artifact_ids=(context.artifacts.occurrences.artifact_id,),
            )
            context.artifacts.claims = ClaimArtifact(
                **{**claim_artifact.__dict__, "artifact_id": artifact_id_of(claim_artifact)}
            )
        context.state["knowledge"] = self._to_domain(context.state["video"].video_id, records, available_from)
        claims_by_id = {claim.claim_id: claim for claim in context.state.claims}
        claims_by_uid = {
            str(record.get("knowledge_uid")): (claims_by_id.get(str(record.get("_claim_id"))) or claim)
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
                context.artifacts.verification.artifact_id if context.artifacts.verification else ""
            ),
            knowledge_units=[unit.knowledge_uid for unit in knowledge_units],
            parent_artifact_ids=(
                (context.artifacts.verification.artifact_id,) if context.artifacts.verification else ()
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
                projected_attributes.update(
                    {
                        "source_available_at": (
                            occurrence.times.source_available_at.isoformat()
                            if occurrence.times.source_available_at
                            else None
                        ),
                        "source_availability_quality": occurrence.times.source_availability_quality.value,
                        "ingested_at": occurrence.times.ingested_at.isoformat(),
                        "extraction_completed_at": occurrence.times.extraction_completed_at.isoformat(),
                        "available_from": occurrence.times.available_from.isoformat(),
                    }
                )
            records.append(
                {
                    "knowledge_uid": projection["knowledge_uid"],
                    "statement": projection["statement"],
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
                }
            )
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
        "source",
        "media",
        "transcript",
        "semantic_segments",
        "evidence",
        "claims",
        "occurrences",
        "verification",
        "lifecycle",
        "knowledge",
        "summary",
    )
    output_types = ()

    def __init__(
        self,
        snapshot_service: SnapshotService,
        artifact_repository=None,
        occurrence_repository=None,
        lifecycle_repository=None,
        verification_repository=None,
        verification_job_repository=None,
        claim_event_repository=None,
        signal_service=None,
        publication_uow=None,
        fenced_effects=None,
    ) -> None:
        self._snapshots = snapshot_service
        self._artifact_repository = artifact_repository
        self._occurrence_repository = occurrence_repository
        self._lifecycle_repository = lifecycle_repository
        self._verification_repository = verification_repository or verification_job_repository
        self._verification_jobs = verification_job_repository or verification_repository
        self._claim_events = claim_event_repository
        self._signal_service = signal_service
        self._publication_uow = publication_uow
        self._fenced_effects = fenced_effects

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Plan and publish under one repository-owned verification UoW."""
        return self._execute_with_planning_uow(context)

    def _execute_with_planning_uow(self, context: PipelineContext) -> PipelineContext:
        keys = [
            (claim.claim_id, str(context.options.get("verification_provider") or "quant"))
            for claim in (context.state.get("claims") or ())
        ]
        planner_uow = getattr(self._verification_jobs, "planning_uow", None)
        if planner_uow is not None and keys:
            scope = planner_uow(keys)
        elif self._fenced_effects is not None and context.worker_id and context.fencing_token is not None:
            # An empty claim set has no verification-planner lock to borrow.
            # Use the same task-fenced UoW rather than passing ``None`` to a
            # production snapshot effect.
            scope = self._fenced_effects.fenced_transaction(
                context.task_id,
                context.worker_id,
                context.fencing_token,
            )
        else:
            scope = nullcontext(None)
        with scope as session:
            return self._execute_in_uow(context, session=session)

    def _execute_in_uow(self, context: PipelineContext, *, session=None) -> PipelineContext:
        self._require_effect_fence(context, session)
        registry = context.artifacts
        mandatory = {
            "source": registry.source,
            "media": registry.media,
            "transcript": registry.transcript,
            "evidence": registry.evidence,
            "claims": registry.claims,
            "verification": registry.verification,
            "knowledge": registry.knowledge,
            "summary": registry.summary,
        }
        semantic_enabled = bool(
            (context.options.get("pipeline_config") or {}).get("semantic_segmentation_enabled", True)
        )
        if semantic_enabled:
            mandatory.update(
                {
                    "semantic_segments": registry.semantic_segments,
                    "occurrences": registry.occurrences,
                    "lifecycle": registry.lifecycle,
                }
            )
        missing = [slot for slot, artifact in mandatory.items() if artifact is None]
        if missing:
            raise ContentSnapshotPersistError(
                f"CONTENT_SNAPSHOT_PERSIST_FAILED: mandatory artifact missing: {sorted(missing)}"
            )
        verification_plan = None
        # Resolve initial verification immediately before publication.  This
        # keeps the PIT candidate equal to the immutable snapshot clock and
        # ensures the exact job/result lineage is what gets committed.
        snapshot_candidate = (
            context.state.occurrences[0].times.snapshot_committed_at
            if context.state.get("occurrences")
            else _stage_timestamp(context)
        )
        for binding in context.state.get("temporal_bindings") or ():
            available_at = getattr(binding, "reference_available_at", None)
            if available_at is None:
                continue
            comparable_candidate = snapshot_candidate
            if comparable_candidate.tzinfo is None and available_at.tzinfo is not None:
                comparable_candidate = comparable_candidate.replace(tzinfo=available_at.tzinfo)
            if available_at > comparable_candidate:
                raise ContentSnapshotPersistError(
                    "REFERENCE_AS_OF_VIOLATION: reference available_at is after snapshot candidate"
                )
        if self._verification_jobs is not None and context.state.get("claims"):
            verification_plan = build_initial_verification_plan(
                claims=list(context.state.claims),
                provider=str(context.options.get("verification_provider") or "quant"),
                snapshot_candidate_time=snapshot_candidate,
                verification_repository=self._verification_repository or self._verification_jobs,
                job_repository=self._verification_jobs,
                current_policy_version=str(context.options.get("verification_rule_version") or "verification_rule.v1"),
                trace_id=context.trace.get("trace_id"),
                session=session,
            )
            self._replace_verification_lineage(context, verification_plan.artifact_results)
        if self._artifact_repository is not None:
            for artifact in registry.artifacts():
                if session is not None and hasattr(self._artifact_repository, "put_in_session"):
                    self._artifact_repository.put_in_session(session, artifact)
                else:
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
            drafts = list(context.state.get("claim_drafts") or ())
            bindings_by_draft = context.state.get("temporal_bindings_by_draft") or {}
            for draft_index, bindings in sorted(bindings_by_draft.items(), key=lambda item: int(item[0])):
                draft = drafts[int(draft_index)] if int(draft_index) < len(drafts) else None
                subject_key = str(getattr(draft, "subject_key", "") or "")
                for binding in bindings:
                    snapshot_id = getattr(binding, "reference_snapshot_id", None)
                    if not snapshot_id:
                        continue
                    data_version = getattr(binding, "reference_data_version", None)
                    available_at = getattr(binding, "reference_available_at", None)
                    if not data_version or available_at is None:
                        raise ContentSnapshotPersistError(
                            "REFERENCE_SNAPSHOT_METADATA_MISSING: immutable reference requires version and available_at"
                        )
                    calendar = str(getattr(getattr(binding, "calendar_type", None), "value", "") or "").upper()
                    has_resolved_period = (
                        getattr(binding, "start_date", None) is not None
                        and getattr(binding, "end_date", None) is not None
                    )
                    reference_type = (
                        "exchange_calendar"
                        if calendar == "EXCHANGE"
                        else ("fiscal_period" if has_resolved_period else "fiscal_calendar")
                    )
                    period_label = str(getattr(binding, "period_label", "") or "")
                    binding_key = f"{reference_type}|{subject_key}|{period_label}"
                    reference_snapshot_ids.add(str(snapshot_id))
                    reference_records.append(
                        {
                            "reference_type": reference_type,
                            "subject_key": subject_key,
                            "period_label": period_label,
                            "binding_key": binding_key,
                            "reference_snapshot_id": str(snapshot_id),
                            "data_version": str(data_version),
                            "available_at": available_at.isoformat(),
                        }
                    )
            # De-duplicate by the complete lookup contract, then sort to make
            # manifest identity independent of draft/expression iteration.
            reference_records = sorted({tuple(sorted(item.items())) for item in reference_records})
            reference_records = [dict(item) for item in reference_records]
            if reference_records:
                producer_manifest["reference_data"] = reference_records
            manifest_models = dict(producer_manifest.get("models") or {})
            snapshot = self._snapshots.record_from_artifacts(
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
                    "normalization": context.options.get("normalization_prompt_version", "normalization.v1"),
                    "verification": context.options.get("verification_prompt_version", "verification.v1"),
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
                    "verification": context.options.get("verification_policy_version", "verification_policy.v1"),
                    "signal": context.options.get("signal_policy_version", "signal_policy.v1"),
                },
                quant_market_snapshot_ids=sorted(
                    {str(item) for item in (context.options.get("quant_market_snapshot_ids") or ())}
                ),
                config_hash=str(producer_manifest["configs"]["config_hash"]),
                snapshot_kind=str(context.options.get("replay_snapshot_kind") or "INITIAL"),
                parent_snapshot_id=context.options.get("replay_parent_snapshot_id"),
                supersedes_snapshot_id=context.options.get("replay_supersedes_snapshot_id"),
                pipeline_version=str(context.options.get("replay_pipeline_version") or "pipeline.v3"),
                created_at=snapshot_candidate,
                _persist=False,
            )
            bundle = {
                "occurrences": tuple(context.state.get("occurrences") or ()),
                "lifecycle_events": tuple(context.state.get("lifecycle_events") or ()),
                "verification_results": (
                    tuple(verification_plan.terminal_results_to_insert) if verification_plan is not None else ()
                ),
                "verification_jobs": (
                    tuple(verification_plan.pending_jobs_to_insert) if verification_plan is not None else ()
                ),
            }
            if self._publication_uow is not None:
                # The publication, snapshot membership, signal rows and
                # outbox share this session. Check the lease immediately
                # before the irreversible SQL publication boundary as well as
                # when the stage begins.
                self._require_effect_fence(context, session)
                signals = []
                if self._signal_service is not None and context.artifacts.verification is not None:
                    for result in context.artifacts.verification.results:
                        claim = next(
                            (item for item in context.state.claims if item.claim_id == result.claim_id),
                            None,
                        )
                        if claim is None or claim.claim_type in {"PRICE", "RETURN", "VALUATION", "FINANCIAL_METRIC"}:
                            continue
                        verification_view = result.model_dump(mode="json") | {"provider": "none"}
                        payload = self._signal_service.build_signal(
                            snapshot,
                            claim,
                            verification_view,
                            verification_artifact_id=context.artifacts.verification.artifact_id,
                            trace_id=context.trace.get("trace_id"),
                            decision_id=context.trace.get("decision_id"),
                        )
                        if self._signal_service.policy.evaluate(claim, verification_view, snapshot=snapshot).allowed:
                            signals.append(payload)
                publication_manifest = dict(snapshot.producer_manifest)
                publication_manifest["artifact_membership"] = dict(snapshot.artifact_ids)
                publication_manifest["sealed_signals"] = list(signals)
                self._publication_uow.publish(
                    content_snapshot_id=snapshot.content_snapshot_id,
                    query_hash="ingest:" + snapshot.content_snapshot_id,
                    signal_policy_version=str(context.options.get("signal_policy_version") or "signal-policy.v1"),
                    manifest=publication_manifest,
                    signals=signals,
                    outbox_events=signals,
                    session=session,
                    snapshot=snapshot,
                    snapshot_bundle=bundle,
                )
            else:
                # Preserve direct/custom stage compatibility while production
                # uses the publication UoW above for the atomic boundary.
                store = getattr(self._snapshots, "_store", None)
                saver = getattr(store, "save_bundle", None)
                if saver is not None:
                    saver(snapshot, session=session, **bundle)
                elif store is not None:
                    store.save(snapshot)
            if self._claim_events is not None and session is not None:
                # Initial state events share the snapshot transaction. Missing
                # historical timestamps are never backfilled with fabricated values.
                occurrences = {str(item.claim_id): item for item in context.state.get("occurrences") or ()}
                entries = list(getattr(verification_plan, "artifact_results", ()) or ())
                by_claim = {str(item.claim_id): item for item in entries}
                # A replay/new snapshot extends each claim's existing
                # append-only chain.  Starting from an empty tail would make
                # the second projection race the persisted history and fail
                # even when the event itself is idempotent.
                tails: dict[str, str] = {}
                existing_ids: dict[str, set[str]] = {}
                existing_initial_keys: dict[str, set[str]] = {}
                for claim_id in {str(item.claim_id) for item in context.state.get("claims") or ()}:
                    existing_events = self._claim_events.list_for_claim(claim_id)
                    existing_ids[claim_id] = {item.event_id for item in existing_events}
                    initial_keys = [
                        event_logical_identity(item)
                        for item in existing_events
                        if item.event_type == "VERIFICATION_INITIAL"
                    ]
                    if len(initial_keys) != len(set(initial_keys)):
                        raise ContentSnapshotPersistError(
                            "CONTENT_SNAPSHOT_PERSIST_FAILED: ambiguous verification initial projection"
                        )
                    existing_initial_keys[claim_id] = set(initial_keys)
                    if existing_events:
                        tails[claim_id] = existing_events[-1].event_hash
                pending_events: list[ClaimStateEvent] = []
                for claim in context.state.get("claims") or ():
                    occurrence = occurrences.get(str(claim.claim_id))
                    known_from = getattr(occurrence, "times", None)
                    known_from = getattr(known_from, "snapshot_committed_at", None)
                    if known_from is None:
                        continue
                    entry = by_claim.get(str(claim.claim_id))
                    status = getattr(getattr(entry, "result", None), "status", None) or "VERIFICATION_PENDING"
                    producer_commit = str(
                        getattr(snapshot, "code_sha", "")
                        or (getattr(snapshot, "producer_manifest", {}) or {}).get("code_sha", "")
                    )
                    if not producer_commit:
                        raise ContentSnapshotPersistError(
                            "CONTENT_SNAPSHOT_PERSIST_FAILED: snapshot producer commit is missing"
                        )
                    state_payload = _claim_state_payload(
                        claim, occurrence, entry, snapshot.content_snapshot_id, producer_commit
                    )
                    state_payload["status"] = status
                    pending_events.append(
                        ClaimStateEvent(
                            claim_id=str(claim.claim_id),
                            event_type="VERIFICATION_INITIAL",
                            payload=state_payload,
                            known_from=known_from,
                            source_available_from=getattr(getattr(occurrence, "times", None), "available_from", None),
                        )
                    )
                for event in context.state.get("lifecycle_events") or ():
                    target_type = str(getattr(event, "target_type", ""))
                    if target_type not in {"LifecycleTargetType.CLAIM", "CLAIM"}:
                        continue
                    pending_events.append(
                        ClaimStateEvent(
                            claim_id=str(event.target_id),
                            event_type="LIFECYCLE",
                            payload={"status": event.to_status, "artifact_id": event.lifecycle_event_id},
                            known_from=event.recorded_at,
                            business_valid_from=event.effective_at,
                            source_available_from=event.recorded_at,
                        )
                    )
                for state_event in sorted(
                    pending_events,
                    key=lambda item: (item.claim_id, item.known_from or snapshot.created_at, item.event_id),
                ):
                    # Re-projecting an unchanged lifecycle event is an
                    # idempotent no-op.  Do not rewrite its original chain
                    # predecessor when a later snapshot extends the chain.
                    if state_event.event_id in existing_ids.get(state_event.claim_id, set()):
                        continue
                    if state_event.event_type == "VERIFICATION_INITIAL" and event_logical_identity(
                        state_event
                    ) in existing_initial_keys.get(state_event.claim_id, set()):
                        continue
                    prior = tails.get(state_event.claim_id)
                    if prior:
                        state_event = ClaimStateEvent(
                            claim_id=state_event.claim_id,
                            event_type=state_event.event_type,
                            payload=dict(state_event.payload),
                            known_from=state_event.known_from,
                            business_valid_from=state_event.business_valid_from,
                            business_valid_to=state_event.business_valid_to,
                            known_to=state_event.known_to,
                            source_available_from=state_event.source_available_from,
                            previous_event_hash=prior,
                            legacy_history_incomplete=state_event.legacy_history_incomplete,
                        )
                    committed = self._claim_events.append_in_session(session, state_event)
                    tails[state_event.claim_id] = committed.event_hash
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

    def _require_effect_fence(self, context: PipelineContext, session) -> None:
        if self._fenced_effects is None or not context.worker_id or context.fencing_token is None:
            return
        if session is None:
            # Production SnapshotRecordingStage always has the verification
            # planning transaction. Refuse a configured fence without its
            # transaction instead of silently downgrading the guarantee.
            raise ContentSnapshotPersistError("CONTENT_SNAPSHOT_PERSIST_FAILED: fenced publication session unavailable")
        self._fenced_effects.require_current_in_session(
            session, context.task_id, context.worker_id, context.fencing_token
        )

    @staticmethod
    def _replace_verification_lineage(context: PipelineContext, entries: list[Any]) -> None:
        """Replace the provisional view and rehash its downstream parents."""
        verification = context.artifacts.verification
        if verification is None:
            return
        candidate = VerificationArtifact(
            artifact_id="verification-initial-pending",
            artifact_type="verification",
            producer_stage=verification.producer_stage,
            producer_version=verification.producer_version,
            schema_version=verification.schema_version,
            claim_artifact_id=verification.claim_artifact_id,
            results=list(entries),
            parent_artifact_ids=verification.parent_artifact_ids,
        )
        context.artifacts.verification = VerificationArtifact(
            **{**candidate.__dict__, "artifact_id": artifact_id_of(candidate)}
        )
        lifecycle = context.artifacts.lifecycle
        if lifecycle is not None:
            rebuilt = LifecycleArtifact(
                artifact_id="lifecycle-initial-pending",
                artifact_type="lifecycle",
                producer_stage=lifecycle.producer_stage,
                producer_version=lifecycle.producer_version,
                schema_version=lifecycle.schema_version,
                claim_lifecycle_event_ids=list(lifecycle.claim_lifecycle_event_ids),
                occurrence_lifecycle_event_ids=list(lifecycle.occurrence_lifecycle_event_ids),
                lifecycle_business_as_of=lifecycle.lifecycle_business_as_of,
                lifecycle_knowledge_as_of=lifecycle.lifecycle_knowledge_as_of,
                policy_version=lifecycle.policy_version,
                parent_artifact_ids=tuple(
                    item.artifact_id
                    for item in (context.artifacts.occurrences, context.artifacts.verification)
                    if item is not None
                ),
            )
            context.artifacts.lifecycle = LifecycleArtifact(
                **{**rebuilt.__dict__, "artifact_id": artifact_id_of(rebuilt)}
            )
        knowledge = context.artifacts.knowledge
        if knowledge is not None:
            parent = context.artifacts.lifecycle or context.artifacts.verification
            rebuilt = KnowledgeArtifact(
                artifact_id="knowledge-initial-pending",
                artifact_type="knowledge",
                producer_stage=knowledge.producer_stage,
                producer_version=knowledge.producer_version,
                schema_version=knowledge.schema_version,
                verification_artifact_id=context.artifacts.verification.artifact_id,
                knowledge_units=list(knowledge.knowledge_units),
                parent_artifact_ids=(parent.artifact_id,) if parent is not None else (),
            )
            context.artifacts.knowledge = KnowledgeArtifact(
                **{**rebuilt.__dict__, "artifact_id": artifact_id_of(rebuilt)}
            )
        summary = context.artifacts.summary
        if summary is not None and context.artifacts.knowledge is not None:
            rebuilt = SummaryArtifact(
                artifact_id="summary-initial-pending",
                artifact_type="summary",
                producer_stage=summary.producer_stage,
                producer_version=summary.producer_version,
                schema_version=summary.schema_version,
                knowledge_artifact_id=context.artifacts.knowledge.artifact_id,
                core_summary=summary.core_summary,
                parent_artifact_ids=(context.artifacts.knowledge.artifact_id,),
            )
            context.artifacts.summary = SummaryArtifact(**{**rebuilt.__dict__, "artifact_id": artifact_id_of(rebuilt)})


def _producer_manifest(context: PipelineContext) -> dict[str, Any]:
    """Normalize immutable release provenance without task metadata."""
    manifest = dict(context.options.get("producer_manifest") or {})
    dependency = context.options.get("dependency_lock_hash", manifest.get("python_lock_hash", "unknown"))
    # The explicit top-level option is authoritative when it conflicts with a
    # nested manifest value.  Otherwise preserve an explicitly supplied
    # manifest value, then fall back to the deployment/default release SHA.
    manifest_code_sha = manifest.get("code_sha")
    effective_code_sha = context.options.get("code_sha") or manifest_code_sha or default_code_sha()
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
        (transcript.asr_model if transcript else None) or context.options.get("asr_model") or "unknown",
    )
    models.setdefault(
        "asr_version",
        (transcript.asr_model_version if transcript else None) or context.options.get("asr_model_version") or "unknown",
    )
    pipeline_config = dict(context.options.get("pipeline_config") or {})
    models.setdefault(
        "ocr",
        context.options.get("ocr_model") or (context.artifacts.ocr[0].engine if context.artifacts.ocr else "fixture"),
    )
    models.setdefault(
        "ocr_version",
        context.options.get("ocr_model_version")
        or (context.artifacts.ocr[0].engine_version if context.artifacts.ocr else "1"),
    )
    if context.artifacts.ocr:
        ocr = context.artifacts.ocr[0]
        models.setdefault("ocr_requested_device", ocr.requested_device or "unknown")
        models.setdefault("ocr_actual_device", ocr.actual_device or "unknown")
        models.setdefault("ocr_runtime_identity", _config_hash_of(ocr.runtime_identity))
    models.setdefault(
        "vision",
        context.options.get("vision_model") or pipeline_config.get("vision_model") or "unknown",
    )
    models.setdefault(
        "vision_version",
        context.options.get("vision_model_version") or pipeline_config.get("vision_model_version") or "unknown",
    )
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
            "schema": getattr(semantic_artifact, "segmentation_schema_version", None) or "semantic-segment.v1",
        },
    )
    manifest.setdefault(
        "atomic_claim_extraction",
        {
            "model": pipeline_config.get("extraction_model")
            or context.options.get("llm_model")
            or models.get("llm", "unknown"),
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
    prompts.setdefault(
        "vision",
        context.options.get("vision_prompt_version")
        or pipeline_config.get("vision_prompt_version", "vision-context.prompt.v1"),
    )
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
    configs.setdefault(
        "knowledge_frame_planner_version",
        pipeline_config.get("knowledge_frame_planner_version", "knowledge-frame-plan.v1"),
    )
    configs.setdefault(
        "knowledge_evidence_window_planner_version",
        pipeline_config.get("knowledge_evidence_window_planner_version", "knowledge-evidence-window.v1"),
    )
    configs.setdefault(
        "transcript_visual_crosscheck_version",
        pipeline_config.get("transcript_visual_crosscheck_version", "transcript-visual-crosscheck.v1"),
    )
    configs.setdefault(
        "vision_adapter_version", pipeline_config.get("vision_adapter_version", "http-vision-adapter.v1")
    )
    manifest["configs"] = configs
    return manifest


class ClaimPersistenceStage:
    """Persist canonical pre-snapshot state under the task fencing boundary.

    Claim, evidence and ClaimArtifact membership are a single logical effect:
    a snapshot must never observe only some of them, and a worker that lost its
    lease must not leave an otherwise resumable partial projection behind.
    """

    name = "claim_persistence"
    required_inputs = ("evidence", "claims")
    output_types = ()

    def __init__(self, claims: Any, artifacts: Any, fenced_effects=None) -> None:
        self._claims = claims
        self._artifacts = artifacts
        self._fenced_effects = fenced_effects

    def execute(self, context: PipelineContext) -> PipelineContext:
        # A production queue attempt supplies both values.  Treat a partial
        # fence context as a configuration error before doing any write.
        fenced_attempt = context.worker_id is not None or context.fencing_token is not None
        if fenced_attempt:
            if (
                self._fenced_effects is None
                or not context.task_id
                or not context.worker_id
                or context.fencing_token is None
            ):
                raise RuntimeError("CLAIM_PERSISTENCE_FENCED_UOW_REQUIRED")
            intent = EffectIntent(
                "pre-snapshot:claim-persistence",
                "PRE_SNAPSHOT_CLAIM_PERSISTENCE",
                {
                    "claim_ids": sorted(str(item.claim_id) for item in context.state.claims),
                    "evidence_artifact_id": str(getattr(context.artifacts.evidence, "artifact_id", "") or ""),
                    "claims_artifact_id": str(getattr(context.artifacts.claims, "artifact_id", "") or ""),
                },
            )
            self._fenced_effects.execute_sql(
                context.task_id,
                context.worker_id,
                context.fencing_token,
                intent,
                lambda session: self._persist(context, session=session),
            )
        else:
            self._persist(context)
        # ``execute_sql`` returns None for a completed stable effect.  That is
        # a successful resume, not a reason for PersistStage to reopen a
        # repository-local write transaction.
        context.state.claims_persisted = True
        return _stage_result(context)

    def _persist(self, context: PipelineContext, *, session=None) -> PipelineContext:
        def write(repository, method: str, *args) -> None:
            if repository is None:
                return
            if session is not None:
                in_session = getattr(repository, f"{method}_in_session", None)
                if in_session is not None:
                    in_session(session, *args)
                    return
                # SQL-shaped repositories may not silently create an
                # independent transaction inside the claimed task effect.
                if getattr(repository, "_sessions", None) is not None:
                    raise RuntimeError(
                        f"CLAIM_PERSISTENCE_SESSION_CAPABLE_REPOSITORY_REQUIRED:{type(repository).__name__}"
                    )
            getattr(repository, method)(*args)

        if self._artifacts is not None and context.artifacts.evidence is not None:
            write(self._artifacts, "put", context.artifacts.evidence)
        for claim in context.state.claims:
            # Final canonical claims deliberately do not own source-specific
            # evidence.  Occurrence role memberships are persisted by the
            # occurrence stage and remain the sole evidence ownership path.
            write(self._claims, "save", claim)
        if self._artifacts is not None and context.artifacts.claims is not None:
            write(self._artifacts, "put", context.artifacts.claims)
            if hasattr(self._artifacts, "put_claim_members"):
                write(self._artifacts, "put_claim_members", context.artifacts.claims)
        return context


def _config_hash_of(config: dict | None) -> str:
    import hashlib
    import json

    if not config:
        return ""
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _claim_state_payload(
    claim: Any, occurrence: Any, entry: Any, snapshot_id: str, producer_commit: str
) -> dict[str, Any]:
    """Capture formal projection inputs in the immutable state event."""
    times = getattr(occurrence, "times", None)

    def value(item: Any) -> Any:
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, (list, tuple)):
            return [value(child) for child in item]
        if hasattr(item, "isoformat"):
            return item.isoformat()
        return str(item)

    support_status = str(getattr(claim, "source_support_status", "") or "UNSUPPORTED").upper()
    support_count = {
        # Formal min_support is a two-level source-support threshold:
        # partially supported claims satisfy level 1, fully supported claims
        # satisfy level 2.  Ambiguous and unsupported claims are excluded.
        "SUPPORTED": 2,
        "PARTIALLY_SUPPORTED": 1,
        "UNSUPPORTED": 0,
        "AMBIGUOUS": 0,
    }.get(support_status, 0)
    # Canonical FinancialClaim retains its established storage vocabulary
    # (SUPPORTED/PARTIALLY_SUPPORTED).  The immutable public ledger owns the
    # current public support vocabulary used by content-knowledge-bundle.v1.
    public_support_status = {
        "SUPPORTED": "SOURCE_SUPPORTED",
        "PARTIALLY_SUPPORTED": "SOURCE_LOCATED",
    }.get(support_status, support_status)
    occurrence_review = _occurrence_review(occurrence) if occurrence is not None else {
        "status": "NOT_REQUIRED", "reason_codes": []
    }
    # The immutable state event must agree with the write-model projection:
    # unresolved multimodal conflicts are source-located audit material, not
    # source-supported public evidence.
    if occurrence_review["status"] == "HUMAN_REVIEW_REQUIRED":
        public_support_status = "SOURCE_LOCATED"
        support_count = min(support_count, 1)
    asserted_at = getattr(times, "asserted_at", None)
    source_quality = getattr(times, "source_availability_quality", "UNKNOWN")
    return {
        "snapshot_id": snapshot_id,
        "claim_id": str(getattr(claim, "claim_id", "")),
        "occurrence_id": str(getattr(occurrence, "occurrence_id", "")),
        "semantic_segment_id": str(getattr(occurrence, "semantic_segment_id", "")),
        "asserted_at": value(asserted_at),
        "source_available_at": value(getattr(times, "source_available_at", None)),
        "available_from": value(getattr(times, "available_from", None)),
        "source_availability_quality": str(getattr(source_quality, "value", source_quality) or "UNKNOWN"),
        "temporal_bindings": value(getattr(claim, "temporal_bindings", ())),
        "evidence_refs": value(getattr(occurrence, "evidence_refs", ())),
        "symbol": str(getattr(claim, "subject_id", "") or ""),
        "support_status": public_support_status,
        "support_count": support_count,
        "occurrence_review": occurrence_review,
        "producer_commit": str(producer_commit),
        "signal_policy_version": "signal-policy.v1",
        "verification_status": str(getattr(getattr(entry, "result", None), "status", "") or ""),
    }


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
        publication_uow=None,
        fenced_effects=None,
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
        self._publication_uow = publication_uow
        self._fenced_effects = fenced_effects

    def execute(self, context: PipelineContext) -> PipelineContext:
        # A queued production task must never fall back to independent
        # repository transactions.  That would make the lease check a
        # best-effort precondition rather than part of the business write.
        if context.worker_id and context.fencing_token is not None:
            if self._fenced_effects is None:
                raise RuntimeError("PERSIST_FENCED_EFFECTS_REQUIRED")
            intent = EffectIntent(
                "projection:video-persist",
                "SQL_PROJECTION",
                {"artifacts": sorted(context.artifacts.artifact_ids()), "task_id": context.task_id},
            )
            return (
                self._fenced_effects.execute_sql(
                    context.task_id,
                    context.worker_id,
                    context.fencing_token,
                    intent,
                    lambda session: self._persist(context, session=session),
                )
                or context
            )
        return self._persist(context)

    def _persist(self, context: PipelineContext, *, session=None) -> PipelineContext:
        def write(repository, method: str, *args) -> None:
            """Use the caller-owned fenced transaction for SQL adapters.

            Production adapters expose ``<method>_in_session``.  A test
            double without a SQL session remains a valid pure-stage seam, but
            a production-shaped adapter without the method is rejected rather
            than silently opening a second transaction.
            """
            if repository is None:
                return
            if session is not None:
                in_session = getattr(repository, f"{method}_in_session", None)
                if in_session is not None:
                    in_session(session, *args)
                    return
                if getattr(repository, "_sessions", None) is not None:
                    raise RuntimeError(f"PERSIST_SESSION_CAPABLE_REPOSITORY_REQUIRED:{type(repository).__name__}")
            getattr(repository, method)(*args)

        if self._artifacts:
            for artifact in context.artifacts.artifacts():
                write(self._artifacts, "put", artifact)
        if self._claims and not context.state.claims_persisted:
            for claim in context.state.claims:
                write(self._claims, "save", claim)
        if self._artifacts and context.artifacts.claims is not None and hasattr(self._artifacts, "put_claim_members"):
            write(self._artifacts, "put_claim_members", context.artifacts.claims)
        video = context.state["video"]
        write(self._videos, "upsert", video, context.state["segments"])
        write(self._chapters, "replace_for_video", video.video_id, context.state["chapters"])
        write(self._knowledge, "replace_for_video", video.video_id, context.state["knowledge"])
        if self._verifications:
            # Verification ledger trace must identify the request lineage, not
            # the task UUID.  The latter is an operational identifier and is
            # already persisted on the task/checkpoint rows.
            write(self._verifications, "append", context.state["knowledge"], context.trace.get("trace_id"))
        if self._multimodal:
            eligible_ids = _eligible_visual_ids(context)
            write(
                self._multimodal,
                "replace",
                video.video_id,
                [item for item in context.state.get("frames") or [] if str(item.get("frame_id") or "") in eligible_ids],
                [
                    item
                    for item in context.state.get("ocr_evidence") or []
                    if str(item.get("frame_id") or "") in eligible_ids
                ],
                list(context.state.get("eligible_frame_insights") or []),
                list(context.state.get("temporal_windows") or []),
            )
        if self._financial:
            write(
                self._financial,
                "replace",
                video.video_id,
                list(context.state.get("financial_numeric_facts") or []),
                list(context.state.get("financial_events") or []),
            )
        if self._entities:
            write(self._entities, "replace", video.video_id, context.state["knowledge"])
        write(self._summaries, "upsert", context.state["summary"])
        return _stage_result(context)


class IndexStage:
    name = "index"
    required_inputs = ("knowledge",)
    output_types = ()

    def __init__(self, index: KnowledgeIndex, fenced_effects=None) -> None:
        self._index = index
        self._fenced_effects = fenced_effects

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.worker_id and context.fencing_token is not None:
            if self._fenced_effects is None:
                raise RuntimeError("INDEX_FENCED_EFFECTS_REQUIRED")
            intent = EffectIntent(
                "index:knowledge",
                "KNOWLEDGE_INDEX",
                {
                    "knowledge_ids": sorted(str(item.knowledge_uid) for item in context.state["knowledge"]),
                    "snapshot_id": str(context.state.content_snapshot_id or ""),
                },
            )
            # Qdrant is a derived, optional projection.  Persisting this
            # intent under the ingestion fence is the only main-pipeline
            # obligation; the worker's durable projection dispatcher invokes
            # the external index after task completion.  In particular, a
            # Qdrant timeout must never turn completed SQL publication into a
            # failed ingestion task.
            self._fenced_effects.prepare_external(
                context.task_id,
                context.worker_id,
                context.fencing_token,
                intent,
            )
        else:
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

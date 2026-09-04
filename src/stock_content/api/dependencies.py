from __future__ import annotations

import os

from stock_content.adapters.http import ContentModelClient, HttpExternalFactProvider, QuantExternalFactProvider
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
    ClaimOccurrenceRepository,
    ClaimStateEventRepository,
    LifecycleRepository,
    PostgresChapterRepository,
    PostgresContentTaskRepository,
    PostgresFinancialEntityRepository,
    PostgresFinancialRepository,
    PostgresKnowledgeRepository,
    PostgresMultimodalRepository,
    PostgresSummaryRepository,
    PostgresTaskRunRepository,
    PostgresVerificationJobRepository,
    PostgresVerificationRepository,
    PostgresVideoRepository,
    PublicationRepository,
    SemanticSegmentRepository,
    SignalOutboxRepository,
    SqlArtifactRepository,
    SqlClaimRepository,
    SqlSnapshotStore,
)
from stock_content.adapters.qdrant import NullKnowledgeIndex, QdrantKnowledgeIndex
from stock_content.adapters.reference import QuantTemporalReferenceAdapter
from stock_content.adapters.sources import BilibiliSourceAdapter, XiaoeHlsSourceAdapter
from stock_content.application.historical_claim_projector import HistoricalClaimProjector
from stock_content.application.pipeline import ContentPipeline
from stock_content.application.publication_unit_of_work import PublicationUnitOfWork
from stock_content.application.service import ContentApplication
from stock_content.application.signal_service import SignalService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.stage_runner import wrap_all
from stock_content.application.stages import (
    ASRStage,
    AtomicClaimExtractionStage,
    AudioStage,
    BuildVideoStage,
    ChapterStage,
    ClaimCanonicalizationStage,
    ClaimOccurrencePersistenceStage,
    ClaimPersistenceStage,
    DownloadStage,
    EvidenceGroundingStage,
    FinancialEnrichmentStage,
    FrameExtractionStage,
    IndexStage,
    KnowledgeExtractionStage,
    LifecycleProjectionStage,
    MultimodalContextStage,
    OCRStage,
    PersistStage,
    ResolveSourceStage,
    SemanticContextStage,
    SemanticSegmentationStage,
    SnapshotRecordingStage,
    SpeakerDiarizationStage,
    SummaryStage,
    TemporalNormalizationStage,
    TemporalWindowStage,
    TranscriptPostprocessStage,
    VerificationStage,
    VisionStage,
)
from stock_content.application.task_lease_service import TaskLeaseService
from stock_content.domain.atomic_claim_extractor import AtomicClaimExtractor
from stock_content.domain.chapter import ChapterSegmenter
from stock_content.domain.external_fact_verifier import ExternalFactVerifier
from stock_content.domain.lineage import default_code_sha
from stock_content.domain.multimodal_context_builder import MultimodalContextBuilder
from stock_content.domain.semantic_segmenter import SemanticSegmenter
from stock_content.domain.signal_contract import validate_signal_v4
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
    "semantic_segmentation": "1.0.0",
    "semantic_context": "1.0.0",
    "atomic_claim_extraction": "1.0.0",
    "evidence_grounding": "1.0.0",
    "temporal_normalization": "final.1.0",
    "claim_canonicalization": "1.0.0",
    "claim_occurrence_persistence": "1.0.0",
    "lifecycle_projection": "1.0.0",
    "chapter": "1.0.0",
    "temporal_window": "1.0.0",
    "knowledge": "final.1.0",
    "verification": "1.0.0",
    "financial_enrichment": "1.0.0",
    "summary": "1.0.0",
    "content_snapshot": "final.1.0",
    "claim_persistence": "final.1.0",
    "persist": "1.0.0",
    "index": "final.1.0",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pipeline_config_from_env() -> dict[str, object]:
    """Read the semantic pipeline's complete, reproducible configuration."""
    config = {
        "semantic_segmentation_enabled": _env_bool("CONTENT_SEMANTIC_SEGMENTATION_ENABLED", True),
        "segmentation_model": os.getenv("CONTENT_SEGMENTATION_MODEL", ""),
        "segmentation_prompt_version": os.getenv(
            "CONTENT_SEGMENTATION_PROMPT_VERSION", "semantic-segmentation.prompt.v1"
        ),
        "extraction_model": os.getenv("CONTENT_EXTRACTION_MODEL", ""),
        "extraction_prompt_version": os.getenv(
            "CONTENT_EXTRACTION_PROMPT_VERSION", "atomic-claim-extraction.prompt.v1"
        ),
        "temporal_normalization_version": os.getenv(
            "CONTENT_TEMPORAL_NORMALIZATION_VERSION", "temporal-normalization.final.v1"
        ),
        "semantic_global_max_safe_tokens": int(
            os.getenv("CONTENT_SEMANTIC_GLOBAL_MAX_SAFE_TOKENS", "3200")
        ),
        "semantic_long_video_block_tokens": int(
            os.getenv("CONTENT_SEMANTIC_LONG_VIDEO_BLOCK_TOKENS", "3200")
        ),
        "semantic_block_overlap_segments": int(
            os.getenv("CONTENT_SEMANTIC_BLOCK_OVERLAP_SEGMENTS", "2")
        ),
        "legacy_chapter_extraction_enabled": _env_bool("CONTENT_LEGACY_CHAPTER_EXTRACTION_ENABLED", False),
        "public_pit_default_mode": os.getenv("CONTENT_PUBLIC_PIT_DEFAULT_MODE", "PUBLIC_STRICT"),
        # Compatibility aliases retained for callers using the first draft.
        "semantic_safe_tokens": int(os.getenv("CONTENT_SEMANTIC_GLOBAL_MAX_SAFE_TOKENS", "3200")),
        "semantic_padding_ms": int(os.getenv("CONTENT_SEMANTIC_PADDING_MS", "4000")),
        "atomic_claim_extraction_enabled": _env_bool("CONTENT_ATOMIC_CLAIM_EXTRACTION_ENABLED", True),
    }
    # Keep the disabled/offline configuration byte-for-byte compatible with
    # historical snapshot identities.  Reference settings enter the config
    # hash only when the feature is explicitly configured.
    if _env_bool("CONTENT_TEMPORAL_REFERENCE_ENABLED", False) or any(os.getenv(name) is not None for name in (
        "CONTENT_TEMPORAL_REFERENCE_REQUIRED", "CONTENT_TEMPORAL_REFERENCE_URL", "CONTENT_TEMPORAL_REFERENCE_API_KEY",
        "CONTENT_TEMPORAL_REFERENCE_TIMEOUT_SECONDS",
    )):
        config.update({
            "temporal_reference_enabled": _env_bool("CONTENT_TEMPORAL_REFERENCE_ENABLED", False),
            "temporal_reference_required": _env_bool("CONTENT_TEMPORAL_REFERENCE_REQUIRED", False),
            "temporal_reference_url": os.getenv("CONTENT_TEMPORAL_REFERENCE_URL", ""),
            "temporal_reference_timeout_seconds": os.getenv(
                "CONTENT_TEMPORAL_REFERENCE_TIMEOUT_SECONDS", "10"
            ),
        })
    return config


def build_application(
    database_url: str | None = None,
    enable_qdrant: bool | None = None,
    *,
    reference_provider=None,
    reference_snapshot_provider=None,
    temporal_reference_provider=None,
    temporal_reference_snapshot_provider=None,
) -> ContentApplication:
    # Validate release identity before opening a database or constructing any
    # external client.  Development/test environments retain the historical
    # ``unknown`` fallback through the domain policy.
    default_code_sha()
    config = pipeline_config_from_env()
    # Explicit aliases keep test/application factories compatible with both
    # the short port names and the fully-qualified configuration vocabulary.
    reference_provider = reference_provider or temporal_reference_provider
    reference_snapshot_provider = reference_snapshot_provider or temporal_reference_snapshot_provider
    if reference_provider is None and bool(config.get("temporal_reference_enabled", False)):
        url = str(config.get("temporal_reference_url", "") or "").strip()
        if not url:
            raise ValueError(
                "CONTENT_TEMPORAL_REFERENCE_URL is required when temporal reference provider is enabled"
            )
        try:
            timeout = float(config.get("temporal_reference_timeout_seconds", "10"))
        except (TypeError, ValueError) as exc:
            raise ValueError("CONTENT_TEMPORAL_REFERENCE_TIMEOUT_SECONDS must be a positive number") from exc
        if timeout <= 0:
            raise ValueError("CONTENT_TEMPORAL_REFERENCE_TIMEOUT_SECONDS must be a positive number")
        reference_provider = QuantTemporalReferenceAdapter(
            url,
            api_key=os.getenv("CONTENT_TEMPORAL_REFERENCE_API_KEY") or None,
            timeout=timeout,
        )
    if reference_snapshot_provider is None and reference_provider is not None:
        if all(hasattr(reference_provider, name) for name in (
            "get_exchange_calendar_snapshot", "get_fiscal_calendar_snapshot", "get_period_snapshot"
        )):
            reference_snapshot_provider = reference_provider
    if bool(config.get("temporal_reference_required", False)) and reference_provider is None:
        raise ValueError(
            "CONTENT_TEMPORAL_REFERENCE_REQUIRED=true requires an enabled or injected temporal reference provider"
        )
    if bool(config.get("temporal_reference_required", False)) and reference_snapshot_provider is None:
        raise ValueError(
            "CONTENT_TEMPORAL_REFERENCE_REQUIRED=true requires an exact temporal reference snapshot provider"
        )
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
    artifacts = SqlArtifactRepository(database.session_factory)
    claims = SqlClaimRepository(database.session_factory)
    verification_jobs = PostgresVerificationJobRepository(database.session_factory)
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
    signal_outbox = SignalOutboxRepository(database.session_factory)
    claim_events = ClaimStateEventRepository(database.session_factory)
    publication_repository = PublicationRepository(database.session_factory)

    def save_snapshot_bundle(session, snapshot, _manifest, bundle):
        store = snapshot_service._store
        store.save_bundle(snapshot, session=session, **bundle)

    def seal_signal_rows(_session, rows, *, publication_run_id=None):
        for payload in rows:
            validate_signal_v4(payload)
        if publication_run_id is not None:
            publication_repository.save_sealed_signals_in_session(
                _session,
                publication_run_id,
                tuple(rows),
            )

    publication_uow = PublicationUnitOfWork(
        publication_repository,
        snapshot_writer=save_snapshot_bundle,
        signal_writer=seal_signal_rows,
        outbox_writer=lambda session, rows: [signal_outbox.enqueue_in_session(session, row) for row in rows],
    )
    occurrences = ClaimOccurrenceRepository(database.session_factory)
    lifecycle = LifecycleRepository(database.session_factory)
    def snapshot_membership(snapshot_id: str, claim_id: str) -> bool:
        from stock_content.adapters.postgres.models import ContentArtifactRow, ContentSnapshotRow
        from stock_content.domain.artifacts import deserialize_artifact
        with database.session_factory() as session:
            snapshot = session.get(ContentSnapshotRow, snapshot_id)
            claims_artifact = (dict(snapshot.artifact_ids or {}).get("claims") if snapshot else None)
            artifact_row = session.get(ContentArtifactRow, claims_artifact) if claims_artifact else None
            if artifact_row is None:
                return False
            artifact = deserialize_artifact(dict(artifact_row.payload or {}))
            return str(claim_id) in {
                str(getattr(item, "claim_id", item)) for item in (getattr(artifact, "claims", ()) or ())
            }

    def snapshot_claim_ids(snapshot_id: str) -> list[str]:
        from stock_content.adapters.postgres.models import ContentArtifactRow, ContentSnapshotRow
        from stock_content.domain.artifacts import deserialize_artifact
        with database.session_factory() as session:
            snapshot = session.get(ContentSnapshotRow, snapshot_id)
            claims_artifact = dict(snapshot.artifact_ids or {}).get("claims") if snapshot else None
            artifact_row = session.get(ContentArtifactRow, claims_artifact) if claims_artifact else None
            if artifact_row is None:
                return []
            artifact = deserialize_artifact(dict(artifact_row.payload or {}))
            return [str(getattr(item, "claim_id", item)) for item in (getattr(artifact, "claims", ()) or ())]

    historical_projector = HistoricalClaimProjector(
        event_loader=claim_events.list_for_claim,
        membership=snapshot_membership,
        history_incomplete=claim_events.is_history_incomplete,
        snapshot_claim_ids=snapshot_claim_ids,
    )
    signal_service = SignalService()
    semantic_segments = SemanticSegmentRepository(database.session_factory)
    segmentation_client = ContentModelClient(model=str(config["segmentation_model"]))
    extraction_client = ContentModelClient(model=str(config["extraction_model"]))
    semantic_segmenter = SemanticSegmenter(
        segmentation_client,
        model_id=str(config["segmentation_model"]),
        prompt_version=str(config["segmentation_prompt_version"]),
        safe_tokens=int(config["semantic_global_max_safe_tokens"]),
        block_tokens=int(config["semantic_long_video_block_tokens"]),
        segment_overlap=int(config["semantic_block_overlap_segments"]),
        allow_offline_fixture=False,
    )
    atomic_extractor = AtomicClaimExtractor(
        extraction_client,
        model_id=str(config["extraction_model"]),
        prompt_version=str(config["extraction_prompt_version"]),
        allow_offline_fixture=False,
    )
    semantic_enabled = bool(config["semantic_segmentation_enabled"])
    legacy_enabled = bool(config["legacy_chapter_extraction_enabled"]) or not semantic_enabled
    semantic_stages = [
        ChapterStage(ChapterSegmenter()),
        TemporalWindowStage(TemporalWindowBuilder()),
        SemanticSegmentationStage(segmenter=semantic_segmenter, repository=semantic_segments),
        SemanticContextStage(padding_ms=int(config["semantic_padding_ms"])),
        AtomicClaimExtractionStage(extractor=atomic_extractor),
        EvidenceGroundingStage(),
        TemporalNormalizationStage(
            normalization_version=str(config["temporal_normalization_version"]),
            reference_provider=reference_provider,
        ),
        ClaimCanonicalizationStage(),
        ClaimOccurrencePersistenceStage(occurrences),
    ] if semantic_enabled else []
    compatibility_stages = [] if semantic_enabled else [
        ChapterStage(ChapterSegmenter()),
        TemporalWindowStage(TemporalWindowBuilder()),
    ]
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
        *semantic_stages,
        *compatibility_stages,
        KnowledgeExtractionStage(
            external_verifier=ExternalFactVerifier(
                provider=external_provider if external_provider.configured() else None
            ),
            authoritative_only=not legacy_enabled,
        ),
        VerificationStage(),
        FinancialEnrichmentStage(),
        LifecycleProjectionStage(lifecycle),
        SummaryStage(SummaryGenerator()),
        ClaimPersistenceStage(claims, artifacts),
        SnapshotRecordingStage(
            snapshot_service, artifacts, occurrences, lifecycle,
            verification_repository=verification_jobs,
            verification_job_repository=verification_jobs,
            claim_event_repository=claim_events,
            signal_service=signal_service,
            publication_uow=publication_uow,
        ),
        PersistStage(
            videos,
            chapters,
            knowledge,
            summaries,
            multimodal,
            financial,
            entities,
            verifications,
            artifacts,
            claims,
            snapshot_service=snapshot_service,
            signal_service=signal_service,
            signal_outbox=signal_outbox,
            publication_uow=publication_uow,
        ),
        IndexStage(index),
    ]
    # P0 C-01：生产主链路全部经 StageRunner，产出 Artifact Checkpoint v2。
    # 版本号来自稳定常量，不得随机生成。
    pipeline = ContentPipeline(
        wrap_all(
            stages,
            STAGE_VERSIONS,
            artifact_repository=artifacts,
            legacy_fallback=False,
        )
    )
    return ContentApplication(
        tasks,
        videos,
        knowledge,
        index,
        pipeline,
        chapters,
        summaries,
        snapshot_service=snapshot_service,
        artifact_repository=artifacts,
        claim_repository=claims,
        signal_outbox=signal_outbox,
        verification_job_repository=verification_jobs,
        occurrence_repository=occurrences,
        lifecycle_repository=lifecycle,
        pipeline_config=config,
        temporal_reference_snapshot_provider=reference_snapshot_provider,
        publication_uow=publication_uow,
        signal_service=signal_service,
        claim_event_repository=claim_events,
        historical_projector=historical_projector,
        task_lease_service=TaskLeaseService(PostgresTaskRunRepository(database.session_factory)),
    )

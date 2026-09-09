"""强类型 Pipeline Artifact（详细修改方案 §4 P0-1）。

Artifact = immutable；Stage 之间不再通过字符串 key 隐式耦合传递核心业务对象。
第一阶段与 ``PipelineContext.data`` 双轨共存（data 标记 deprecated），
全部 Stage 迁移完成后再删除 ``data``。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

ARTIFACT_SCHEMA_VERSION = "artifact.v1"

# ``artifact.v1`` was already persisted before the targeted-frame and GPU OCR
# provenance fields were introduced.  The schema label therefore cannot tell
# us which canonical identity rule created a row.  Keep this compatibility
# marker internal to the Python object/serializer: it is inferred only from a
# historical payload which lacks the additive field(s), and is deliberately
# not itself persisted.
_IDENTITY_PROFILE_CURRENT = "current"
_IDENTITY_PROFILE_LEGACY_FRAME = "legacy-frame.v1"
_IDENTITY_PROFILE_LEGACY_OCR = "legacy-ocr.v1"
_IDENTITY_PROFILES = {
    _IDENTITY_PROFILE_CURRENT,
    _IDENTITY_PROFILE_LEGACY_FRAME,
    _IDENTITY_PROFILE_LEGACY_OCR,
}


def canonical_json(payload: Any) -> str:
    """canonical JSON：key 排序、紧凑分隔符、统一序列化。"""

    def _default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return str(value)

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)


def _membership_key(value: Any) -> str:
    """Stable key for final artifact fields that represent set membership."""
    for name in ("claim_id", "occurrence_id", "lifecycle_event_id", "knowledge_uid"):
        candidate = getattr(value, name, None)
        if candidate is None and isinstance(value, dict):
            candidate = value.get(name)
        if candidate is not None:
            return f"{name}:{candidate}"
    return canonical_json(value)


def _stable_membership(
    values: list[Any] | tuple[Any, ...] | None,
    *,
    field_identity: bool = True,
) -> list[Any]:
    """Deduplicate and sort membership while retaining the original values.

    Verification results are intentionally keyed by their complete payload:
    two decisions for the same claim must not be collapsed merely because
    they share a claim_id.
    """
    candidates = sorted(
        (
            (
                _membership_key(value) if field_identity else canonical_json(value),
                canonical_json(value),
                value,
            )
            for value in (values or ())
        ),
        key=lambda item: (item[0], item[1]),
    )
    result: list[Any] = []
    seen: set[str] = set()
    for key, _serialized, value in candidates:
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def content_hash_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def artifact_identity_payload(artifact: "ArtifactBase") -> dict[str, Any]:
    """Return the immutable, content-addressed identity of an artifact.

    Runtime/database fields are deliberately excluded.  In particular an
    artifact's id and creation time must never be able to change its hash.
    This helper is also used by SQL repositories when checking an existing
    row, so the rule is shared by all persistence implementations.
    """
    payload = artifact.to_dict()
    excluded = {"artifact_id", "created_at", "content_hash"}
    if artifact.artifact_type == "frame" and artifact._identity_profile != _IDENTITY_PROFILE_LEGACY_FRAME:
        # Durable object roots vary by worker and replay environment.  Frame
        # identity is the immutable media/plan/timestamp/image tuple, never a
        # local object-store locator.
        excluded.add("storage_ref")
    return {key: value for key, value in payload.items() if key not in excluded}


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ArtifactBase:
    artifact_id: str
    artifact_type: str
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    created_at: datetime = field(default_factory=_utcnow)
    producer_stage: str = ""
    producer_version: str = "1.0.0"
    parent_artifact_ids: tuple[str, ...] = ()
    # Persisted immutable value.  It is populated from the canonical
    # identity when callers use the public constructors without supplying it;
    # deserialization supplies the stored value explicitly so tampering is
    # visible to repository integrity checks.
    content_hash: str = ""
    # Internal deserialization marker.  It remains an init field so existing
    # reconstruction idioms using ``artifact.__dict__`` keep working, but is
    # removed by ``to_dict`` and cannot become part of a persisted payload.
    _identity_profile: str = field(
        default=_IDENTITY_PROFILE_CURRENT,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._identity_profile not in _IDENTITY_PROFILES:
            raise ValueError("unknown artifact identity profile")
        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                content_hash_of(artifact_identity_payload(self)),
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        profile = payload.pop("_identity_profile", _IDENTITY_PROFILE_CURRENT)
        # Re-serialize a historic row in the exact field shape that was
        # hashed when it was first persisted.  New artifacts always retain
        # the additive identity fields, including their explicit empty values.
        if profile == _IDENTITY_PROFILE_LEGACY_FRAME:
            payload.pop("planner_request_id", None)
        elif profile == _IDENTITY_PROFILE_LEGACY_OCR:
            payload.pop("requested_device", None)
            payload.pop("actual_device", None)
            payload.pop("runtime_identity", None)
        return payload


@dataclass(frozen=True)
class SourceArtifact(ArtifactBase):
    source_type: str = ""
    source_ref: str = ""
    source_content_hash: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)
    raw_content_hash: str = ""
    raw_content_length: int | None = None
    raw_storage_uri: str | None = None
    source_identity_hash: str = ""
    source_version_id: str = ""
    resolver_version: str = ""


@dataclass(frozen=True)
class MediaArtifact(ArtifactBase):
    source_artifact_id: str = ""
    media_uri: str = ""
    duration_ms: int | None = None
    audio_hash: str | None = None
    video_hash: str | None = None
    container: str | None = None
    audio_codec: str | None = None
    video_codec: str | None = None
    extractor_version: str | None = None


@dataclass(frozen=True)
class TranscriptSegmentItem:
    # Empty is retained solely for old serialized fixtures.  New transcript
    # artifacts should provide media_artifact_id/asr manifest to derive it.
    segment_id: str = ""
    segment_index: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    text: str = ""
    raw_text: str | None = None
    normalized_text: str | None = None
    confidence: float | None = None
    source: str = "ASR"
    source_artifact_id: str = ""
    alignment_status: str = "ALIGNED"
    speaker_id: str | None = None
    media_artifact_id: str = ""
    asr_model: str = "unknown"
    asr_model_version: str = "unknown"

    def __post_init__(self) -> None:
        if self.segment_id:
            return
        raw_text = self.raw_text if self.raw_text is not None else self.text
        # Compatibility construction is deliberately explicit in its marker;
        # it is never used as the authoritative media/ASR identity.
        if self.media_artifact_id:
            object.__setattr__(
                self,
                "segment_id",
                transcript_segment_id(
                    media_artifact_id=self.media_artifact_id,
                    asr_model=self.asr_model,
                    asr_model_version=self.asr_model_version,
                    segment_index=self.segment_index,
                    start_ms=round(self.start_seconds * 1000),
                    end_ms=round(self.end_seconds * 1000),
                    raw_text=raw_text,
                ),
            )
        else:
            object.__setattr__(
                self,
                "segment_id",
                legacy_transcript_segment_id(self.segment_index, self.start_seconds, self.end_seconds, raw_text),
            )

    @property
    def start_ms(self) -> int:
        return round(self.start_seconds * 1000)

    @property
    def end_ms(self) -> int:
        return round(self.end_seconds * 1000)


@dataclass(frozen=True)
class TranscriptArtifact(ArtifactBase):
    media_artifact_id: str = ""
    language: str | None = None
    segments: list[TranscriptSegmentItem] = field(default_factory=list)
    asr_model: str = "unknown"
    asr_model_version: str = "unknown"

    def __post_init__(self) -> None:
        # Dataclass frozen means we replace the list rather than mutating a
        # caller-owned list.  Existing callers can continue omitting ids.
        normalized: list[TranscriptSegmentItem] = []
        for item in self.segments:
            manifest_mismatch = (
                not item.media_artifact_id
                or item.media_artifact_id != self.media_artifact_id
                or item.asr_model != self.asr_model
                or item.asr_model_version != self.asr_model_version
            )
            if manifest_mismatch:
                item = TranscriptSegmentItem(
                    # Any manifest mismatch invalidates the prior identity;
                    # never preserve an id computed under another manifest.
                    segment_id="",
                    segment_index=item.segment_index,
                    start_seconds=item.start_seconds,
                    end_seconds=item.end_seconds,
                    text=item.text,
                    raw_text=item.raw_text,
                    normalized_text=item.normalized_text,
                    confidence=item.confidence,
                    source=item.source,
                    source_artifact_id=item.source_artifact_id,
                    alignment_status=item.alignment_status,
                    speaker_id=item.speaker_id,
                    media_artifact_id=self.media_artifact_id,
                    asr_model=self.asr_model,
                    asr_model_version=self.asr_model_version,
                )
            normalized.append(item)
        object.__setattr__(self, "segments", normalized)
        super().__post_init__()


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_type: str
    evidence_text: str = ""
    start_ms: int | None = None
    end_ms: int | None = None
    confidence_score: float | None = None
    locator: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    normalized_text: str = ""
    # Exact immutable producer artifact for this evidence item.  Kept last
    # with a default so older serialized EvidenceItem payloads remain valid.
    source_artifact_id: str = ""


@dataclass(frozen=True)
class EvidenceArtifact(ArtifactBase):
    transcript_artifact_id: str = ""
    evidences: list[EvidenceItem] = field(default_factory=list)
    source_artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameArtifact(ArtifactBase):
    media_artifact_id: str = ""
    frame_id: str = ""
    timestamp_ms: int = 0
    image_hash: str = ""
    storage_ref: str = ""
    extraction_reason: str = ""
    # Targeted visual evidence must retain its transcript-derived provenance
    # through checkpoints/replay without embedding transcript text or URLs.
    semantic_segment_ids: tuple[str, ...] = ()
    evidence_window_ids: tuple[str, ...] = ()
    planner_version: str = ""
    # Hash of the complete, non-secret planner request.  It joins a materialised
    # frame to its semantic/evidence window even when timestamps collide.
    planner_request_id: str = ""

    def __post_init__(self) -> None:
        if self._identity_profile == _IDENTITY_PROFILE_LEGACY_FRAME and self.planner_request_id:
            raise ValueError("legacy frame identity cannot carry planner_request_id")
        super().__post_init__()


@dataclass(frozen=True)
class OCRArtifact(ArtifactBase):
    frame_artifact_id: str = ""
    frame_id: str = ""
    timestamp_ms: int = 0
    image_hash: str = ""
    semantic_segment_ids: tuple[str, ...] = ()
    evidence_window_ids: tuple[str, ...] = ()
    text: str = ""
    bbox: list[Any] | None = None
    confidence_score: float | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    engine: str = ""
    engine_version: str = ""
    # Non-secret, content-addressed GPU provenance. Defaults preserve legacy
    # CPU artifact deserialization; the bumped OCR checkpoint rejects resume.
    requested_device: str = ""
    actual_device: str = ""
    runtime_identity: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self._identity_profile == _IDENTITY_PROFILE_LEGACY_OCR and (
            self.requested_device or self.actual_device or self.runtime_identity
        ):
            raise ValueError("legacy OCR identity cannot carry GPU runtime provenance")
        super().__post_init__()


@dataclass(frozen=True)
class VisionArtifact(ArtifactBase):
    frame_artifact_id: str = ""
    frame_id: str = ""
    timestamp_ms: int = 0
    image_hash: str = ""
    semantic_segment_ids: tuple[str, ...] = ()
    evidence_window_ids: tuple[str, ...] = ()
    label: str = ""
    labels: list[str] = field(default_factory=list)
    confidence_score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    model_version: str = ""


@dataclass(frozen=True)
class TranscriptVisualCorrectionCandidate:
    """An audit-only OCR candidate; it can never rewrite transcript evidence."""

    kind: str = ""
    original: tuple[str, ...] = ()
    replacement: tuple[str, ...] = ()
    frame_id: str = ""
    transcript_segment_ids: tuple[str, ...] = ()
    ocr_engine: str = ""
    ocr_engine_version: str = ""
    ocr_confidence_score: float | None = None
    status: str = "TRACE_ONLY_REQUIRES_INDEPENDENT_TRANSCRIPT_GROUNDING"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscriptVisualCorrectionCandidate":
        return cls(
            kind=str(value.get("kind") or ""),
            original=tuple(sorted(str(item) for item in value.get("original") or ())),
            replacement=tuple(sorted(str(item) for item in value.get("replacement") or ())),
            frame_id=str(value.get("frame_id") or ""),
            transcript_segment_ids=tuple(sorted(str(item) for item in value.get("transcript_segment_ids") or ())),
            ocr_engine=str(value.get("ocr_engine") or ""),
            ocr_engine_version=str(value.get("ocr_engine_version") or ""),
            ocr_confidence_score=value.get("ocr_confidence_score"),
            status=str(value.get("status") or "TRACE_ONLY_REQUIRES_INDEPENDENT_TRANSCRIPT_GROUNDING"),
        )


@dataclass(frozen=True)
class TranscriptVisualCrosscheckRecord:
    """Compact, typed relation between one frame and its owning transcript."""

    frame_id: str = ""
    frame_artifact_id: str = ""
    timestamp_ms: int = 0
    semantic_segment_ids: tuple[str, ...] = ()
    evidence_window_ids: tuple[str, ...] = ()
    transcript_segment_ids: tuple[str, ...] = ()
    relation: str = "UNKNOWN"
    reason_codes: tuple[str, ...] = ()
    matches: dict[str, tuple[str, ...]] = field(default_factory=dict)
    mismatches: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    confidence_inputs: dict[str, Any] = field(default_factory=dict)
    correction_candidates: tuple[TranscriptVisualCorrectionCandidate, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranscriptVisualCrosscheckRecord":
        raw_matches = value.get("matches") if isinstance(value.get("matches"), dict) else {}
        raw_mismatches = value.get("mismatches") if isinstance(value.get("mismatches"), dict) else {}
        raw_confidence = value.get("confidence_inputs") if isinstance(value.get("confidence_inputs"), dict) else {}
        candidates = value.get("correction_candidates") or ()
        return cls(
            frame_id=str(value.get("frame_id") or ""),
            frame_artifact_id=str(value.get("frame_artifact_id") or ""),
            timestamp_ms=int(value.get("timestamp_ms") or 0),
            semantic_segment_ids=tuple(sorted(str(item) for item in value.get("semantic_segment_ids") or ())),
            evidence_window_ids=tuple(sorted(str(item) for item in value.get("evidence_window_ids") or ())),
            transcript_segment_ids=tuple(sorted(str(item) for item in value.get("transcript_segment_ids") or ())),
            relation=str(value.get("relation") or "UNKNOWN"),
            reason_codes=tuple(sorted(str(item) for item in value.get("reason_codes") or ())),
            matches={
                str(key): tuple(sorted(str(item) for item in items or ())) for key, items in sorted(raw_matches.items())
            },
            mismatches={
                str(key): {
                    str(side): tuple(sorted(str(item) for item in items or ())) for side, items in sorted(parts.items())
                }
                for key, parts in sorted(raw_mismatches.items())
                if isinstance(parts, dict)
            },
            confidence_inputs={
                str(key): tuple(value) if isinstance(value, list) else value
                for key, value in sorted(raw_confidence.items())
            },
            correction_candidates=tuple(
                sorted(
                    (
                        item
                        if isinstance(item, TranscriptVisualCorrectionCandidate)
                        else TranscriptVisualCorrectionCandidate.from_dict(item)
                        for item in candidates
                        if isinstance(item, (dict, TranscriptVisualCorrectionCandidate))
                    ),
                    key=lambda item: (item.kind, item.frame_id, item.original, item.replacement),
                )
            ),
        )


@dataclass(frozen=True)
class TranscriptVisualCrosscheckArtifact(ArtifactBase):
    """Internal visual-admission audit, content-addressed without media locators."""

    transcript_artifact_id: str = ""
    semantic_segment_artifact_id: str = ""
    crosscheck_version: str = ""
    visual_identity: dict[str, str] = field(default_factory=dict)
    relations: tuple[TranscriptVisualCrosscheckRecord, ...] = ()
    eligible_frame_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        records = tuple(
            sorted(
                self.relations,
                key=lambda item: (item.timestamp_ms, item.frame_id, item.frame_artifact_id),
            )
        )
        if any(
            item.relation not in {"SUPPORTS", "CONTRADICTS", "SUPPORTS_DISPLAYED_SECONDARY", "UNRELATED", "UNKNOWN"}
            for item in records
        ):
            raise ValueError("invalid transcript visual crosscheck relation")
        if any(not item.frame_id or not item.frame_artifact_id for item in records):
            raise ValueError("crosscheck relation requires frame identity")
        eligible = tuple(sorted(set(str(item) for item in self.eligible_frame_ids if str(item))))
        allowed = {item.frame_id for item in records if item.relation in {"SUPPORTS", "CONTRADICTS"}}
        if set(eligible) != allowed:
            raise ValueError("eligible visual frames must exactly equal admitted crosscheck relations")
        object.__setattr__(self, "relations", records)
        object.__setattr__(self, "eligible_frame_ids", eligible)
        object.__setattr__(
            self, "visual_identity", {str(key): str(value) for key, value in sorted(self.visual_identity.items())}
        )
        super().__post_init__()


@dataclass(frozen=True)
class ClaimArtifact(ArtifactBase):
    evidence_artifact_id: str = ""
    claims: list[Any] = field(default_factory=list)  # list[FinancialClaim]（claims.py 定义）

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", _stable_membership(self.claims))
        super().__post_init__()


@dataclass(frozen=True)
class ClaimOccurrenceArtifact(ArtifactBase):
    semantic_segment_artifact_id: str = ""
    evidence_artifact_id: str = ""
    occurrence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrence_ids", _stable_membership(self.occurrence_ids))
        super().__post_init__()


@dataclass(frozen=True)
class LifecycleArtifact(ArtifactBase):
    claim_lifecycle_event_ids: list[str] = field(default_factory=list)
    occurrence_lifecycle_event_ids: list[str] = field(default_factory=list)
    lifecycle_business_as_of: datetime | None = None
    lifecycle_knowledge_as_of: datetime | None = None
    policy_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "claim_lifecycle_event_ids",
            _stable_membership(self.claim_lifecycle_event_ids),
        )
        object.__setattr__(
            self,
            "occurrence_lifecycle_event_ids",
            _stable_membership(self.occurrence_lifecycle_event_ids),
        )
        super().__post_init__()


@dataclass(frozen=True)
class VerificationArtifact(ArtifactBase):
    claim_artifact_id: str = ""
    results: list[Any] = field(default_factory=list)  # list[VerificationResult]（claims.py 定义）

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", _stable_membership(self.results, field_identity=False))
        super().__post_init__()


@dataclass(frozen=True)
class KnowledgeArtifact(ArtifactBase):
    verification_artifact_id: str = ""
    knowledge_units: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "knowledge_units", _stable_membership(self.knowledge_units))
        super().__post_init__()


@dataclass(frozen=True)
class SummaryArtifact(ArtifactBase):
    knowledge_artifact_id: str = ""
    core_summary: str = ""


@dataclass(frozen=True)
class SemanticSegmentArtifact(ArtifactBase):
    # Kept as an import-compatible artifact alias; the rich item definition
    # lives in semantic_segment.py to avoid coupling ArtifactBase to domain.
    transcript_artifact_id: str = ""
    segments: list[Any] = field(default_factory=list)
    model_id: str = ""
    prompt_version: str = ""
    segmentation_schema_version: str = "semantic-segment.v1"

    def to_dict(self) -> dict[str, Any]:
        """Keep the unnamespaced semantic artifact representation byte-stable.

        ``derivation_namespace`` was added to segment items to give migration
        replay a distinct immutable identity.  The empty value is the legacy
        and normal-ingestion case, though, so serializing it would alter the
        canonical identity of every pre-existing semantic artifact when it is
        hydrated and checked.  Only a non-empty migration namespace is part
        of the persisted/artifact identity payload.
        """
        payload = super().to_dict()
        for segment in payload["segments"]:
            if not segment.get("derivation_namespace"):
                segment.pop("derivation_namespace", None)
        return payload


ARTIFACT_SLOT_NAMES = (
    "source",
    "media",
    "transcript",
    "evidence",
    "claims",
    "verification",
    "knowledge",
    "summary",
)

# Formal single-value slots introduced by the semantic/temporal chain.  The
# legacy tuple remains stable for callers that enumerate the original slots;
# this unified registry set is the source of truth for all slot operations.
SINGLE_ARTIFACT_SLOT_NAMES = ("semantic_segments", "occurrences", "lifecycle", "transcript_visual_crosscheck")

VISUAL_ARTIFACT_SLOT_NAMES = ("frames", "ocr", "vision")


@dataclass
class ArtifactRegistry:
    """Pipeline 内所有正式 Artifact 的强类型注册表。"""

    source: SourceArtifact | None = None
    media: MediaArtifact | None = None
    transcript: TranscriptArtifact | None = None
    evidence: EvidenceArtifact | None = None
    claims: ClaimArtifact | None = None
    verification: VerificationArtifact | None = None
    knowledge: KnowledgeArtifact | None = None
    summary: SummaryArtifact | None = None
    frames: list[FrameArtifact] = field(default_factory=list)
    ocr: list[OCRArtifact] = field(default_factory=list)
    vision: list[VisionArtifact] = field(default_factory=list)
    semantic_segments: SemanticSegmentArtifact | None = None
    occurrences: ClaimOccurrenceArtifact | None = None
    lifecycle: LifecycleArtifact | None = None
    transcript_visual_crosscheck: TranscriptVisualCrosscheckArtifact | None = None

    def set(self, slot: str, artifact: ArtifactBase) -> None:
        if slot in SINGLE_ARTIFACT_SLOT_NAMES:
            existing = getattr(self, slot)
            if (
                existing is not None
                and existing.artifact_id == artifact.artifact_id
                and existing.content_hash != artifact.content_hash
            ):
                raise ValueError(f"artifact id {artifact.artifact_id} already has a different payload")
            expected = {
                "semantic_segments": SemanticSegmentArtifact,
                "occurrences": ClaimOccurrenceArtifact,
                "lifecycle": LifecycleArtifact,
                "transcript_visual_crosscheck": TranscriptVisualCrosscheckArtifact,
            }[slot]
            if not isinstance(artifact, expected):
                raise TypeError(f"{slot} expects {expected.__name__}")
            setattr(self, slot, artifact)
            return
        for existing_artifact in self.artifacts():
            if existing_artifact.artifact_id == artifact.artifact_id and (
                existing_artifact.content_hash != artifact.content_hash
            ):
                raise ValueError(f"artifact id {artifact.artifact_id} already has a different payload")
        if slot in VISUAL_ARTIFACT_SLOT_NAMES:
            self.add(slot, artifact)
            return
        if slot not in ARTIFACT_SLOT_NAMES:
            raise KeyError(f"unknown artifact slot: {slot}")
        existing = getattr(self, slot)
        if (
            existing is not None
            and existing.artifact_id == artifact.artifact_id
            and (existing.content_hash != artifact.content_hash)
        ):
            raise ValueError(f"artifact id {artifact.artifact_id} already has a different payload")
        setattr(self, slot, artifact)

    def add(self, slot: str, artifact: ArtifactBase) -> None:
        """Append a visual artifact without losing sibling frames/results."""
        if slot not in VISUAL_ARTIFACT_SLOT_NAMES:
            raise KeyError(f"unknown visual artifact slot: {slot}")
        expected = {"frames": FrameArtifact, "ocr": OCRArtifact, "vision": VisionArtifact}[slot]
        if not isinstance(artifact, expected):
            raise TypeError(f"{slot} expects {expected.__name__}")
        for existing_artifact in self.artifacts():
            if existing_artifact.artifact_id == artifact.artifact_id and (
                existing_artifact.content_hash != artifact.content_hash
            ):
                raise ValueError(f"artifact id {artifact.artifact_id} already has a different payload")
        for existing in getattr(self, slot):
            if existing.artifact_id == artifact.artifact_id and existing.content_hash != artifact.content_hash:
                raise ValueError(f"artifact id {artifact.artifact_id} already has a different payload")
            if existing.artifact_id == artifact.artifact_id:
                return
        getattr(self, slot).append(artifact)

    @property
    def visual_artifacts(self) -> tuple[ArtifactBase, ...]:
        return tuple(item for slot in VISUAL_ARTIFACT_SLOT_NAMES for item in getattr(self, slot))

    def get(self, slot: str) -> ArtifactBase | None:
        if slot in SINGLE_ARTIFACT_SLOT_NAMES:
            return getattr(self, slot)
        if slot in VISUAL_ARTIFACT_SLOT_NAMES:
            return getattr(self, slot)  # type: ignore[return-value]
        if slot not in ARTIFACT_SLOT_NAMES:
            raise KeyError(f"unknown artifact slot: {slot}")
        return getattr(self, slot)

    def artifact_ids(self) -> dict[str, str]:
        result = {
            slot: artifact.artifact_id for slot in ARTIFACT_SLOT_NAMES if (artifact := getattr(self, slot)) is not None
        }
        result.update(
            {
                slot: artifact.artifact_id
                for slot in SINGLE_ARTIFACT_SLOT_NAMES
                if (artifact := getattr(self, slot)) is not None
            }
        )
        for slot in VISUAL_ARTIFACT_SLOT_NAMES:
            visual = sorted(getattr(self, slot), key=lambda item: item.artifact_id)
            result.update({f"{slot}:{index}": artifact.artifact_id for index, artifact in enumerate(visual)})
        return result

    def artifacts(self) -> list[ArtifactBase]:
        base = [artifact for slot in ARTIFACT_SLOT_NAMES if (artifact := getattr(self, slot)) is not None]
        extras = [artifact for slot in SINGLE_ARTIFACT_SLOT_NAMES if (artifact := getattr(self, slot)) is not None]
        return base + list(self.visual_artifacts) + extras


def serialize_artifact(artifact: ArtifactBase) -> dict[str, Any]:
    payload = artifact.to_dict()
    payload["content_hash"] = artifact.content_hash
    return payload


def deserialize_artifact(payload: dict[str, Any]) -> ArtifactBase:
    """按 artifact_type 还原 Artifact（用于 checkpoint/replay）。"""
    artifact_type = payload.get("artifact_type")
    cls = _TYPE_REGISTRY.get(str(artifact_type))
    if cls is None:
        raise ValueError(f"unknown artifact_type: {artifact_type}")
    kwargs = {}
    for key, value in payload.items():
        if key == "segments" and cls is TranscriptArtifact:
            value = [TranscriptSegmentItem(**item) if isinstance(item, dict) else item for item in value]
        if key == "segments" and cls is SemanticSegmentArtifact:
            from stock_content.domain.semantic_segment import SemanticSegmentItem

            value = [SemanticSegmentItem(**item) if isinstance(item, dict) else item for item in value]
        if key == "relations" and cls is TranscriptVisualCrosscheckArtifact:
            value = tuple(
                item
                if isinstance(item, TranscriptVisualCrosscheckRecord)
                else TranscriptVisualCrosscheckRecord.from_dict(item)
                for item in value
            )
        if key == "evidences" and cls is EvidenceArtifact:
            value = [EvidenceItem(**item) if isinstance(item, dict) else item for item in value]
        if key == "results" and cls is VerificationArtifact:
            from stock_content.domain.claims import VerificationArtifactEntry, VerificationResult

            converted = []
            for item in value:
                if isinstance(item, dict) and (
                    item.get("verification_id") is not None
                    or item.get("verification_job_id") is not None
                    or item.get("provider") is not None
                ):
                    # ``VerificationArtifactEntry.model_dump`` deliberately
                    # flattens the nested result for legacy consumers.  Put
                    # that exact result back during hydration so the
                    # content-addressed artifact hash is stable across a
                    # persist/reload cycle.
                    entry_payload = dict(item)
                    # Pending entries intentionally carry only the exact job
                    # reference; hydrating a synthetic pending result changes
                    # the content-addressed artifact hash on reload.
                    entry_payload["result"] = (
                        None
                        if item.get("status") == "VERIFICATION_PENDING"
                        and "verification_rule_version" not in item
                        and "available_at" not in item
                        else VerificationResult.model_validate(item)
                    )
                    converted.append(VerificationArtifactEntry.model_validate(entry_payload))
                else:
                    converted.append(VerificationResult.model_validate(item) if isinstance(item, dict) else item)
            value = converted
        if key == "created_at" and isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if key in {
            "lifecycle_business_as_of",
            "lifecycle_knowledge_as_of",
        } and isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if key in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            kwargs[key] = value
    stored_content_hash = bool(kwargs.get("content_hash"))
    artifact = cls(**kwargs)
    identity_profile = _identity_profile_from_serialized_payload(payload, artifact_type)
    if identity_profile != _IDENTITY_PROFILE_CURRENT:
        object.__setattr__(artifact, "_identity_profile", identity_profile)
        # Serialized production artifacts always carry ``content_hash``.  Be
        # defensive for callers hydrating an old in-memory fixture without it:
        # once the historical profile is known, compute the correct identity.
        if not stored_content_hash:
            object.__setattr__(artifact, "content_hash", content_hash_of(artifact_identity_payload(artifact)))
    return artifact


def _identity_profile_from_serialized_payload(payload: dict[str, Any], artifact_type: Any) -> str:
    """Infer the only compatible historical identity form from field presence.

    ``artifact.v1`` cannot be bumped retroactively.  Field *presence* is the
    stable discriminator: a new serializer writes all additive fields, even
    if their values are empty, whereas old rows do not contain them at all.
    """
    if artifact_type == "frame" and "planner_request_id" not in payload:
        return _IDENTITY_PROFILE_LEGACY_FRAME
    if artifact_type == "ocr" and not {
        "requested_device",
        "actual_device",
        "runtime_identity",
    }.intersection(payload):
        return _IDENTITY_PROFILE_LEGACY_OCR
    return _IDENTITY_PROFILE_CURRENT


_TYPE_REGISTRY: dict[str, type[ArtifactBase]] = {
    "source": SourceArtifact,
    "media": MediaArtifact,
    "transcript": TranscriptArtifact,
    "evidence": EvidenceArtifact,
    "claims": ClaimArtifact,
    "verification": VerificationArtifact,
    "knowledge": KnowledgeArtifact,
    "summary": SummaryArtifact,
    "frame": FrameArtifact,
    "ocr": OCRArtifact,
    "vision": VisionArtifact,
    "transcript_visual_crosscheck": TranscriptVisualCrosscheckArtifact,
    "semantic_segments": SemanticSegmentArtifact,
    "occurrences": ClaimOccurrenceArtifact,
    "lifecycle": LifecycleArtifact,
}


def transcript_segment_identity_payload(
    *,
    media_artifact_id: str,
    asr_model: str,
    asr_model_version: str,
    segment_index: int,
    start_ms: int,
    end_ms: int,
    raw_text: str,
) -> dict[str, Any]:
    """The only authoritative transcript segment identity inputs."""
    if not media_artifact_id or not asr_model or not asr_model_version:
        raise ValueError("media artifact and ASR manifest are required for segment identity")
    return {
        "media_artifact_id": media_artifact_id,
        "asr_model": asr_model,
        "asr_model_version": asr_model_version,
        "segment_index": segment_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "raw_text": raw_text,
    }


def transcript_segment_id(
    *,
    media_artifact_id: str,
    asr_model: str,
    asr_model_version: str,
    segment_index: int,
    start_ms: int,
    end_ms: int,
    raw_text: str,
) -> str:
    payload = transcript_segment_identity_payload(
        media_artifact_id=media_artifact_id,
        asr_model=asr_model,
        asr_model_version=asr_model_version,
        segment_index=segment_index,
        start_ms=start_ms,
        end_ms=end_ms,
        raw_text=raw_text,
    )
    return "trseg_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


segment_id_of = transcript_segment_id


def legacy_transcript_segment_id(
    segment_index: int,
    start_seconds: float,
    end_seconds: float,
    raw_text: str,
    *,
    legacy_namespace: str = "",
) -> str:
    payload = {
        "legacy_namespace": legacy_namespace,
        "segment_index": segment_index,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "raw_text": raw_text,
    }
    return "legacy_trseg_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_artifact_id(artifact_type: str, payload: Any) -> str:
    return f"{artifact_type}-{content_hash_of(payload)[:32]}"


def artifact_id_of(artifact: ArtifactBase) -> str:
    """Canonical content address for an already constructed artifact."""
    return make_artifact_id(artifact.artifact_type, artifact_identity_payload(artifact))


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_SLOT_NAMES",
    "SINGLE_ARTIFACT_SLOT_NAMES",
    "VISUAL_ARTIFACT_SLOT_NAMES",
    "ArtifactBase",
    "ArtifactRegistry",
    "ClaimArtifact",
    "ClaimOccurrenceArtifact",
    "EvidenceArtifact",
    "EvidenceItem",
    "KnowledgeArtifact",
    "MediaArtifact",
    "FrameArtifact",
    "OCRArtifact",
    "VisionArtifact",
    "TranscriptVisualCorrectionCandidate",
    "TranscriptVisualCrosscheckArtifact",
    "TranscriptVisualCrosscheckRecord",
    "SourceArtifact",
    "SummaryArtifact",
    "LifecycleArtifact",
    "SemanticSegmentArtifact",
    "TranscriptArtifact",
    "TranscriptSegmentItem",
    "transcript_segment_id",
    "transcript_segment_identity_payload",
    "legacy_transcript_segment_id",
    "segment_id_of",
    "VerificationArtifact",
    "canonical_json",
    "content_hash_of",
    "artifact_identity_payload",
    "artifact_id_of",
    "deserialize_artifact",
    "make_artifact_id",
    "serialize_artifact",
]

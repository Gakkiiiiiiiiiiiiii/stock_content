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


def canonical_json(payload: Any) -> str:
    """canonical JSON：key 排序、紧凑分隔符、统一序列化。"""
    def _default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return str(value)

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)


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
    return {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_id", "created_at", "content_hash"}
    }


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

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                content_hash_of(artifact_identity_payload(self)),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
    segment_index: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    text: str = ""
    confidence: float | None = None
    speaker_id: str | None = None


@dataclass(frozen=True)
class TranscriptArtifact(ArtifactBase):
    media_artifact_id: str = ""
    language: str | None = None
    segments: list[TranscriptSegmentItem] = field(default_factory=list)
    asr_model: str = "unknown"
    asr_model_version: str = "unknown"


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


@dataclass(frozen=True)
class OCRArtifact(ArtifactBase):
    frame_artifact_id: str = ""
    text: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    engine: str = ""
    engine_version: str = ""


@dataclass(frozen=True)
class VisionArtifact(ArtifactBase):
    frame_artifact_id: str = ""
    label: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    model_version: str = ""


@dataclass(frozen=True)
class ClaimArtifact(ArtifactBase):
    evidence_artifact_id: str = ""
    claims: list[Any] = field(default_factory=list)  # list[FinancialClaim]（claims.py 定义）


@dataclass(frozen=True)
class VerificationArtifact(ArtifactBase):
    claim_artifact_id: str = ""
    results: list[Any] = field(default_factory=list)  # list[VerificationResult]（claims.py 定义）


@dataclass(frozen=True)
class KnowledgeArtifact(ArtifactBase):
    verification_artifact_id: str = ""
    knowledge_units: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryArtifact(ArtifactBase):
    knowledge_artifact_id: str = ""
    core_summary: str = ""


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

    def set(self, slot: str, artifact: ArtifactBase) -> None:
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
        if existing is not None and existing.artifact_id == artifact.artifact_id and (
            existing.content_hash != artifact.content_hash
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
        if slot in VISUAL_ARTIFACT_SLOT_NAMES:
            return getattr(self, slot)  # type: ignore[return-value]
        if slot not in ARTIFACT_SLOT_NAMES:
            raise KeyError(f"unknown artifact slot: {slot}")
        return getattr(self, slot)

    def artifact_ids(self) -> dict[str, str]:
        result = {
            slot: artifact.artifact_id
            for slot in ARTIFACT_SLOT_NAMES
            if (artifact := getattr(self, slot)) is not None
        }
        for slot in VISUAL_ARTIFACT_SLOT_NAMES:
            visual = sorted(getattr(self, slot), key=lambda item: item.artifact_id)
            result.update({f"{slot}:{index}": artifact.artifact_id for index, artifact in enumerate(visual)})
        return result

    def artifacts(self) -> list[ArtifactBase]:
        return [artifact for slot in ARTIFACT_SLOT_NAMES if (artifact := getattr(self, slot)) is not None] + list(
            self.visual_artifacts
        )


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
        if key == "evidences" and cls is EvidenceArtifact:
            value = [EvidenceItem(**item) if isinstance(item, dict) else item for item in value]
        if key == "results" and cls is VerificationArtifact:
            from stock_content.domain.claims import VerificationResult

            value = [VerificationResult.model_validate(item) if isinstance(item, dict) else item for item in value]
        if key == "created_at" and isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if key in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            kwargs[key] = value
    return cls(**kwargs)


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
}


def make_artifact_id(artifact_type: str, payload: Any) -> str:
    return f"{artifact_type}-{content_hash_of(payload)[:32]}"


def artifact_id_of(artifact: ArtifactBase) -> str:
    """Canonical content address for an already constructed artifact."""
    return make_artifact_id(artifact.artifact_type, artifact_identity_payload(artifact))


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_SLOT_NAMES",
    "VISUAL_ARTIFACT_SLOT_NAMES",
    "ArtifactBase",
    "ArtifactRegistry",
    "ClaimArtifact",
    "EvidenceArtifact",
    "EvidenceItem",
    "KnowledgeArtifact",
    "MediaArtifact",
    "FrameArtifact",
    "OCRArtifact",
    "VisionArtifact",
    "SourceArtifact",
    "SummaryArtifact",
    "TranscriptArtifact",
    "TranscriptSegmentItem",
    "VerificationArtifact",
    "canonical_json",
    "content_hash_of",
    "artifact_identity_payload",
    "artifact_id_of",
    "deserialize_artifact",
    "make_artifact_id",
    "serialize_artifact",
]

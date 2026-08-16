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
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def content_hash_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k not in {"created_at"}}
        return content_hash_of(payload)


@dataclass(frozen=True)
class SourceArtifact(ArtifactBase):
    source_type: str = ""
    source_ref: str = ""
    source_content_hash: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MediaArtifact(ArtifactBase):
    source_artifact_id: str = ""
    media_uri: str = ""
    duration_ms: int | None = None
    audio_hash: str | None = None
    video_hash: str | None = None


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


@dataclass(frozen=True)
class EvidenceArtifact(ArtifactBase):
    transcript_artifact_id: str = ""
    evidences: list[EvidenceItem] = field(default_factory=list)


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

    def set(self, slot: str, artifact: ArtifactBase) -> None:
        if slot not in ARTIFACT_SLOT_NAMES:
            raise KeyError(f"unknown artifact slot: {slot}")
        setattr(self, slot, artifact)

    def get(self, slot: str) -> ArtifactBase | None:
        if slot not in ARTIFACT_SLOT_NAMES:
            raise KeyError(f"unknown artifact slot: {slot}")
        return getattr(self, slot)

    def artifact_ids(self) -> dict[str, str]:
        return {
            slot: artifact.artifact_id
            for slot in ARTIFACT_SLOT_NAMES
            if (artifact := getattr(self, slot)) is not None
        }

    def artifacts(self) -> list[ArtifactBase]:
        return [artifact for slot in ARTIFACT_SLOT_NAMES if (artifact := getattr(self, slot)) is not None]


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
        if key == "content_hash":
            continue
        if key == "segments" and cls is TranscriptArtifact:
            value = [TranscriptSegmentItem(**item) if isinstance(item, dict) else item for item in value]
        if key == "evidences" and cls is EvidenceArtifact:
            value = [EvidenceItem(**item) if isinstance(item, dict) else item for item in value]
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
}


def make_artifact_id(artifact_type: str, payload: Any) -> str:
    return f"{artifact_type}-{content_hash_of(payload)[:16]}"


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_SLOT_NAMES",
    "ArtifactBase",
    "ArtifactRegistry",
    "ClaimArtifact",
    "EvidenceArtifact",
    "EvidenceItem",
    "KnowledgeArtifact",
    "MediaArtifact",
    "SourceArtifact",
    "SummaryArtifact",
    "TranscriptArtifact",
    "TranscriptSegmentItem",
    "VerificationArtifact",
    "canonical_json",
    "content_hash_of",
    "deserialize_artifact",
    "make_artifact_id",
    "serialize_artifact",
]

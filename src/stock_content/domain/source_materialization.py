"""Source-ingestion values with a hard boundary between public and secret data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, SecretStr


class CredentialReference(BaseModel):
    credential_ref: str
    provider: Literal["file-secret", "vault", "k8s-secret"]


class MediaStream(BaseModel):
    stream_id: str
    kind: Literal["video", "audio", "muxed", "hls", "dash"]
    url: SecretStr
    headers: dict[str, SecretStr] = {}
    expires_at: datetime | None = None
    codec: str | None = None
    bitrate: int | None = None
    width: int | None = None
    height: int | None = None


class SubtitleTrack(BaseModel):
    track_id: str
    language: str
    source: Literal["official", "automatic"]
    format: str
    url: SecretStr
    headers: dict[str, SecretStr] = {}


class ResolvedSource(BaseModel):
    source_type: str
    canonical_source_ref: str
    # A stable, secret-free public page projection.  It is intentionally
    # separate from ``canonical_source_ref`` because Xiaoe queues a compact
    # product/lesson identity rather than a browser URL.
    canonical_url: str | None = None
    source_identity_hash: str
    platform_id: str
    part_id: str | None = None
    title: str
    author: str | None = None
    published_at: datetime | None = None
    duration_seconds: float | None = None


class SourceMaterialization(BaseModel):
    public: ResolvedSource
    streams: list[MediaStream]
    subtitles: list[SubtitleTrack]
    credential_ref_hash: str | None = None
    requires_reresolve: bool = False


@dataclass(frozen=True, slots=True)
class MaterializedSubtitleCue:
    """A parsed subtitle cue safe to retain only in the worker workspace.

    The resolver's URL, headers and credential reference intentionally do not
    cross this boundary.  The raw/normalised hashes provide deterministic
    lineage without turning a local subtitle file into a durable locator.
    """

    start_ms: int
    end_ms: int
    raw_text: str
    normalized_text: str
    cue_hash: str


@dataclass(frozen=True, slots=True)
class MaterializedSubtitleTrack:
    """Typed, secret-free subtitle output from a source materializer."""

    track_id: str
    language: str
    source: Literal["official", "automatic"]
    raw_sha256: str
    normalized_sha256: str
    artifact_id: str
    cues: tuple[MaterializedSubtitleCue, ...]


@dataclass(frozen=True, slots=True)
class ContentIngestionCommand:
    """Canonical, pre-resolution request.

    This is the persisted-command boundary: it carries hashes only, never the
    credential reference or signed locator submitted at HTTP ingress.
    """

    source_type: str
    canonical_source_ref: str
    part: int | None
    transcript_policy: str
    options: dict[str, object]
    idempotency_key: str | None = None
    credential_ref_hash: str | None = None
    locator_secret_hash: str | None = None
    trace_id: str | None = None
    decision_id: str | None = None


__all__ = [
    "ContentIngestionCommand", "CredentialReference", "MediaStream", "ResolvedSource",
    "MaterializedSubtitleCue", "MaterializedSubtitleTrack", "SourceMaterialization", "SubtitleTrack",
]

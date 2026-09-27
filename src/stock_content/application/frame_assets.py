"""Read-only, fail-closed projection of sealed frame assets."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

from stock_content.application.replay.integrity import ReplayIntegrityMixin
from stock_content.domain.artifacts import (
    ClaimOccurrenceArtifact,
    FrameArtifact,
    KnowledgeVisualEvidenceArtifact,
    MediaArtifact,
    OCRArtifact,
    SemanticSegmentArtifact,
    SourceArtifact,
    TranscriptArtifact,
    TranscriptVisualCrosscheckArtifact,
    VisionArtifact,
    artifact_id_of,
    artifact_identity_payload,
    content_hash_of,
)
from stock_content.domain.lineage import (
    compute_artifact_root_hash,
    compute_content_snapshot_id,
    snapshot_identity_payload,
)


class FrameAssetError(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def _conflict() -> FrameAssetError:
    return FrameAssetError(409, "FRAME_ASSET_INTEGRITY_ERROR")


_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _verified_artifact(repository: Any, artifact_id: str, expected_type: type) -> Any:
    try:
        artifact = repository.get(artifact_id)
        if artifact is None or not isinstance(artifact, expected_type):
            raise _conflict()
        if repository.verify(artifact_id) is not True:
            raise _conflict()
        if artifact.artifact_id != artifact_id or artifact_id_of(artifact) != artifact_id:
            raise _conflict()
        if content_hash_of(artifact_identity_payload(artifact)) != artifact.content_hash:
            raise _conflict()
        return artifact
    except FrameAssetError:
        raise
    except Exception as exc:
        raise _conflict() from exc


def _verify_snapshot(snapshot: Any) -> dict[str, str]:
    try:
        mapping = dict(snapshot.artifact_ids)
        if not mapping or not snapshot.artifact_root_hash:
            raise _conflict()
        if compute_artifact_root_hash(mapping) != snapshot.artifact_root_hash:
            raise _conflict()
        if mapping.get("source") != snapshot.source_artifact_id:
            raise _conflict()
        fields = {
            name: getattr(snapshot, name)
            for name in (
                "source_content_hash", "source_artifact_id", "artifact_root_hash", "pipeline_version",
                "parser_version", "asr_model", "asr_model_version", "vision_model", "llm_model",
                "prompt_bundle_version", "entity_alias_version", "verification_policy_version",
                "quant_market_snapshot_ids", "code_sha", "config_hash", "producer_manifest",
                "model_versions", "prompt_versions", "configuration", "external_snapshots",
                "policy_versions", "snapshot_kind", "parent_snapshot_id", "supersedes_snapshot_id",
            )
        }
        identity = snapshot_identity_payload(**fields)
        if snapshot.content_snapshot_id != f"cs-{compute_content_snapshot_id(identity)[:32]}":
            raise _conflict()
        return mapping
    except FrameAssetError:
        raise
    except Exception as exc:
        raise _conflict() from exc


def _safe_summary(value: Any) -> str:
    text = str(value or "")
    if "/" in text or "\\" in text or "?" in text:
        return "[redacted]"
    return text


def _scope_metadata(
    service: Any, mapping: dict[str, str], packet: KnowledgeVisualEvidenceArtifact,
    frame: FrameArtifact, repository: Any,
) -> list[dict[str, Any]]:
    occurrence_artifact = _verified_artifact(repository, mapping["occurrences"], ClaimOccurrenceArtifact)
    crosscheck = _verified_artifact(
        repository, mapping["transcript_visual_crosscheck"], TranscriptVisualCrosscheckArtifact
    )
    semantic_artifact = _verified_artifact(repository, mapping["semantic_segments"], SemanticSegmentArtifact)
    transcript = _verified_artifact(repository, mapping["transcript"], TranscriptArtifact)
    if (
        mapping["transcript_visual_crosscheck"] not in packet.parent_artifact_ids
        or transcript.media_artifact_id != mapping.get("media")
        or semantic_artifact.transcript_artifact_id != transcript.artifact_id
        or transcript.artifact_id not in semantic_artifact.parent_artifact_ids
        or crosscheck.transcript_artifact_id != transcript.artifact_id
        or occurrence_artifact.semantic_segment_artifact_id != semantic_artifact.artifact_id
        or semantic_artifact.artifact_id not in occurrence_artifact.parent_artifact_ids
        or crosscheck.semantic_segment_artifact_id != semantic_artifact.artifact_id
        or transcript.artifact_id not in crosscheck.parent_artifact_ids
        or semantic_artifact.artifact_id not in crosscheck.parent_artifact_ids
        or frame.artifact_id not in crosscheck.parent_artifact_ids
        or packet.artifact_id not in occurrence_artifact.parent_artifact_ids
        or crosscheck.artifact_id not in occurrence_artifact.parent_artifact_ids
        or service._occurrence_repository is None
    ):
        raise _conflict()
    semantic_ids = {
        str(item.get("semantic_segment_id") if isinstance(item, dict) else getattr(item, "semantic_segment_id", ""))
        for item in semantic_artifact.segments
    }
    active_ids = set(occurrence_artifact.occurrence_ids)
    active_artifact_ids = set(mapping.values())
    scopes: list[dict[str, Any]] = []
    for occurrence in packet.occurrence_packets:
        if not isinstance(occurrence, dict):
            raise _conflict()
        knowledge_id = str(occurrence.get("knowledge_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", knowledge_id):
            raise _conflict()
        if knowledge_id != occurrence.get("occurrence_id") or knowledge_id not in active_ids:
            raise _conflict()
        try:
            row = service._occurrence_repository.get(knowledge_id)
        except Exception as exc:
            raise _conflict() from exc
        semantic_id = str(getattr(row, "semantic_segment_id", "") or "")
        if (
            row is None or getattr(row, "occurrence_id", "") != knowledge_id
            or getattr(row, "source_artifact_id", "") != mapping.get("source")
            or getattr(row, "transcript_artifact_id", "") != transcript.artifact_id
            or semantic_id not in semantic_ids
            or (getattr(row, "provenance", {}) or {}).get("visual_evidence") != occurrence
        ):
            raise _conflict()
        for window in occurrence.get("windows") or ():
            if not isinstance(window, dict):
                raise _conflict()
            window_id = str(window.get("evidence_window_id") or "")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", window_id):
                raise _conflict()
            for entry in window.get("frames") or ():
                if not isinstance(entry, dict):
                    raise _conflict()
                if entry.get("frame_id") != frame.frame_id:
                    continue
                if (
                    window_id not in frame.evidence_window_ids
                    or entry.get("frame_artifact_id") != frame.artifact_id
                    or entry.get("frame_artifact_hash") != f"sha256:{frame.content_hash}"
                    or entry.get("image_hash") != frame.image_hash
                    or entry.get("timestamp_ms") != frame.timestamp_ms
                ):
                    raise _conflict()
                scoped = [
                    relation for relation in crosscheck.relations
                    if relation.frame_id == frame.frame_id
                    and tuple(relation.evidence_window_ids) == (window_id,)
                    and tuple(relation.semantic_segment_ids) == (semantic_id,)
                ]
                if any(
                    relation.frame_artifact_id != frame.artifact_id or relation.timestamp_ms != frame.timestamp_ms
                    for relation in scoped
                ):
                    raise _conflict()
                packet_status = str(occurrence.get("status") or "")
                mixed_review = packet_status == "HUMAN_REVIEW_REQUIRED" and any(
                    isinstance(other, dict) and other.get("status") == "HUMAN_REVIEW_REQUIRED"
                    for other in occurrence.get("windows") or ()
                )
                gap_reason = str(window.get("reason") or "")
                lawful_unresolved_gap = (
                    window.get("status") == "GAP"
                    and (packet_status == "GAP" or mixed_review)
                    and entry.get("relation") == "UNKNOWN"
                    and (
                        (gap_reason == "CROSSCHECK_MISSING" and len(scoped) == 0)
                        or (gap_reason == "CROSSCHECK_SCOPE_AMBIGUOUS" and len(scoped) != 1)
                    )
                )
                if not lawful_unresolved_gap and (
                    len(scoped) != 1 or scoped[0].frame_artifact_id != frame.artifact_id
                    or scoped[0].timestamp_ms != frame.timestamp_ms
                    or scoped[0].relation != entry.get("relation")
                ):
                    raise _conflict()
                loaded: dict[str, Any] = {frame.artifact_id: frame}
                for modality, expected_type in (("ocr", OCRArtifact), ("vision", VisionArtifact)):
                    for result in entry.get(modality) or ():
                        if not isinstance(result, dict):
                            raise _conflict()
                        artifact_id = str(result.get("artifact_id") or "")
                        if artifact_id not in active_artifact_ids:
                            raise _conflict()
                        artifact = _verified_artifact(repository, artifact_id, expected_type)
                        if (
                            window_id not in artifact.evidence_window_ids
                            or artifact.frame_artifact_id != frame.artifact_id
                            or frame.artifact_id not in artifact.parent_artifact_ids
                            or artifact_id not in crosscheck.parent_artifact_ids
                        ):
                            raise _conflict()
                        loaded[artifact_id] = artifact
                try:
                    ReplayIntegrityMixin._validate_visual_packet_frame(
                        packet=occurrence, frame=entry, evidence_window_id=window_id,
                        semantic_segment_id=semantic_id, crosscheck_artifact=crosscheck,
                        visual_parent_ids=set(packet.parent_artifact_ids), loaded=loaded,
                        artifact_id=packet.artifact_id, window_status=str(window.get("status") or ""),
                        window_reason=str(window.get("reason") or ""),
                    )
                except Exception as exc:
                    raise _conflict() from exc
                def projection(items: Any) -> list[dict[str, Any]]:
                    if not isinstance(items, list):
                        raise _conflict()
                    if any(not isinstance(item, dict) for item in items):
                        raise _conflict()
                    for item in items:
                        confidence = item.get("confidence")
                        if confidence is not None and (
                            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                            or not math.isfinite(confidence) or not 0 <= confidence <= 1
                        ):
                            raise _conflict()
                    return [
                        {"summary": _safe_summary(item.get("summary")), "confidence": item.get("confidence")}
                        for item in items
                    ]

                relation = str(entry.get("relation") or "UNKNOWN")
                if relation not in {
                    "SUPPORTS", "SUPPORTS_DISPLAYED_SECONDARY", "SUPPORTS_DISPLAYED_MENTION",
                    "ENTITY_CORRECTION_PENDING", "CONTRADICTS", "UNRELATED", "UNKNOWN",
                }:
                    raise _conflict()
                scopes.append({
                    "knowledge_id": knowledge_id,
                    "evidence_window_id": window_id,
                    "relation": relation,
                    "ocr": projection(entry.get("ocr") or []),
                    "vision": projection(entry.get("vision") or []),
                })
    keys = [(item["knowledge_id"], item["evidence_window_id"]) for item in scopes]
    if len(keys) != len(set(keys)):
        raise _conflict()
    return sorted(scopes, key=lambda item: (item["knowledge_id"], item["evidence_window_id"]))


def read_frame_asset(service: Any, snapshot_id: str, frame_id: str) -> tuple[dict[str, Any], bytes, str]:
    """Return metadata and the exact bytes whose digest was verified."""
    if not re.fullmatch(r"cs-[a-f0-9]{32}", snapshot_id) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", frame_id):
        raise FrameAssetError(404, "FRAME_ASSET_NOT_FOUND")
    try:
        snapshot = service._snapshots.get(snapshot_id)
    except Exception as exc:
        raise _conflict() from exc
    if snapshot is None:
        raise FrameAssetError(404, "FRAME_ASSET_NOT_FOUND")
    mapping = _verify_snapshot(snapshot)
    repository = service._artifact_repository
    if repository is None:
        raise _conflict()
    source_id = mapping.get("source")
    media_id = mapping.get("media")
    if not source_id or not media_id:
        raise _conflict()
    source = _verified_artifact(repository, source_id, SourceArtifact)
    media = _verified_artifact(repository, media_id, MediaArtifact)
    if (
        media.source_artifact_id != source.artifact_id
        or source.source_content_hash != snapshot.source_content_hash
        or media.source_artifact_id not in media.parent_artifact_ids
    ):
        raise _conflict()
    frame_ids = [value for key, value in mapping.items() if key.startswith("frames:")]
    if not frame_ids:
        raise FrameAssetError(404, "FRAME_ASSET_NOT_FOUND")
    matches: list[FrameArtifact] = []
    for artifact_id in frame_ids:
        frame = _verified_artifact(repository, artifact_id, FrameArtifact)
        if frame.frame_id == frame_id:
            matches.append(frame)
    if not matches:
        raise FrameAssetError(404, "FRAME_ASSET_NOT_FOUND")
    if len(matches) != 1:
        raise _conflict()
    frame = matches[0]
    if (
        frame.producer_stage != "knowledge_frame"
        or not frame.planner_request_id or not frame.planner_version
        or not frame.evidence_window_ids or not frame.media_artifact_id
        or frame.media_artifact_id not in frame.parent_artifact_ids
        or mapping.get("media") != frame.media_artifact_id
    ):
        raise _conflict()
    packet_id = mapping.get("knowledge_visual_evidence")
    if not packet_id:
        raise _conflict()
    packet = _verified_artifact(repository, packet_id, KnowledgeVisualEvidenceArtifact)
    if frame.artifact_id not in packet.parent_artifact_ids:
        raise _conflict()
    try:
        scopes = _scope_metadata(service, mapping, packet, frame, repository)
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise _conflict() from exc
    if not scopes:
        raise _conflict()

    configured = os.getenv("CONTENT_RAW_STORAGE_DIR", "").strip()
    if not configured or not re.fullmatch(r"[a-fA-F0-9]{64}", frame.image_hash):
        raise _conflict()
    root = Path(configured) / "frames"
    try:
        candidate = Path(frame.storage_ref)
        if (
            not root.is_absolute() or not candidate.is_absolute()
            or candidate.parent != root or root.is_symlink() or candidate.is_symlink()
        ):
            raise _conflict()
        resolved_root = root.resolve(strict=True)
        if resolved_root != root.absolute():
            raise _conflict()
        resolved = candidate.resolve(strict=True)
        if resolved.parent != resolved_root or not resolved.is_file():
            raise _conflict()
        allowed_names = {f"{frame.image_hash.lower()}.{extension}" for extension in ("jpg", "jpeg", "png")}
        if candidate.name.lower() not in allowed_names:
            raise _conflict()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        try:
            opened = os.fstat(descriptor)
            linked = candidate.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
                or opened.st_size > _MAX_IMAGE_BYTES
                or candidate.resolve(strict=True).parent != resolved_root
            ):
                raise _conflict()
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                data = handle.read(_MAX_IMAGE_BYTES + 1)
            if len(data) > _MAX_IMAGE_BYTES:
                raise _conflict()
        finally:
            os.close(descriptor)
    except FrameAssetError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise _conflict() from exc
    if hashlib.sha256(data).hexdigest().lower() != frame.image_hash.lower():
        raise _conflict()
    if data.startswith(b"\xff\xd8\xff") and candidate.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n") and candidate.suffix.lower() == ".png":
        media_type = "image/png"
    else:
        raise _conflict()
    metadata = {
        "frame_id": frame.frame_id,
        "timestamp_ms": frame.timestamp_ms,
        "image_hash": frame.image_hash,
        "scopes": scopes,
    }
    return metadata, data, media_type

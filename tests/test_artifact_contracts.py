"""Artifact 强类型契约测试（详细修改方案 §4 P0-1）。"""
from __future__ import annotations

import pytest

from stock_content.domain.artifacts import (
    ARTIFACT_SLOT_NAMES,
    ArtifactRegistry,
    ClaimArtifact,
    EvidenceArtifact,
    EvidenceItem,
    KnowledgeArtifact,
    MediaArtifact,
    SourceArtifact,
    SummaryArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    VerificationArtifact,
    content_hash_of,
    deserialize_artifact,
    make_artifact_id,
    serialize_artifact,
)


def _sample_artifacts():
    return [
        SourceArtifact(
            artifact_id="source-1",
            artifact_type="source",
            source_type="bilibili",
            source_ref="BV1",
            source_content_hash="abc",
            source_metadata={"title": "t"},
        ),
        MediaArtifact(artifact_id="media-1", artifact_type="media", source_artifact_id="source-1", media_uri="file://v"),
        TranscriptArtifact(
            artifact_id="transcript-1",
            artifact_type="transcript",
            media_artifact_id="media-1",
            segments=[TranscriptSegmentItem(segment_index=0, text="营收三十亿", confidence=0.9)],
            asr_model="faster-whisper",
            asr_model_version="large-v3",
        ),
        EvidenceArtifact(
            artifact_id="evidence-1",
            artifact_type="evidence",
            transcript_artifact_id="transcript-1",
            evidences=[EvidenceItem(evidence_id="ev-1", source_type="ASR", evidence_text="营收三十亿")],
        ),
        ClaimArtifact(artifact_id="claim-1", artifact_type="claims", evidence_artifact_id="evidence-1", claims=[]),
        VerificationArtifact(
            artifact_id="verification-1", artifact_type="verification", claim_artifact_id="claim-1", results=[]
        ),
        KnowledgeArtifact(
            artifact_id="knowledge-1", artifact_type="knowledge", verification_artifact_id="verification-1"
        ),
        SummaryArtifact(artifact_id="summary-1", artifact_type="summary", core_summary="核心总结"),
    ]


def test_all_artifact_types_serialize_deserialize_roundtrip():
    for artifact in _sample_artifacts():
        payload = serialize_artifact(artifact)
        assert payload["content_hash"]
        restored = deserialize_artifact(payload)
        assert type(restored) is type(artifact)
        assert restored.artifact_id == artifact.artifact_id
        assert restored.content_hash == artifact.content_hash


def test_content_hash_is_deterministic_and_sensitive():
    first = SourceArtifact(
        artifact_id="s", artifact_type="source", source_type="bilibili", source_ref="BV1", source_content_hash="h"
    )
    second = SourceArtifact(
        artifact_id="s", artifact_type="source", source_type="bilibili", source_ref="BV1", source_content_hash="h"
    )
    changed = SourceArtifact(
        artifact_id="s", artifact_type="source", source_type="bilibili", source_ref="BV2", source_content_hash="h"
    )
    assert first.content_hash == second.content_hash
    assert first.content_hash != changed.content_hash


def test_unknown_artifact_type_rejected():
    with pytest.raises(ValueError):
        deserialize_artifact({"artifact_type": "unknown"})


def test_registry_slots_strictly_typed():
    registry = ArtifactRegistry()
    source = SourceArtifact(artifact_id="source-1", artifact_type="source")
    registry.set("source", source)
    assert registry.get("source") is source
    assert registry.artifact_ids() == {"source": "source-1"}
    with pytest.raises(KeyError):
        registry.set("not_a_slot", source)
    with pytest.raises(KeyError):
        registry.get("not_a_slot")
    assert set(ARTIFACT_SLOT_NAMES) == {
        "source",
        "media",
        "transcript",
        "evidence",
        "claims",
        "verification",
        "knowledge",
        "summary",
    }


def test_make_artifact_id_is_content_addressed():
    assert make_artifact_id("source", {"a": 1}) == make_artifact_id("source", {"a": 1})
    assert make_artifact_id("source", {"a": 1}) != make_artifact_id("source", {"a": 2})
    assert content_hash_of({"a": 1}) == content_hash_of({"a": 1})

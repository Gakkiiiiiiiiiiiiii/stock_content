"""Artifact 强类型契约测试（详细修改方案 §4 P0-1）。"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stock_content.domain.artifacts import (
    ARTIFACT_SLOT_NAMES,
    ArtifactRegistry,
    ClaimArtifact,
    ClaimOccurrenceArtifact,
    EvidenceArtifact,
    EvidenceItem,
    KnowledgeArtifact,
    LifecycleArtifact,
    MediaArtifact,
    SourceArtifact,
    SummaryArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    VerificationArtifact,
    artifact_id_of,
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


def test_final_membership_lists_are_order_and_duplicate_independent():
    claims_a = ClaimArtifact(
        artifact_id="claims-a",
        artifact_type="claims",
        claims=["claim-b", "claim-a", "claim-b"],
    )
    claims_b = ClaimArtifact(
        artifact_id="claims-b",
        artifact_type="claims",
        claims=["claim-a", "claim-b"],
    )
    occurrences_a = ClaimOccurrenceArtifact(
        artifact_id="occurrences-a",
        artifact_type="occurrences",
        occurrence_ids=["occ-b", "occ-a", "occ-b"],
    )
    occurrences_b = ClaimOccurrenceArtifact(
        artifact_id="occurrences-b",
        artifact_type="occurrences",
        occurrence_ids=["occ-a", "occ-b"],
    )
    lifecycle_a = LifecycleArtifact(
        artifact_id="lifecycle-a",
        artifact_type="lifecycle",
        claim_lifecycle_event_ids=["claim-event-b", "claim-event-a", "claim-event-b"],
        occurrence_lifecycle_event_ids=["occ-event-b", "occ-event-a", "occ-event-b"],
    )
    lifecycle_b = LifecycleArtifact(
        artifact_id="lifecycle-b",
        artifact_type="lifecycle",
        claim_lifecycle_event_ids=["claim-event-a", "claim-event-b"],
        occurrence_lifecycle_event_ids=["occ-event-a", "occ-event-b"],
    )

    assert artifact_id_of(claims_a) == artifact_id_of(claims_b)
    assert claims_a.claims == ["claim-a", "claim-b"]
    assert artifact_id_of(occurrences_a) == artifact_id_of(occurrences_b)
    assert occurrences_a.occurrence_ids == ["occ-a", "occ-b"]
    assert artifact_id_of(lifecycle_a) == artifact_id_of(lifecycle_b)
    assert lifecycle_a.claim_lifecycle_event_ids == ["claim-event-a", "claim-event-b"]
    assert lifecycle_a.occurrence_lifecycle_event_ids == ["occ-event-a", "occ-event-b"]

    knowledge_a = KnowledgeArtifact(
        artifact_id="knowledge-a",
        artifact_type="knowledge",
        knowledge_units=["unit-b", "unit-a", "unit-b"],
    )
    knowledge_b = KnowledgeArtifact(
        artifact_id="knowledge-b",
        artifact_type="knowledge",
        knowledge_units=["unit-a", "unit-b"],
    )
    assert artifact_id_of(knowledge_a) == artifact_id_of(knowledge_b)
    assert knowledge_a.knowledge_units == ["unit-a", "unit-b"]

    verification_a = VerificationArtifact(
        artifact_id="verification-a",
        artifact_type="verification",
        results=[
            {"claim_id": "claim-1", "status": "VERIFIED"},
            {"claim_id": "claim-1", "status": "CONTRADICTED"},
            {"claim_id": "claim-1", "status": "VERIFIED"},
        ],
    )
    verification_b = VerificationArtifact(
        artifact_id="verification-b",
        artifact_type="verification",
        results=[
            {"claim_id": "claim-1", "status": "CONTRADICTED"},
            {"claim_id": "claim-1", "status": "VERIFIED"},
        ],
    )
    assert artifact_id_of(verification_a) == artifact_id_of(verification_b)
    assert verification_a.results == verification_b.results
    assert len(verification_a.results) == 2


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


def test_retrieval_and_signal_contracts_declare_pit_and_v5_surfaces():
    root = Path(__file__).parents[1]
    content = yaml.safe_load((root / "contracts" / "content.v1.yaml").read_text(encoding="utf-8"))
    endpoints = content["endpoints"]
    search = [item for item in endpoints if item["path"] == "/api/v1/knowledge/search"]
    assert {(item["method"], item["name"]) for item in search} == {
        ("POST", "search_knowledge"), ("GET", "search_knowledge_get")
    }
    assert {
        "availability_as_of", "target_start", "target_end", "temporal_role",
        "semantic_segment_id", "business_as_of", "knowledge_as_of", "pit_mode",
    } <= set(search[0]["request"])
    v5 = yaml.safe_load((root / "contracts" / "content-factor-signal.v5.yaml").read_text(encoding="utf-8"))
    assert v5["version"] == "content-factor-signal.v5"
    assert {
        "claim_id", "occurrence_id", "semantic_segment_id", "asserted_at",
        "source_available_at", "source_availability_quality", "available_from",
        "temporal_bindings", "lifecycle", "content_snapshot_id",
    } <= set(v5["required"])

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.claim_occurrence_repository import ClaimOccurrenceRepository
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import AtomicClaimValidationStage
from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.knowledge_projection_builder import KnowledgeProjectionBuilder
from stock_content.domain.semantic_segment import materialize_semantic_segments
from stock_content.domain.temporal_semantics import OccurrenceTimes


def _transcript() -> TranscriptArtifact:
    return TranscriptArtifact(
        artifact_id="transcript-atomic", artifact_type="transcript", media_artifact_id="media",
        asr_model="fixture", asr_model_version="1",
        segments=[TranscriptSegmentItem(
            segment_index=0, start_seconds=0, end_seconds=1,
            text="600519营收增长10%。", raw_text="600519营收增长10%。",
            media_artifact_id="media", asr_model="fixture", asr_model_version="1",
        )],
    )


def _payload(segment_id: str, **changes) -> dict:
    value = {
        "semantic_segment_id": segment_id, "claim_type": "FINANCIAL_METRIC",
        "verbatim_quote": "600519营收增长10%", "normalized_statement": "600519营收增长10%",
        "subject": {"subject_type": "EQUITY", "subject_key": "600519"},
        "predicate": "营收增长", "object": {"text": "10%", "value": "10%"},
        "sentiment": "BULLISH", "evidence_segment_indices": [0], "extraction_confidence": 0.9,
    }
    value.update(changes)
    return value


def test_rejected_raw_atomic_payload_cannot_bypass_accepted_draft_boundary():
    transcript = _transcript()
    segment = materialize_semantic_segments(transcript, [])[0]
    context = PipelineContext(task_id="atomic", source={"type": "fixture", "ref": "one"}, options={
        "atomic_claim_payload": {"claims": [_payload(segment.semantic_segment_id, verbatim_quote="invented 11%")]},
    })
    context.artifacts.transcript = transcript
    context.state.semantic_segments = [segment]
    context.state.transcript_quality_report = SimpleNamespace(quality_status="PASS")
    AtomicClaimValidationStage().execute(context)
    assert context.state.claim_drafts == []
    assert len(context.state.atomic_claim_rejections) == 1


def test_grounded_projection_persists_exact_statement_quote_and_formal_eligibility(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'atomic.db'}")
    database.create_schema()
    claims = SqlClaimRepository(database.session_factory)
    occurrences = ClaimOccurrenceRepository(database.session_factory)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    claim = FinancialClaim(
        claim_type="FINANCIAL_METRIC", subject_type="EQUITY", subject_id="600519", ticker="600519",
        predicate="营收增长", value="10%", source_confidence=0.9, extractor_confidence=0.9,
        normalized_statement="600519营收增长10%", grounding_status="GROUNDED",
        condition_text="需求回暖", invalidation_text="需求下降",
        contradiction_group_id="contradiction_fixed", claim_schema_version="claim.atomic.v1",
        legacy_grounding_incomplete=False,
    )
    occurrence = ClaimOccurrence(
        claim_id=claim.claim_id, source_artifact_id="source", transcript_artifact_id="transcript",
        semantic_segment_id="segment", evidence_refs=["evidence-0"], primary_quote="600519营收增长10%",
        condition_evidence_refs=["evidence-condition"],
        invalidation_evidence_refs=["evidence-invalidation"],
        normalized_statement=claim.normalized_statement, grounding_status="GROUNDED",
        contradiction_group_id=claim.contradiction_group_id, claim_schema_version="claim.atomic.v1",
        legacy_grounding_incomplete=False,
        times=OccurrenceTimes(
            ingested_at=now,
            extraction_completed_at=now,
            snapshot_committed_at=now,
            available_from=now,
        ),
    )
    claims.save(claim)
    occurrences.save(occurrence)
    projection = KnowledgeProjectionBuilder().build(claim, occurrence)
    assert projection["statement"] == "600519营收增长10%"
    assert projection["attributes"]["verbatim_quote"] == "600519营收增长10%"
    assert projection["attributes"]["evidence_refs"] == ["evidence-0"]
    assert projection["attributes"]["condition"] == "需求回暖"
    assert projection["attributes"]["invalidation"] == "需求下降"
    assert projection["attributes"]["condition_evidence_refs"] == ["evidence-condition"]
    assert projection["attributes"]["invalidation_evidence_refs"] == ["evidence-invalidation"]
    assert projection["attributes"]["grounding_status"] == "GROUNDED"
    with pytest.raises(ValueError, match="without primary evidence"):
        KnowledgeProjectionBuilder().build(claim)
    assert claims.formal_bundle_eligible_claim_ids() == [claim.claim_id]

    legacy = FinancialClaim(
        claim_type="OPINION", subject_type="CONTENT", subject_id="legacy", predicate="legacy",
        source_confidence=0.1, extractor_confidence=0.1, evidence_refs=["legacy-evidence"],
    )
    claims.save(legacy)
    assert claims.formal_bundle_eligible_claim_ids() == [claim.claim_id]

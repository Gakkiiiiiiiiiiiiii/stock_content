from datetime import UTC, datetime

from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import LifecycleProjectionStage
from stock_content.domain.artifacts import ClaimOccurrenceArtifact, KnowledgeArtifact, VerificationArtifact
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.models import KnowledgeUnit
from stock_content.domain.temporal_semantics import OccurrenceTimes


def _claim(subject: str) -> FinancialClaim:
    return FinancialClaim(
        claim_type="OPINION",
        subject_type="THEME",
        subject_id=subject,
        predicate="allocation_rule",
        evidence_refs=[f"evidence-{subject}"],
        source_support_status="SUPPORTED",
        source_confidence=0.9,
        extractor_confidence=0.9,
        normalized_statement=f"{subject} 的规则",
        grounding_status="GROUNDED",
        legacy_grounding_incomplete=False,
        claim_schema_version="claim.atomic.v1",
    )


def _occurrence(claim: FinancialClaim, *, needs_review: bool) -> ClaimOccurrence:
    now = datetime(2026, 9, 9, tzinfo=UTC)
    return ClaimOccurrence(
        claim_id=claim.claim_id,
        source_artifact_id="source-fixture",
        transcript_artifact_id="transcript-fixture",
        semantic_segment_id=f"segment-{claim.subject_id}",
        evidence_refs=[f"evidence-{claim.subject_id}"],
        times=OccurrenceTimes(
            ingested_at=now,
            extraction_completed_at=now,
            snapshot_committed_at=now,
            available_from=now,
        ),
        primary_quote=claim.normalized_statement,
        normalized_statement=claim.normalized_statement,
        grounding_status="GROUNDED",
        legacy_grounding_incomplete=False,
        claim_schema_version="claim.atomic.v1",
        provenance={
            "bundle_v2": {
                "occurrence_review": {
                    "status": "HUMAN_REVIEW_REQUIRED" if needs_review else "NOT_REQUIRED",
                    "reason_codes": ["ASR_OCR_NUMERIC_CONFLICT"] if needs_review else [],
                }
            }
        },
    )


def test_human_review_required_occurrence_is_extracted_not_active_while_normal_item_remains_active():
    flagged_claim = _claim("flagged")
    ordinary_claim = _claim("ordinary")
    flagged = _occurrence(flagged_claim, needs_review=True)
    ordinary = _occurrence(ordinary_claim, needs_review=False)
    context = PipelineContext(
        task_id="human-review-lifecycle",
        source={"type": "fixture", "ref": "video"},
        options={"as_of": "2026-09-09T00:00:00Z"},
    )
    context.state.claims = [flagged_claim, ordinary_claim]
    context.state.occurrences = [flagged, ordinary]
    context.state.knowledge = [
        KnowledgeUnit(
            knowledge_uid=flagged.occurrence_id,
            video_id="video",
            chapter_id=None,
            statement=flagged_claim.normalized_statement,
            support_status="SOURCE_SUPPORTED",
            attributes={"occurrence_id": flagged.occurrence_id},
        ),
        KnowledgeUnit(
            knowledge_uid=ordinary.occurrence_id,
            video_id="video",
            chapter_id=None,
            statement=ordinary_claim.normalized_statement,
            support_status="SOURCE_SUPPORTED",
            attributes={"occurrence_id": ordinary.occurrence_id},
        ),
    ]
    context.artifacts.occurrences = ClaimOccurrenceArtifact(
        artifact_id="occurrences-fixture",
        artifact_type="occurrences",
        semantic_segment_artifact_id="semantic-fixture",
        evidence_artifact_id="evidence-fixture",
        occurrence_ids=[flagged.occurrence_id, ordinary.occurrence_id],
    )
    context.artifacts.verification = VerificationArtifact(
        artifact_id="verification-fixture",
        artifact_type="verification",
        claim_artifact_id="claims-fixture",
        results=[],
    )
    context.artifacts.knowledge = KnowledgeArtifact(
        artifact_id="knowledge-fixture",
        artifact_type="knowledge",
        verification_artifact_id="verification-fixture",
        knowledge_units=[flagged.occurrence_id, ordinary.occurrence_id],
    )

    LifecycleProjectionStage().execute(context)

    event_statuses = {
        (event.target_type.value, event.target_id): event.to_status
        for event in context.state.lifecycle_events
    }
    assert event_statuses[("CLAIM", flagged_claim.claim_id)] == "EXTRACTED"
    assert event_statuses[("OCCURRENCE", flagged.occurrence_id)] == "EXTRACTED"
    assert event_statuses[("CLAIM", ordinary_claim.claim_id)] == "ACTIVE"
    assert event_statuses[("OCCURRENCE", ordinary.occurrence_id)] == "ACTIVE"

    units = {unit.knowledge_uid: unit for unit in context.state.knowledge}
    flagged_unit = units[flagged.occurrence_id]
    assert flagged_unit.lifecycle_status == "EXTRACTED"
    assert flagged_unit.support_status == "SOURCE_LOCATED"
    assert flagged_unit.review_status == "UNREVIEWED"
    assert flagged_unit.attributes["occurrence_review"] == {
        "status": "HUMAN_REVIEW_REQUIRED",
        "reason_codes": ["ASR_OCR_NUMERIC_CONFLICT"],
    }
    assert units[ordinary.occurrence_id].lifecycle_status == "ACTIVE"
    assert units[ordinary.occurrence_id].support_status == "SOURCE_SUPPORTED"

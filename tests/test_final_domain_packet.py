from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.video_repository import PostgresVideoRepository
from stock_content.domain.artifacts import (
    ArtifactRegistry,
    LifecycleArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    deserialize_artifact,
    serialize_artifact,
    transcript_segment_id,
)
from stock_content.domain.claim_canonicalizer import ClaimCanonicalizer
from stock_content.domain.claim_draft import ClaimOccurrenceDraft, TemporalExpressionDraft
from stock_content.domain.claim_draft_grounder import ClaimDraftGrounder
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.lifecycle_event import KnowledgeLifecycleEvent, select_lifecycle_event
from stock_content.domain.models import TranscriptSegment, VideoAsset
from stock_content.domain.semantic_segment import SemanticBoundary, materialize_semantic_segments
from stock_content.domain.temporal_normalizer import TemporalNormalizer
from stock_content.domain.temporal_semantics import (
    ClaimTemporalBinding,
    OccurrenceTimes,
    TemporalRole,
    TemporalScope,
    TemporalValueType,
    temporal_binding_id_of,
)


def _transcript() -> TranscriptArtifact:
    return TranscriptArtifact(
        artifact_id="tr-artifact",
        artifact_type="transcript",
        media_artifact_id="media-1",
        asr_model="whisper",
        asr_model_version="v1",
        segments=[
            TranscriptSegmentItem(
                segment_index=i,
                start_seconds=float(i),
                end_seconds=float(i + 1),
                text=text,
                raw_text=text,
                media_artifact_id="media-1",
                asr_model="whisper",
                asr_model_version="v1",
            )
            for i, text in enumerate(("营收增长 10%", "条件是需求回暖", "以后风险上升"))
        ],
    )


def test_segment_id_uses_full_asr_identity_and_materialization_is_gap_free():
    transcript = _transcript()
    assert transcript.segments[0].segment_id == transcript_segment_id(
        media_artifact_id="media-1",
        asr_model="whisper",
        asr_model_version="v1",
        segment_index=0,
        start_ms=0,
        end_ms=1000,
        raw_text="营收增长 10%",
    )
    segments = materialize_semantic_segments(
        transcript,
        [SemanticBoundary(1, next_topic="next topic", next_subject="next subject")],
    )
    assert [(x.start_segment_index, x.end_segment_index) for x in segments] == [(0, 1), (2, 2)]
    assert segments[0].topic is None and segments[1].topic == "next topic"
    mismatched = TranscriptArtifact(
        artifact_id="tr-mismatch",
        artifact_type="transcript",
        media_artifact_id="media-new",
        asr_model="new-model",
        asr_model_version="v2",
        segments=[
            TranscriptSegmentItem(
                segment_id="trseg_old",
                segment_index=0,
                start_seconds=0,
                end_seconds=1,
                text="same",
                raw_text="same",
                media_artifact_id="media-old",
                asr_model="old-model",
                asr_model_version="v1",
            )
        ],
    )
    assert mismatched.segments[0].segment_id != "trseg_old"
    with pytest.raises(ValueError):
        materialize_semantic_segments(transcript, [{"after_segment_index": 99}])


def test_temporal_date_timestamp_invariant_and_normalized_unresolved_identity():
    normalized = TemporalNormalizer().normalize("2026Q2")
    assert normalized.value_type is TemporalValueType.DATE
    assert normalized.start_date.isoformat() == "2026-04-01"
    assert normalized.end_date.isoformat() == "2026-06-30"
    with pytest.raises(ValueError):
        ClaimTemporalBinding(role="VALID_AT", scope="POINT", value_type="DATE", start_time=datetime.now(timezone.utc))
    assert (
        TemporalNormalizer().normalize("以后").temporal_binding_id
        != TemporalNormalizer().normalize("很久以后").temporal_binding_id
    )
    assert TemporalNormalizer().normalize("今天").normalization_status == "UNRESOLVED"
    forecast = TemporalNormalizer().normalize("2026Q2", role=TemporalRole.FORECAST_TARGET)
    assert forecast.scope is TemporalScope.FORECAST
    assert forecast.assertion_status.value == "EXPECTED"
    for expression in ("FY2027", "FY2027Q2"):
        unresolved = TemporalNormalizer().normalize(expression)
        assert unresolved.scope in {TemporalScope.UNKNOWN, TemporalScope.INTERVAL}
        assert unresolved.value_type is TemporalValueType.NONE
        assert unresolved.period_label == expression
        assert unresolved.normalization_status == "PARTIAL"
        assert unresolved.expression_key


def test_temporal_binding_identity_excludes_parser_metadata_but_keeps_partial_expression_key():
    first = ClaimTemporalBinding(
        role="VALID_AT",
        scope="POINT",
        value_type="DATE",
        start_date="2026-01-02",
        end_date="2026-01-02",
        normalization_status="NORMALIZED",
        normalization_version="normalization.v1",
        raw_expression="first spelling",
        source_evidence_refs=["ev-a"],
        reference_snapshot_id="snap-a",
        reference_data_version="a",
    )
    second = first.model_copy(
        update={
            "normalization_version": "normalization.v9",
            "source_evidence_refs": ["ev-b"],
            "reference_snapshot_id": "snap-b",
            "reference_data_version": "b",
            "normalization_reason": "different parser",
        }
    )
    assert temporal_binding_id_of(first) == temporal_binding_id_of(second)


def test_grounder_is_fail_closed_and_occurrence_is_model_independent():
    transcript = _transcript()
    sem = materialize_semantic_segments(transcript, [SemanticBoundary(1)])[0]
    draft = ClaimOccurrenceDraft(
        semantic_segment_id=sem.semantic_segment_id,
        knowledge_kind="FACT",
        claim_type="FINANCIAL_METRIC",
        subject_key="600519.SH",
        predicate_key="revenue_growth",
        conclusion="营收增长 10%",
        evidence_segment_indices=[0],
        temporal_expressions=[
            TemporalExpressionDraft(
                role="VALID_AT", raw_expression="营收", evidence_segment_indices=[0], confidence=0.8
            )
        ],
        extraction_confidence=0.8,
    )
    grounded = ClaimDraftGrounder().ground(draft, transcript, sem)
    assert grounded.evidences[0].source_artifact_id == transcript.artifact_id
    temporal = [item for item in grounded.evidences if item.evidence_id in grounded.temporal_evidence_refs]
    assert temporal and temporal[0].locator["segment_index"] == 0
    assert temporal[0].locator["semantic_segment_id"] == sem.semantic_segment_id
    multi_role = draft.model_copy(update={"condition_evidence_segment_indices": [0]})
    grounded_multi = ClaimDraftGrounder().ground(multi_role, transcript, sem)
    assert grounded_multi.primary_evidence_refs == grounded_multi.condition_evidence_refs
    assert grounded_multi.evidences[0].evidence_id == grounded.evidences[0].evidence_id
    times = {
        "ingested_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "extraction_completed_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "snapshot_committed_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
        "available_from": datetime(2026, 1, 3, tzinfo=timezone.utc),
    }
    first = ClaimOccurrence(
        claim_id="claim-1",
        source_artifact_id="source",
        transcript_artifact_id="tr",
        semantic_segment_id="seg",
        evidence_refs=["ev"],
        times=times,
        provenance={"model_id": "a"},
    )
    second = ClaimOccurrence(
        claim_id="claim-1",
        source_artifact_id="source",
        transcript_artifact_id="tr",
        semantic_segment_id="seg",
        temporal_evidence_refs=["ev"],
        times=times,
        provenance={"model_id": "b"},
    )
    assert first.occurrence_id == second.occurrence_id


def test_canonicalizer_ignores_untrusted_category_and_uses_binding_version():
    draft = ClaimOccurrenceDraft(
        semantic_segment_id="seg",
        knowledge_kind="OPINION",
        claim_type="FINANCIAL_METRIC",
        subject_key="600519.SH",
        predicate_key="revenue",
        value=10,
        extraction_confidence=0.8,
    )
    binding = ClaimTemporalBinding(
        role="REPORTING_PERIOD",
        scope="INTERVAL",
        value_type="DATE",
        start_date="2026-04-01",
        end_date="2026-06-30",
        normalization_version="temporal-normalization.final.7",
    )
    claim = ClaimCanonicalizer().canonicalize(draft, temporal_bindings=[binding])
    assert claim.fact_category == "FACT"
    assert claim.normalization_version == "temporal-normalization.final.7"
    with pytest.raises(ValueError, match="normalization version"):
        ClaimCanonicalizer(normalization_version="temporal-normalization.final.8").canonicalize(
            draft, temporal_bindings=[binding]
        )


def test_occurrence_times_require_snapshot_availability_equality():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="available_from must equal"):
        OccurrenceTimes(
            ingested_at=now,
            extraction_completed_at=now,
            snapshot_committed_at=now,
            available_from=now.replace(day=2),
        )


def test_artifact_roundtrip_lifecycle_and_migration_guardrails():
    artifact = LifecycleArtifact(artifact_id="life", artifact_type="lifecycle", policy_version="v1")
    assert deserialize_artifact(serialize_artifact(artifact)).content_hash == artifact.content_hash
    registry = ArtifactRegistry()
    registry.set("lifecycle", artifact)
    assert registry.get("lifecycle") is artifact
    sql = "\n".join(
        Path("migrations", f"{number:03d}_{suffix}.sql").read_text(encoding="utf-8")
        for number, suffix in (
            (17, "semantic_segments"),
            (18, "temporal_semantics_final"),
            (19, "claim_occurrence_final"),
            (20, "lifecycle_event_ledger"),
            (21, "lifecycle_artifact_support"),
            (22, "temporal_backfill"),
        )
    )
    assert "CHECK (target_type IN ('CLAIM', 'OCCURRENCE'))" in sql
    assert "value_type <> 'DATE'" in sql and "value_type <> 'TIMESTAMP'" in sql
    assert "Do not map legacy valid_to" in sql
    assert "ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS lifecycle_artifact_id" in sql
    assert "INSERT INTO knowledge_lifecycle_event" in sql
    assert "LEGACY_IMPORTED" in sql
    assert "ALTER TABLE video_segment ALTER COLUMN segment_id SET NOT NULL" in sql
    event = KnowledgeLifecycleEvent(
        target_type="CLAIM",
        target_id="c",
        to_status="ACTIVE",
        effective_at=datetime(2026, 1, 1),
        recorded_at=datetime(2026, 1, 2),
        reason_code="EXTRACTED",
        policy_version="v1",
    )
    replay = event.model_copy(update={"recorded_at": datetime(2026, 2, 1)})
    assert event.lifecycle_event_id == replay.lifecycle_event_id


def test_lifecycle_selector_is_bitemporal_and_target_strict():
    def event(target_id, effective, recorded, suffix):
        return KnowledgeLifecycleEvent(
            target_type="CLAIM",
            target_id=target_id,
            to_status=suffix,
            effective_at=datetime.fromisoformat(effective),
            recorded_at=datetime.fromisoformat(recorded),
            reason_code="TEST",
            policy_version="v1",
        )

    events = [
        event("c", "2026-01-01T00:00:00", "2026-01-03T00:00:00", "A"),
        event("c", "2026-01-01T00:00:00", "2026-01-04T00:00:00", "B"),
        event("c", "2026-02-01T00:00:00", "2026-02-02T00:00:00", "C"),
        event("other", "2026-12-01T00:00:00", "2026-12-02T00:00:00", "OTHER"),
    ]
    selected = select_lifecycle_event(
        events,
        target_type="CLAIM",
        target_id="c",
        business_as_of=datetime.fromisoformat("2026-02-15T00:00:00"),
        knowledge_as_of=datetime.fromisoformat("2026-01-03T12:00:00"),
    )
    assert selected is not None and selected.to_status == "A"
    assert select_lifecycle_event(
        events,
        target_type="OCCURRENCE",
        target_id="c",
        business_as_of=datetime.fromisoformat("2027-01-01T00:00:00"),
        knowledge_as_of=datetime.fromisoformat("2027-01-01T00:00:00"),
    ) is None


def test_video_segment_legacy_schema_upgrade_and_persistence(tmp_path):
    url = f"sqlite:///{tmp_path / 'legacy-segments.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text("""
            CREATE TABLE video_asset (
                video_id VARCHAR(64) PRIMARY KEY, source_type VARCHAR(32), source_ref TEXT,
                title TEXT, author VARCHAR(255), duration_seconds FLOAT, transcript_text TEXT,
                source_hash VARCHAR(64), created_at TIMESTAMP, updated_at TIMESTAMP
            )
        """)
        )
        connection.execute(
            text("""
            CREATE TABLE video_segment (
                id INTEGER PRIMARY KEY, video_id VARCHAR(64), segment_index INTEGER,
                start_seconds FLOAT, end_seconds FLOAT, text TEXT, confidence FLOAT,
                UNIQUE(video_id, segment_index)
            )
        """)
        )
        connection.execute(
            text(
                "INSERT INTO video_asset (video_id, source_type, source_ref, title, transcript_text) "
                "VALUES ('v1', 'fixture', 'r1', 'title', '')"
            )
        )
        connection.execute(text("INSERT INTO video_segment VALUES (1, 'v1', 0, 0, 1, 'legacy text', 0.8)"))
    database = Database(url)
    database.create_schema()
    with database.session_factory() as session:
        row = session.execute(text("SELECT segment_id FROM video_segment WHERE id = 1")).one()
        assert row.segment_id.startswith("legacy_trseg_")
    repo = PostgresVideoRepository(database.session_factory)
    repo.upsert(
        VideoAsset(video_id="v1", source_type="fixture", source_ref="r1", title="title"),
        [TranscriptSegment(segment_index=0, start_seconds=0, end_seconds=1, text="new text")],
    )
    stored = repo.get("v1")
    assert stored["segments"][0]["segment_id"].startswith("legacy_trseg_")


def test_canonical_claim_is_evidence_free_and_binding_order_independent():
    draft = ClaimOccurrenceDraft(
        semantic_segment_id="seg",
        knowledge_kind="FACT",
        claim_type="FINANCIAL_METRIC",
        subject_key="600519.SH",
        predicate_key="revenue",
        conclusion="revenue",
        extraction_confidence=0.8,
    )
    first_binding = TemporalNormalizer().normalize("2026Q2")
    second_binding = TemporalNormalizer().normalize("2026Q3")
    first = ClaimCanonicalizer().canonicalize(
        draft,
        temporal_bindings=[first_binding, second_binding],
        evidence_refs=["source-a"],
    )
    second = ClaimCanonicalizer().canonicalize(
        draft,
        temporal_bindings=[second_binding, first_binding],
        evidence_refs=["source-b"],
    )
    assert first.evidence_refs == []
    assert first.claim_schema_version == "claim.final.v1"
    assert first.model_dump() == type(first).model_validate(first.model_dump()).model_dump()
    assert first.claim_id == second.claim_id
    assert first.condition_key is None

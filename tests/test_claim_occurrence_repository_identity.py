from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.claim_occurrence_repository import ClaimOccurrenceRepository
from stock_content.domain.claim_occurrence import ClaimOccurrence
from stock_content.domain.temporal_semantics import OccurrenceTimes


def _occurrence(*, semantic_segment_id: str = "segment-1", now: datetime):
    return ClaimOccurrence(
        claim_id="claim-1",
        source_artifact_id="source-1",
        transcript_artifact_id="transcript-1",
        semantic_segment_id=semantic_segment_id,
        evidence_refs=["evidence-1"],
        source_support_status="SOURCE_LOCATED",
        source_confidence=0.4,
        extractor_confidence=0.5,
        raw_temporal_expressions=[{"raw": "2026Q1", "model": "model-a"}],
        provenance={"model_id": "model-a", "prompt_version": "prompt-a"},
        times=OccurrenceTimes(
            source_published_at=now - timedelta(days=3),
            source_available_at=now - timedelta(days=2),
            ingested_at=now - timedelta(days=2),
            extraction_completed_at=now - timedelta(days=1),
            snapshot_committed_at=now,
            available_from=now,
        ),
    )


def test_same_occurrence_id_from_different_model_prompt_and_times_keeps_first_row(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'occurrence-idempotency.db'}")
    database.create_schema()
    repository = ClaimOccurrenceRepository(database.session_factory)
    first = _occurrence(now=datetime(2026, 1, 3, tzinfo=UTC))
    second = first.model_copy(
        update={
            "source_support_status": "SOURCE_SUPPORTED",
            "source_confidence": 0.99,
            "extractor_confidence": 0.98,
            "raw_temporal_expressions": [{"raw": "2026Q1", "model": "model-b"}],
            "provenance": {"model_id": "model-b", "prompt_version": "prompt-b"},
            "times": first.times.model_copy(
                update={
                    "source_available_at": datetime(2026, 1, 4, tzinfo=UTC),
                    "ingested_at": datetime(2026, 1, 4, tzinfo=UTC),
                    "extraction_completed_at": datetime(2026, 1, 5, tzinfo=UTC),
                    "snapshot_committed_at": datetime(2026, 1, 6, tzinfo=UTC),
                    "available_from": datetime(2026, 1, 6, tzinfo=UTC),
                }
            ),
        }
    )
    assert second.occurrence_id == first.occurrence_id

    assert repository.save(first).times.source_available_at == first.times.source_available_at
    returned = repository.save(second)
    stored = repository.get(first.occurrence_id)
    assert returned == stored
    assert stored is not None
    assert stored.times == first.times
    assert stored.provenance == first.provenance
    assert stored.source_support_status == first.source_support_status
    assert stored.source_confidence == first.source_confidence


def test_identity_coordinate_mismatch_is_rejected(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'occurrence-identity.db'}")
    database.create_schema()
    repository = ClaimOccurrenceRepository(database.session_factory)
    first = _occurrence(now=datetime(2026, 1, 3, tzinfo=UTC))
    repository.save(first)

    # Keep the supplied ID while changing a coordinate: an invalid supplied
    # identity must fail closed rather than overwrite the existing row.
    mismatch = first.model_copy(update={"semantic_segment_id": "segment-2"})
    with pytest.raises(ValueError, match="immutable identity|evidence coordinates"):
        repository.save(mismatch)

    # A normal coordinate change derives a distinct locator and occurrence ID.
    changed_coordinate = _occurrence(semantic_segment_id="segment-2", now=datetime(2026, 1, 3, tzinfo=UTC))
    assert changed_coordinate.assertion_locator_hash != first.assertion_locator_hash
    assert changed_coordinate.occurrence_id != first.occurrence_id
    repository.save(changed_coordinate)
    assert repository.get(changed_coordinate.occurrence_id) is not None


def test_evidence_role_reassignment_with_same_union_keeps_first_write_roles(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'occurrence-role-identity.db'}")
    database.create_schema()
    repository = ClaimOccurrenceRepository(database.session_factory)
    first = _occurrence(now=datetime(2026, 1, 3, tzinfo=UTC))
    repository.save(first)
    reassigned = first.model_copy(
        update={"evidence_refs": [], "condition_evidence_refs": ["evidence-1"]}
    )
    assert reassigned.occurrence_id == first.occurrence_id
    stored = repository.save(reassigned)
    assert stored.evidence_refs == first.evidence_refs
    assert stored.condition_evidence_refs == first.condition_evidence_refs


def test_concurrent_same_occurrence_has_one_winner_row_and_one_role_projection(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'occurrence-concurrent.db'}")
    database.create_schema()
    repository = ClaimOccurrenceRepository(database.session_factory)
    first = _occurrence(now=datetime(2026, 1, 3, tzinfo=UTC))
    second = first.model_copy(
        update={
            "evidence_refs": [],
            "condition_evidence_refs": ["evidence-1"],
            "provenance": {"model_id": "model-b", "prompt_version": "prompt-b"},
            "times": first.times.model_copy(
                update={
                    "source_available_at": datetime(2026, 1, 4, tzinfo=UTC),
                    "ingested_at": datetime(2026, 1, 4, tzinfo=UTC),
                    "extraction_completed_at": datetime(2026, 1, 5, tzinfo=UTC),
                    "snapshot_committed_at": datetime(2026, 1, 6, tzinfo=UTC),
                    "available_from": datetime(2026, 1, 6, tzinfo=UTC),
                }
            ),
        }
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        returned = list(pool.map(repository.save, (first, second)))

    stored = repository.get(first.occurrence_id)
    assert stored is not None
    assert len(repository.list_for_claim(first.claim_id)) == 1
    assert (tuple(stored.evidence_refs), tuple(stored.condition_evidence_refs)) in {
        (tuple(first.evidence_refs), tuple(first.condition_evidence_refs)),
        (tuple(second.evidence_refs), tuple(second.condition_evidence_refs)),
    }
    assert len({item.occurrence_id for item in returned}) == 1
    assert returned[0].provenance == stored.provenance
    assert returned[1].provenance == stored.provenance

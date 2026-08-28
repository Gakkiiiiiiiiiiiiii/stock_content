from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    TemporalBindingRow,
    TemporalRelationRow,
)
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.temporal_semantics import ClaimTemporalBinding, ClaimTemporalRelation


def _claim(subject_id: str, binding: ClaimTemporalBinding, relation: ClaimTemporalRelation) -> FinancialClaim:
    return FinancialClaim(
        claim_type="FINANCIAL_METRIC",
        subject_type="EQUITY",
        subject_id=subject_id,
        predicate="revenue",
        value=100,
        evidence_refs=["claim-evidence"],
        source_confidence=0.8,
        extractor_confidence=0.9,
        temporal_bindings=[binding],
        temporal_relations=[relation],
    )


def test_temporal_identity_can_be_shared_while_provenance_stays_claim_local(tmp_path):
    binding_one = ClaimTemporalBinding(
        role="REPORTING_PERIOD",
        scope="INTERVAL",
        value_type="DATE",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        raw_expression="first quarter",
        source_evidence_refs=["evidence-one"],
        confidence=0.7,
    )
    binding_two = binding_one.model_copy(
        update={
            "raw_expression": "Q1 2026",
            "source_evidence_refs": ["evidence-two"],
            "confidence": 0.95,
        }
    )
    relation_one = ClaimTemporalRelation(
        relation_type="CAUSAL_LAG",
        from_binding_id=binding_one.temporal_binding_id,
        to_binding_id=binding_one.temporal_binding_id,
        lag_value=1,
        lag_unit="QUARTER",
        confidence=0.6,
    )
    relation_two = relation_one.model_copy(update={"confidence": 0.9})
    assert binding_one.temporal_binding_id == binding_two.temporal_binding_id
    assert relation_one.temporal_relation_id == relation_two.temporal_relation_id

    database = Database(f"sqlite:///{tmp_path / 'temporal-owner-keys.db'}")
    database.create_schema()
    repository = SqlClaimRepository(database.session_factory)
    claim_one = _claim("equity-one", binding_one, relation_one)
    claim_two = _claim("equity-two", binding_two, relation_two)
    repository.save(claim_one)
    repository.save(claim_two)

    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(TemporalBindingRow)) == 2
        assert session.scalar(select(func.count()).select_from(TemporalRelationRow)) == 2
        rows = session.scalars(
            select(TemporalBindingRow).order_by(TemporalBindingRow.claim_id)
        ).all()
        assert {row.claim_id for row in rows} == {claim_one.claim_id, claim_two.claim_id}

    stored_one = repository.temporal_bindings(claim_one.claim_id)[0]
    stored_two = repository.temporal_bindings(claim_two.claim_id)[0]
    assert stored_one["temporal_binding_id"] == stored_two["temporal_binding_id"]
    assert stored_one["raw_expression"] == "first quarter"
    assert stored_two["raw_expression"] == "Q1 2026"
    assert stored_one["source_evidence_refs"] == ["evidence-one"]
    assert stored_two["source_evidence_refs"] == ["evidence-two"]
    relation_rows_one = repository.temporal_relations(claim_one.claim_id)
    relation_rows_two = repository.temporal_relations(claim_two.claim_id)
    assert relation_rows_one[0]["temporal_relation_id"] == relation_rows_two[0]["temporal_relation_id"]


def test_temporal_orm_uses_claim_owner_composite_keys():
    assert [column.name for column in TemporalBindingRow.__table__.primary_key.columns] == [
        "claim_id",
        "temporal_binding_id",
    ]
    assert [column.name for column in TemporalRelationRow.__table__.primary_key.columns] == [
        "claim_id",
        "temporal_relation_id",
    ]
    migration = Path(__file__).parents[1] / "migrations" / "023_temporal_owner_keys.sql"
    sql = migration.read_text(encoding="utf-8")
    assert "PRIMARY KEY (claim_id, temporal_binding_id)" in sql
    assert "PRIMARY KEY (claim_id, temporal_relation_id)" in sql


def test_temporal_identity_mismatch_for_same_claim_fails_closed(tmp_path):
    binding = ClaimTemporalBinding(
        role="REPORTING_PERIOD",
        scope="INTERVAL",
        value_type="DATE",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        raw_expression="Q1",
    )
    relation = ClaimTemporalRelation(
        relation_type="CAUSAL_LAG",
        from_binding_id=binding.temporal_binding_id,
        to_binding_id=binding.temporal_binding_id,
        lag_value=1,
        lag_unit="QUARTER",
    )
    database = Database(f"sqlite:///{tmp_path / 'temporal-owner-mismatch.db'}")
    database.create_schema()
    repository = SqlClaimRepository(database.session_factory)
    claim = _claim("equity-one", binding, relation)
    repository.save(claim)

    changed_binding = binding.model_copy(update={"start_date": date(2026, 2, 1)})
    with pytest.raises(ValueError, match="temporal binding id"):
        repository.save(claim.model_copy(update={"temporal_bindings": [changed_binding]}))

    changed_relation = relation.model_copy(update={"lag_value": 2})
    with pytest.raises(ValueError, match="temporal relation id"):
        repository.save(claim.model_copy(update={"temporal_relations": [changed_relation]}))

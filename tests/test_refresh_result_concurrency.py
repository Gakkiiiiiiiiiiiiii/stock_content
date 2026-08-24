from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ClaimVerificationResultRow,
    ContentSourceHeadRow,
)
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    VerificationJobIntegrityError,
    persist_verification_result,
)
from stock_content.application.verification_refresh import _upsert_source_head
from stock_content.domain.claims import FinancialClaim, VerificationResult


def _claim() -> FinancialClaim:
    return FinancialClaim(
        claim_type="PRICE",
        subject_type="EQUITY",
        subject_id="600000.SH",
        predicate="price",
        value=10,
        evidence_refs=["evidence-1"],
        source_confidence=0.9,
        extractor_confidence=0.9,
    )


def _result_values(claim: FinancialClaim, result: VerificationResult, now: datetime) -> dict:
    return {
        "verification_id": "verification-concurrency",
        "claim_id": claim.claim_id,
        "provider": "quant",
        "status": result.status,
        "market_snapshot_id": result.market_snapshot_id,
        "market_data_version": result.market_data_version,
        "result_payload": result.model_dump(mode="json"),
        "trace_id": "trace-1",
        "fact_date": result.fact_date,
        "adjustment": result.adjustment,
        "verification_timestamp": result.verification_timestamp,
        "verification_rule_version": result.verification_rule_version,
        "verified_at": result.verification_timestamp,
        "created_at": now,
    }


def test_result_conflict_does_not_poison_outer_transaction(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'result-conflict.db'}")
    database.create_schema()
    claim = _claim()
    SqlClaimRepository(database.session_factory).save(claim)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    result = VerificationResult(
        claim_id=claim.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-1",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    values = _result_values(claim, result, now)

    with database.session_factory.begin() as session:
        first = persist_verification_result(session, values)
        second = persist_verification_result(session, values)
        assert first.verification_id == second.verification_id
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == 1

    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == 1


def test_result_conflict_rejects_immutable_payload(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'result-payload.db'}")
    database.create_schema()
    claim = _claim()
    SqlClaimRepository(database.session_factory).save(claim)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    result = VerificationResult(
        claim_id=claim.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-1",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    values = _result_values(claim, result, now)
    with database.session_factory.begin() as session:
        persist_verification_result(session, values)
        conflicting = {**values, "result_payload": {**values["result_payload"], "reason": "different"}}
        with pytest.raises(VerificationJobIntegrityError):
            persist_verification_result(session, conflicting)
        # The savepoint/native upsert conflict leaves the outer UoW usable.
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == 1


def test_first_source_head_refresh_upsert_and_verified_pointer_are_monotonic(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'head-upsert.db'}")
    database.create_schema()
    source_hash = "source-hash"
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 1, 2, tzinfo=UTC)
    with database.session_factory.begin() as session:
        _upsert_source_head(
            session,
            source_identity_hash=source_hash,
            snapshot_id="snapshot-old",
            verified_snapshot_id="snapshot-old",
            updated_at=older,
        )
        _upsert_source_head(
            session,
            source_identity_hash=source_hash,
            snapshot_id="snapshot-new",
            verified_snapshot_id=None,
            updated_at=newer,
        )
        _upsert_source_head(
            session,
            source_identity_hash=source_hash,
            snapshot_id="snapshot-older-retry",
            verified_snapshot_id="snapshot-older-retry",
            updated_at=older,
        )

    with database.session_factory() as session:
        head = session.get(ContentSourceHeadRow, source_hash)
        assert head is not None
        assert head.latest_snapshot_id == "snapshot-new"
        assert head.latest_verified_snapshot_id == "snapshot-old"

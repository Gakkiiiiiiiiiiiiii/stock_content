from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ClaimVerificationResultRow,
    ContentArtifactRow,
    ContentSnapshotRow,
    SignalOutboxRow,
)
from stock_content.adapters.postgres.repositories import (
    PostgresVerificationJobRepository,
    SignalOutboxRepository,
    SqlClaimRepository,
)
from stock_content.api.dependencies import build_application
from stock_content.application.signal_publisher import SignalPublisherApplication
from stock_content.application.verification_refresh import VerificationRefreshService
from stock_content.domain.claims import FinancialClaim, VerificationResult
from stock_content.domain.signal_contract import signal_id_v4, validate_signal_v4
from stock_content.domain.signal_policy import SignalPolicy


def _claim(kind: str, support: str = "SUPPORTED", confidence: float = 0.9) -> FinancialClaim:
    return FinancialClaim(
        claim_type=kind,
        subject_type="EQUITY",
        subject_id="600000.SH",
        predicate="statement",
        value=1,
        fact_time=datetime(2026, 1, 1, tzinfo=UTC),
        evidence_refs=["ev-1"],
        source_support_status=support,
        source_confidence=confidence,
        extractor_confidence=confidence,
    )


def test_signal_policy_table_and_exact_id():
    policy = SignalPolicy()
    snapshot = {"content_snapshot_id": "s1", "pipeline_version": "pipeline.v3", "created_at": "now"}
    verified = VerificationResult(
        claim_id="unused",
        status="VERIFIED",
        market_snapshot_id="m1",
        market_data_version="v1",
        fact_date=datetime(2026, 1, 1, tzinfo=UTC).date(),
        adjustment="NONE",
        verification_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    fact = _claim("PRICE")
    assert policy.evaluate(fact, verified).status == "NORMAL"
    assert policy.evaluate(fact, {"status": "PARTIALLY_VERIFIED"}).status == "DEGRADED"
    assert policy.evaluate(fact, {"status": "CONTRADICTED"}).status == "CONTRADICTION"
    assert policy.evaluate(fact, {"status": "MANUAL_REVIEW"}).status == "SUPPRESSED"
    forecast = _claim("FORECAST")
    opinion = _claim("OPINION")
    inference = _claim("INFERENCE")
    assert policy.evaluate(forecast).truth_scope == "AUTHOR_FORECAST"
    assert policy.evaluate(opinion).truth_scope == "AUTHOR_OPINION"
    assert policy.evaluate(inference).truth_scope == "SYSTEM_INFERENCE"
    signal = policy.build_signal(snapshot, fact, verified, verification_artifact_id="va1")
    expected = signal_id_v4(
        content_snapshot_id="s1",
        claim_id=fact.claim_id,
        policy_version=policy.version,
        signal_type="FACT",
    )
    assert signal["signal_id"] == expected
    assert validate_signal_v4(signal)["signal_id"] == expected
    with pytest.raises(ValueError, match="published_at"):
        validate_signal_v4({key: value for key, value in signal.items() if key != "published_at"})
    with pytest.raises(ValueError, match="source"):
        validate_signal_v4(signal | {"source": {}})
    with pytest.raises(ValueError, match="producer"):
        validate_signal_v4(signal | {"producer": {}})
    with pytest.raises(ValueError, match="unsupported"):
        validate_signal_v4(signal | {"signal_schema_version": "content-factor-signal.v5"})
    with pytest.raises(ValueError):
        validate_signal_v4(signal | {"order_qty": 1})


def test_v4_event_time_fallback_and_forecast_opinion_contract():
    policy = SignalPolicy()
    snapshot = {
        "content_snapshot_id": "s-fallback", "pipeline_version": "pipeline.v3",
        "created_at": "2026-02-03T04:05:06+00:00", "source_type": "fixture", "source_ref": "fallback",
    }
    verified = {"artifact_id": "verification-fallback", "status": "VERIFIED"}
    for kind in ("FORECAST", "OPINION"):
        claim = _claim(kind).model_copy(update={"fact_time": None, "published_at": None})
        signal = policy.build_signal(snapshot, claim, verified)
        assert signal["event_time"] == snapshot["created_at"]
        assert signal["published_at"] == snapshot["created_at"]
        assert signal["decision_id"] == signal["producer"]["decision_id"]
        validate_signal_v4(signal)


def test_outbox_idempotency_publisher_and_retry(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'outbox.db'}")
    database.create_schema()
    outbox = SignalOutboxRepository(database.session_factory)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    payload = {
        "signal_id": "signal-1",
        "signal_schema_version": "content-factor-signal.v4",
        "content_snapshot_id": "s1",
        "claim_id": "c1",
    }
    outbox.enqueue(payload, now)
    outbox.enqueue(payload, now)
    assert len(outbox.claim_due("publisher-a", now=now)) == 1
    assert outbox.claim_due("publisher-b", now=now) == []

    class Publisher:
        def __init__(self):
            self.calls = 0

        def publish(self, payload, idempotency_key):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transport down")

    publisher = Publisher()
    app = SignalPublisherApplication(outbox, publisher)
    # The row is currently leased; recover it after expiry.
    first = app.run_once("publisher-a", now=now + timedelta(seconds=61))
    assert first["retried"] == 1
    second = app.run_once("publisher-a", now=now + timedelta(seconds=121))
    assert second["published"] == 1


def _run_fixture(tmp_path: Path):
    app = build_application(f"sqlite:///{tmp_path / 'refresh.db'}", enable_qdrant=False)
    app.enqueue(
        "bilibili",
        "BV1refresh",
        {"metadata": {"title": "refresh"}, "transcript": "股票600000基本面良好。", "offline_fixture": True},
    )
    result = app.process_next("ingest")
    assert result["status"] == "SUCCEEDED", result
    session_factory = app._pipeline._stages[0]._artifact_repository._sessions  # noqa: SLF001
    return app, result, session_factory


def test_verification_refresh_atomic_rollback_and_append(tmp_path):
    app, initial, session_factory = _run_fixture(tmp_path)
    jobs = PostgresVerificationJobRepository(session_factory)
    claims = SqlClaimRepository(session_factory)
    job = jobs.claim_due("verify", now=datetime.now(UTC))[0]
    now = datetime.now(UTC)
    result = VerificationResult(
        claim_id=job.claim_id,
        status="VERIFIED",
        market_snapshot_id="market-1",
        market_data_version="bars.v1",
        fact_date=now.date(),
        adjustment="NONE",
        verification_timestamp=now,
    )
    refresh = VerificationRefreshService(session_factory, jobs, claims)
    with session_factory() as session:
        baseline_artifacts = session.scalar(select(func.count()).select_from(ContentArtifactRow))
    with pytest.raises(RuntimeError):
        refresh.complete(
            job.job_id,
            "verify",
            result,
            parent_snapshot_id=initial["content_snapshot_id"],
            failure_hook=lambda stage: (_ for _ in ()).throw(RuntimeError("inject"))
            if stage == "outbox"
            else None,
        )
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == 0
        assert session.scalar(select(func.count()).select_from(ContentArtifactRow)) == baseline_artifacts
        assert session.scalar(select(func.count()).select_from(ContentSnapshotRow)) == 1
        assert session.scalar(select(func.count()).select_from(SignalOutboxRow)) == 0
    # Lease was rolled back, so the same job can be completed successfully.
    job = jobs.claim_due("verify", now=now + timedelta(seconds=61))[0]
    first = refresh.complete(
        job.job_id,
        "verify",
        result,
        parent_snapshot_id=initial["content_snapshot_id"],
        now=now,
    )
    assert first["snapshot_id"] and first["signal_id"]
    duplicate = refresh.complete(
        job.job_id,
        "other",
        result,
        parent_snapshot_id=initial["content_snapshot_id"],
        now=now,
    )
    assert duplicate["idempotent"] is True
    jobs.requeue(job.job_id, now=now + timedelta(seconds=1))
    job = jobs.claim_due("verify-2", now=now + timedelta(seconds=1))[0]
    second_result = result.model_copy(update={"market_snapshot_id": "market-2"})
    second = refresh.complete(
        job.job_id,
        "verify-2",
        second_result,
        parent_snapshot_id=first["snapshot_id"],
        now=now + timedelta(seconds=1),
    )
    assert second["snapshot_id"] != first["snapshot_id"]
    # A duplicate of result A must resolve to S2 even after result B created S3.
    duplicate_after_s3 = refresh.complete(
        job.job_id,
        "verify-duplicate",
        result,
        parent_snapshot_id=first["snapshot_id"],
        now=now + timedelta(seconds=2),
    )
    assert duplicate_after_s3["idempotent"] is True
    assert duplicate_after_s3["snapshot_id"] == first["snapshot_id"]
    assert duplicate_after_s3["signal_id"] == first["signal_id"]
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ContentSnapshotRow)) == 3
        assert session.scalar(select(func.count()).select_from(ClaimVerificationResultRow)) == 2

from datetime import date, datetime, timedelta, timezone

import pytest

from stock_content.application.pipeline import PipelineContext
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.stages import (
    ContentSnapshotPersistError,
    SnapshotRecordingStage,
    TemporalNormalizationStage,
)
from stock_content.domain.artifacts import SourceArtifact
from stock_content.domain.claim_draft import ClaimOccurrenceDraft, TemporalExpressionDraft
from stock_content.domain.temporal_semantics import (
    CalendarType,
    ClaimTemporalBinding,
    TemporalRole,
    TemporalScope,
    TemporalValueType,
)
from stock_content.ports.temporal_reference import ExchangeCalendarRef, FiscalCalendarRef, ResolvedPeriod
from stock_content.ports.temporal_reference_snapshot import (
    PinnedTemporalReferenceProvider,
    TemporalReferenceSnapshotNotFoundError,
)

AVAILABLE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class ProductionReference:
    def __init__(self, latest="R2"):
        self.latest = latest
        self.resolve_calls = []
        self.snapshot_calls = []

    def resolve_fiscal_calendar(self, subject_key, as_of):
        self.resolve_calls.append(("fiscal_calendar", subject_key))
        return FiscalCalendarRef("issuer-fy", self.latest, "v2", AVAILABLE, subject_key)

    def resolve_period(self, subject_key, period_label, as_of):
        self.resolve_calls.append(("period", subject_key, period_label))
        return ResolvedPeriod(
            date(2027, 4, 1), date(2027, 6, 30), period_label, "FISCAL", "issuer-fy",
            self.latest, "v2", AVAILABLE, subject_key,
        )

    def resolve_exchange_calendar(self, subject_key, as_of):
        self.resolve_calls.append(("exchange", subject_key))
        return ExchangeCalendarRef("XSHG", "Asia/Shanghai", self.latest, "v2", AVAILABLE, subject_key)

    def get_fiscal_calendar_snapshot(self, reference_snapshot_id):
        raise TemporalReferenceSnapshotNotFoundError(reference_snapshot_id)

    def get_exchange_calendar_snapshot(self, reference_snapshot_id):
        raise TemporalReferenceSnapshotNotFoundError(reference_snapshot_id)

    def get_period_snapshot(self, reference_snapshot_id, *, subject_key, period_label):
        self.snapshot_calls.append((reference_snapshot_id, subject_key, period_label))
        if reference_snapshot_id != "R1":
            raise TemporalReferenceSnapshotNotFoundError(reference_snapshot_id)
        return ResolvedPeriod(
            date(2027, 4, 1), date(2027, 6, 30), period_label, "FISCAL", "issuer-fy",
            "R1", "v1", AVAILABLE, subject_key,
        )


def _draft():
    return ClaimOccurrenceDraft(
        semantic_segment_id="seg-1", knowledge_kind="FACT", claim_type="FINANCIAL_METRIC",
        subject_key="ABC", predicate_key="revenue", conclusion="FY2027Q2 revenue",
        temporal_expressions=[TemporalExpressionDraft(
            role="REPORTING_PERIOD", raw_expression="FY2027Q2", confidence=1.0,
        )],
    )


def test_build_application_wires_real_temporal_stage_for_fiscal_and_exchange(tmp_path):
    from stock_content.api.dependencies import build_application

    provider = ProductionReference(latest="R1")
    application = build_application(
        f"sqlite:///{tmp_path / 'wired.db'}", enable_qdrant=False,
        reference_provider=provider, reference_snapshot_provider=provider,
    )
    stage = next(item._stage for item in application._pipeline._stages if item.name == "temporal_normalization")
    context = PipelineContext(
        task_id="wired", source={"type": "fixture", "ref": "wired"},
        options={"as_of": AVAILABLE.isoformat()},
    )
    context.state.claim_drafts = [_draft()]
    stage.execute(context)
    binding = context.state.temporal_bindings[0]
    assert (binding.calendar_type.value, binding.reference_snapshot_id,
            binding.reference_data_version, binding.reference_available_at) == (
                "FISCAL", "R1", "v2", AVAILABLE,
            )
    exchange_context = PipelineContext(
        task_id="wired-exchange", source={"type": "fixture", "ref": "wired"},
        options={"as_of": AVAILABLE.isoformat()},
    )
    exchange_context.state.claim_drafts = [ClaimOccurrenceDraft(
        semantic_segment_id="seg-2", knowledge_kind="FACT", claim_type="PRICE", subject_key="ABC",
        predicate_key="close", conclusion="today close",
        temporal_expressions=[TemporalExpressionDraft(
            role="VALID_AT", raw_expression="今天收盘", confidence=1.0,
        )],
    )]
    stage.execute(exchange_context)
    exchange = exchange_context.state.temporal_bindings[0]
    assert (exchange.calendar_type.value, exchange.market_session, exchange.timezone) == (
        "EXCHANGE", "REGULAR", "Asia/Shanghai",
    )
    assert exchange.reference_snapshot_id == "R1"


def test_reprocess_uses_historical_r1_and_does_not_persist_runtime_provider(tmp_path):
    latest = ProductionReference(latest="R2")
    snapshots = SnapshotService()
    source = SourceArtifact(
        artifact_id="source-1", artifact_type="source", source_type="fixture", source_ref="replay",
        source_content_hash="hash", raw_content_hash="hash", raw_storage_uri="fixture://replay",
    )
    manifest = {"reference_data": [{
        "reference_type": "fiscal_period", "subject_key": "ABC", "period_label": "FY2027Q2",
        "binding_key": "fiscal_period|ABC|FY2027Q2", "reference_snapshot_id": "R1",
        "data_version": "v1", "available_at": AVAILABLE.isoformat(),
    }]}
    snapshot = snapshots.record_from_artifacts(
        source_type="fixture", source_ref="replay", source_content_hash="hash",
        source_artifact_id="source-1", artifact_ids={"source": "source-1"},
        producer_manifest=manifest, created_at=AVAILABLE,
    )
    historical_context = PipelineContext(
        task_id="historical", source={"type": "fixture", "ref": "replay"},
        options={"as_of": AVAILABLE.isoformat()},
    )
    historical_context.state.claim_drafts = [_draft()]
    historical_stage = TemporalNormalizationStage(
        reference_provider=PinnedTemporalReferenceProvider(
            latest, {"fiscal_period|ABC|FY2027Q2": "R1"}
        )
    )
    historical_stage.execute(historical_context)
    historical_binding_id = historical_context.state.temporal_bindings[0].temporal_binding_id

    class Artifacts:
        def get(self, artifact_id):
            return source if artifact_id == "source-1" else None

        def verify(self, artifact_id):
            return None

        def find_task_options_for_snapshot(self, artifact_ids):
            return {"offline_fixture": True, "transcript": "FY2027Q2", "as_of": AVAILABLE.isoformat()}

    class Tasks:
        def __init__(self):
            self.created = []

        def create(self, task):
            self.created.append(task)

        def succeed(self, *args):
            return None

        def fail(self, *args):
            return None

    class Pipeline:
        def __init__(self):
            self.context = None
            self.stage = TemporalNormalizationStage(reference_provider=latest)

        def process(self, context):
            context.state.claim_drafts = [_draft()]
            self.stage.execute(context)
            self.context = context
            return context

    tasks, pipeline = Tasks(), Pipeline()
    replay = ReplayService(
        snapshots, artifact_repository=Artifacts(), task_repository=tasks, pipeline=pipeline,
        temporal_reference_snapshot_provider=latest,
    )
    result = replay.replay(snapshot.content_snapshot_id, mode="REPROCESS")
    binding = pipeline.context.state.temporal_bindings[0]
    assert binding.reference_snapshot_id == "R1"
    assert binding.temporal_binding_id == historical_binding_id
    assert binding.start_date == date(2027, 4, 1)
    assert latest.snapshot_calls
    assert all(item == ("R1", "ABC", "FY2027Q2") for item in latest.snapshot_calls)
    assert not any(item[0] == "period" for item in latest.resolve_calls)
    assert tasks.created and "temporal_reference_provider" not in tasks.created[0].options
    assert result["mode"] == "REPROCESS"


def _captured_real_context(tmp_path, monkeypatch):
    from stock_content.api.dependencies import build_application

    application = build_application(
        f"sqlite:///{tmp_path / 'snapshot-stage.db'}", enable_qdrant=False
    )
    recording = next(
        item._stage for item in application._pipeline._stages if item.name == "content_snapshot"
    )
    captured = []
    original_execute = recording.execute

    def capture(context):
        captured.append(context)
        return original_execute(context)

    monkeypatch.setattr(recording, "execute", capture)
    application.enqueue(
        "bilibili", "snapshot-stage-source", {
            "metadata": {"title": "snapshot stage"},
            "transcript": "股票600000营收增长10%。",
            "offline_fixture": True,
            "as_of": "2026-01-01T00:00:00+00:00",
        },
    )
    result = application.process_next("snapshot-stage-worker")
    assert result["status"] == "SUCCEEDED"
    assert captured
    return application, captured[0], original_execute


def test_snapshot_recording_rejects_late_reference_before_publishing(tmp_path, monkeypatch):
    application, context, original_execute = _captured_real_context(tmp_path, monkeypatch)
    candidate = context.options["snapshot_commit_candidate"]
    late = ClaimTemporalBinding(
        role=TemporalRole.REPORTING_PERIOD,
        scope=TemporalScope.INTERVAL,
        value_type=TemporalValueType.DATE,
        start_date=date(2027, 4, 1), end_date=date(2027, 6, 30),
        period_label="FY2027Q2", calendar_type=CalendarType.FISCAL,
        reference_snapshot_id="R-late", reference_data_version="v1",
        reference_available_at=candidate + timedelta(seconds=1),
        normalization_status="BOUND",
    )
    context.state.temporal_bindings = [late]
    context.state.temporal_bindings_by_draft = {0: [late]}
    before = len(application._snapshots._store.list_for_source("bilibili", "snapshot-stage-source"))
    with pytest.raises(ContentSnapshotPersistError, match="REFERENCE_AS_OF_VIOLATION"):
        original_execute(context)
    after = len(application._snapshots._store.list_for_source("bilibili", "snapshot-stage-source"))
    assert after == before


def test_reference_lineage_stays_out_of_quant_market_snapshot_ids(tmp_path, monkeypatch):
    application, context, _ = _captured_real_context(tmp_path, monkeypatch)
    candidate = context.options["snapshot_commit_candidate"]
    binding = ClaimTemporalBinding(
        role=TemporalRole.REPORTING_PERIOD,
        scope=TemporalScope.INTERVAL,
        value_type=TemporalValueType.DATE,
        start_date=date(2027, 4, 1), end_date=date(2027, 6, 30),
        period_label="FY2027Q2", calendar_type=CalendarType.FISCAL,
        reference_snapshot_id="R-reference", reference_data_version="v1",
        reference_available_at=candidate - timedelta(seconds=1),
        normalization_status="BOUND",
    )
    context.state.temporal_bindings = [binding]
    context.state.temporal_bindings_by_draft = {0: [binding]}
    context.options["quant_market_snapshot_ids"] = ["M-market"]
    snapshots = SnapshotService()
    SnapshotRecordingStage(snapshots).execute(context)
    snapshot = snapshots.get(context.state.content_snapshot_id)
    assert snapshot is not None
    assert snapshot.quant_market_snapshot_ids == ("M-market",)
    assert set(snapshot.external_snapshots) == {"M-market", "R-reference"}
    record = snapshot.producer_manifest["reference_data"][0]
    assert set(record) == {
        "reference_type", "subject_key", "period_label", "binding_key",
        "reference_snapshot_id", "data_version", "available_at",
    }


def test_replay_rejects_non_object_or_incomplete_reference_manifest():
    snapshots = SnapshotService()
    malformed = snapshots.record_from_artifacts(
        source_type="fixture", source_ref="malformed", source_content_hash="hash",
        producer_manifest={"reference_data": [{"reference_type": "fiscal_period"}, "not-an-object"]},
        created_at=AVAILABLE,
    )
    result = ReplayService(snapshots).replay(malformed.content_snapshot_id)
    assert result["error"] == "REPLAY_REFERENCE_SNAPSHOT_MISMATCH"


def test_required_reference_configuration_fails_closed(monkeypatch):
    from stock_content.api.dependencies import build_application

    monkeypatch.setenv("CONTENT_TEMPORAL_REFERENCE_REQUIRED", "true")
    monkeypatch.delenv("CONTENT_TEMPORAL_REFERENCE_ENABLED", raising=False)
    monkeypatch.delenv("CONTENT_TEMPORAL_REFERENCE_URL", raising=False)
    with pytest.raises(ValueError, match="requires an enabled or injected"):
        build_application("sqlite:///should-not-be-created.db", enable_qdrant=False)

    monkeypatch.setenv("CONTENT_TEMPORAL_REFERENCE_ENABLED", "true")
    with pytest.raises(ValueError, match="URL is required"):
        build_application("sqlite:///should-not-be-created.db", enable_qdrant=False)

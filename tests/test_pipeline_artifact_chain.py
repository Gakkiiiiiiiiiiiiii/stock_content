from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ClaimArtifactMemberRow,
    ClaimOccurrenceEvidenceRow,
    ClaimVerificationJobRow,
    ContentArtifactRow,
    ContentSnapshotRow,
    ContentStageCheckpointRow,
    FinancialClaimRow,
    KnowledgeUnitRow,
)
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.api.dependencies import STAGE_VERSIONS, build_application
from stock_content.application.pipeline import PipelineContext
from stock_content.application.snapshot_service import SnapshotService
from stock_content.application.stage_runner import StageResult, StageRunner
from stock_content.application.stages import FrameExtractionStage
from stock_content.domain.artifacts import SourceArtifact
from stock_content.domain.atomic_claim_extractor import AtomicClaimExtractor
from stock_content.domain.checkpoint import CheckpointValidationError
from stock_content.domain.models import KnowledgeUnit
from stock_content.domain.semantic_context_builder import SemanticContext


def _run(
    tmp_path: Path,
    frame_hash: str,
    transcript: str = "股票600000基本面良好。",
) -> tuple[object, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    task = application.enqueue(
        "bilibili",
        "BV-chain",
        {
            "metadata": {"title": "fixture"},
            "transcript": transcript,
            "offline_fixture": True,
            "frames": [{"frame_id": "frame-1", "timestamp_ms": 100, "image_hash": frame_hash}],
            "ocr_evidence": [{"frame_id": "frame-1", "timestamp_ms": 100, "text": "100"}],
            "frame_insights": [{"frame_id": "frame-1", "label": "chart"}],
        },
    )
    result = application.process_next("pipeline-test")
    return application, (task, result)


def test_production_fixture_persists_complete_artifact_claim_dag(tmp_path):
    application, (_, result) = _run(tmp_path, "frame-a", "股票600000营收增长10%。")
    assert result["status"] == "SUCCEEDED"
    repo = application._pipeline._stages[0]._artifact_repository  # noqa: SLF001
    with repo._sessions() as session:  # noqa: SLF001
        artifacts = session.scalars(select(ContentArtifactRow)).all()
        claims = session.scalars(select(FinancialClaimRow)).all()
        evidence = session.scalars(select(ClaimOccurrenceEvidenceRow)).all()
        jobs = session.scalars(select(ClaimVerificationJobRow)).all()
        claim_members = session.scalars(select(ClaimArtifactMemberRow)).all()
        checkpoints = session.scalars(select(ContentStageCheckpointRow)).all()
        snapshots = session.scalars(select(ContentSnapshotRow)).all()
    types = {item.artifact_type for item in artifacts}
    assert {
        "source",
        "media",
        "transcript",
        "frame",
        "ocr",
        "vision",
        "evidence",
        "claims",
        "verification",
        "knowledge",
        "summary",
    } <= types
    assert claims and evidence and jobs and claim_members
    assert all(job.status == "VERIFICATION_PENDING" for job in jobs)
    artifact_ids = {item.artifact_id for item in artifacts}
    source_rows = [item for item in artifacts if item.artifact_type == "source"]
    assert source_rows and all(item.payload.get("raw_content_hash") for item in source_rows)
    for item in artifacts:
        assert set(item.parent_artifact_ids or ()) <= artifact_ids
    assert {item.stage for item in checkpoints} >= {"resolve", "download", "knowledge", "content_snapshot"}
    snapshot = snapshots[-1]
    active = {
        slot: next(item for item in artifacts if item.artifact_id == artifact_id)
        for slot, artifact_id in snapshot.artifact_ids.items()
    }
    assert active["transcript"].artifact_id in active["semantic_segments"].parent_artifact_ids
    assert active["semantic_segments"].artifact_id in active["evidence"].parent_artifact_ids
    assert active["evidence"].artifact_id in active["occurrences"].parent_artifact_ids
    assert active["occurrences"].artifact_id in active["claims"].parent_artifact_ids
    assert active["claims"].artifact_id in active["verification"].parent_artifact_ids
    assert active["verification"].artifact_id in active["lifecycle"].parent_artifact_ids
    assert active["lifecycle"].artifact_id in active["knowledge"].parent_artifact_ids
    assert active["knowledge"].artifact_id in active["summary"].parent_artifact_ids


def test_non_quant_claim_persists_not_verifiable_result_without_job(tmp_path):
    application, (_, result) = _run(tmp_path, "frame-non-quant", "行业增长放缓。")
    assert result["status"] == "SUCCEEDED"
    claim_repository = application._claim_repository  # noqa: SLF001
    with claim_repository._sessions() as session:  # noqa: SLF001
        claims = session.scalars(select(FinancialClaimRow)).all()
        jobs = session.scalars(select(ClaimVerificationJobRow)).all()
        from stock_content.adapters.postgres.models import ClaimVerificationResultRow

        results = session.scalars(select(ClaimVerificationResultRow)).all()
    assert claims
    assert not jobs
    assert results and all(item.status == "NOT_VERIFIABLE" for item in results)


def test_offline_fixture_ticker_policy_is_adapter_only():
    extractor = AtomicClaimExtractor()
    ticker_context = SemanticContext(
        semantic_segment_id="segment-ticker",
        start_ms=0,
        end_ms=1000,
        transcript_segments=[{"segment_index": 0, "text": "股票600000基本面良好。"}],
    )
    opinion_context = SemanticContext(
        semantic_segment_id="segment-opinion",
        start_ms=0,
        end_ms=1000,
        transcript_segments=[{"segment_index": 0, "text": "行业增长放缓。"}],
    )
    assert extractor.extract(ticker_context, offline_fixture=True)[0].claim_type == "FINANCIAL_METRIC"
    assert extractor.extract(opinion_context, offline_fixture=True)[0].claim_type == "INDUSTRY_RELATION"


def test_raw_visual_change_changes_snapshot_with_same_transcript(tmp_path):
    first, first_result = _run(tmp_path / "first", "frame-a")
    second, second_result = _run(tmp_path / "second", "frame-b")
    assert first_result[1]["content_snapshot_id"] != second_result[1]["content_snapshot_id"]


def test_stage_runner_accepts_explicit_stage_result():
    class ExplicitStage:
        name = "explicit"

        def execute(self, context):
            return StageResult(context=context, produced_artifacts=(), metrics={"items": 1.0})

    context = PipelineContext(task_id="explicit", source={})
    result = StageRunner(ExplicitStage()).execute(context)
    assert result.checkpoints[-1].status == "SUCCEEDED"


def test_production_runner_rejects_implicit_registry_diff():
    class LegacyStage:
        name = "legacy"
        required_inputs = ()
        output_types = ()

        def execute(self, context):
            return context

    context = PipelineContext(task_id="legacy", source={})
    try:
        StageRunner(LegacyStage(), legacy_fallback=False).execute(context)
    except TypeError as exc:
        assert "must return StageResult" in str(exc)
    else:
        raise AssertionError("implicit production stage result was accepted")


def test_repeated_visual_outputs_are_optional_but_undeclared_outputs_rejected():
    class EmptyFrames:
        name = "frame"
        required_inputs = ()
        output_types = ("frame",)
        optional_output_types = ("frame",)

        def execute(self, context):
            return StageResult(context=context)

    assert StageRunner(EmptyFrames(), legacy_fallback=False).execute(
        PipelineContext(task_id="empty-frame", source={})
    ).checkpoints[-1].status == "SUCCEEDED"

    class WrongOutput:
        name = "frame"
        required_inputs = ()
        output_types = ("frame",)
        optional_output_types = ("frame",)

        def execute(self, context):
            source = SourceArtifact(artifact_id="source-legacy", artifact_type="source")
            return StageResult(context=context, produced_artifacts=(source,))

    try:
        StageRunner(WrongOutput(), legacy_fallback=False).execute(
            PipelineContext(task_id="wrong-output", source={})
        )
    except ValueError as exc:
        assert "undeclared outputs" in str(exc)
    else:
        raise AssertionError("undeclared output was accepted")


def test_failed_stage_persists_failed_checkpoint(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'failed.db'}")
    database.create_schema()
    repository = SqlArtifactRepository(database.session_factory)

    class FailingStage:
        name = "failing"
        required_inputs = ()
        output_types = ()

        def execute(self, context):
            raise RuntimeError("fixture failure")

    context = PipelineContext(task_id="failed-task", source={})
    try:
        StageRunner(FailingStage(), artifact_repository=repository).execute(context)
    except RuntimeError as exc:
        assert str(exc) == "fixture failure"
    else:
        raise AssertionError("failing stage unexpectedly succeeded")

    with database.session_factory() as session:
        checkpoint = session.get(ContentStageCheckpointRow, "failed-task:failing:1.0.0")
    assert checkpoint is not None
    assert checkpoint.status == "FAILED"


def test_production_sources_do_not_use_legacy_context_data():
    for path in ("src/stock_content/application/stages.py", "src/stock_content/application/service.py"):
        source = Path(path).read_text(encoding="utf-8")
        assert "context.data[" not in source
        assert "context.data.get" not in source


def test_knowledge_units_keep_distinct_claim_links(tmp_path):
    application, _ = _run(tmp_path, "frame-distinct")
    knowledge_runner = next(
        runner for runner in application._pipeline._stages if runner.name == "knowledge"  # noqa: SLF001
    )
    knowledge_stage = knowledge_runner._stage  # noqa: SLF001
    knowledge_stage._fixture_extractor.extract = lambda *_args: [  # noqa: SLF001
        KnowledgeUnit(
            knowledge_uid="unit-a",
            video_id="fixture",
            chapter_id=None,
            statement="甲公司营收增长",
            subject="甲公司",
            predicate_key="revenue",
        ),
        KnowledgeUnit(
            knowledge_uid="unit-b",
            video_id="fixture",
            chapter_id=None,
            statement="乙公司利润稳定",
            subject="乙公司",
            predicate_key="profit",
        ),
    ]
    # A second task exercises the replaced fixture extractor.
    application.enqueue(
        "bilibili",
            "BV1distinct",
            {
                "metadata": {"title": "distinct"},
                "transcript": "甲公司营收增长。乙公司利润稳定。",
                "offline_fixture": True,
            },
    )
    result = application.process_next("distinct")
    assert result["status"] == "SUCCEEDED", result
    repo = application._pipeline._stages[0]._artifact_repository  # noqa: SLF001
    with repo._sessions() as session:  # noqa: SLF001
        rows = session.scalars(select(FinancialClaimRow)).all()
        knowledge_rows = session.scalars(select(KnowledgeUnitRow)).all()
    assert len(rows) >= 2
    claim_ids = {row.claim_id for row in rows}
    # Final canonical claims are deliberately source-independent; evidence
    # is owned by occurrence role memberships instead of claim payloads.
    assert all(not row.payload.get("evidence_refs") for row in rows)
    assert claim_ids
    distinct = {
        row.knowledge_uid: tuple((row.attributes or {}).get("claim_ids") or ())
        for row in knowledge_rows
        if row.knowledge_uid in {"unit-a", "unit-b"}
    }
    assert set(distinct) == {"unit-a", "unit-b"}
    assert all(len(ids) == 1 for ids in distinct.values())
    assert distinct["unit-a"] != distinct["unit-b"]


def test_frame_file_hash_uses_bytes(tmp_path):
    image = tmp_path / "frame.bin"
    image.write_bytes(b"stable-image-bytes")
    expected = __import__("hashlib").sha256(image.read_bytes()).hexdigest()
    assert FrameExtractionStage._image_hash({"image_path": str(image)}) == expected


def test_supplied_visual_evidence_gets_deterministic_frame_parent(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'orphan.db'}", enable_qdrant=False)
    application.enqueue(
        "bilibili",
        "BV1orphan",
        {
            "metadata": {"title": "orphan"},
            "transcript": "股票事实。",
            "offline_fixture": True,
            "ocr_evidence": [{"frame_id": "missing-frame", "text": "100"}],
            "frame_insights": [{"frame_id": "missing-frame", "label": "chart"}],
        },
    )
    result = application.process_next("orphan")
    assert result["status"] == "SUCCEEDED", result
    repository = application._pipeline._stages[0]._artifact_repository  # noqa: SLF001
    with repository._sessions() as session:  # noqa: SLF001
        artifacts = session.scalars(select(ContentArtifactRow)).all()
    artifact_ids = {item.artifact_id for item in artifacts}
    visual = [item for item in artifacts if item.artifact_type in {"ocr", "vision"}]
    assert visual
    assert all(set(item.parent_artifact_ids or ()) <= artifact_ids for item in visual)


def test_snapshot_manifest_is_identity_and_memory_store_is_immutable():
    service = SnapshotService()
    kwargs = {
        "source_type": "fixture",
        "source_ref": "one",
        "source_content_hash": "raw",
        "artifact_ids": {"source": "source-a"},
        "source_artifact_id": "source-a",
        "producer_manifest": {"code_sha": "abc", "container_image": "one"},
        "model_versions": {"asr_model": "asr-1"},
        "prompt_versions": {"extraction": "p1"},
        "configuration": {"temperature": 0},
        "policy_versions": {"claim": "claim-1"},
    }
    first = service.record_from_artifacts(**kwargs)
    assert service.record_from_artifacts(**kwargs).content_snapshot_id == first.content_snapshot_id
    changed = service.record_from_artifacts(
        **{**kwargs, "producer_manifest": {"code_sha": "abc", "container_image": "two"}}
    )
    assert changed.content_snapshot_id != first.content_snapshot_id


def test_retry_failed_checkpoint_preserves_attempt_history(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'retry.db'}")
    database.create_schema()
    repository = SqlArtifactRepository(database.session_factory)
    attempts = iter([False, True])

    class RetryStage:
        name = "retry"
        required_inputs = ()
        output_types = ()

        def execute(self, context):
            if not next(attempts):
                raise RuntimeError("first attempt")
            return StageResult(context=context)

    context = PipelineContext(task_id="retry-task", source={})
    runner = StageRunner(RetryStage(), artifact_repository=repository)
    try:
        runner.execute(context)
    except RuntimeError:
        pass
    runner.execute(context)
    with database.session_factory() as session:
        row = session.get(ContentStageCheckpointRow, "retry-task:retry:1.0.0")
    assert row is not None and row.status == "SUCCEEDED"
    assert len((row.payload or {}).get("attempt_history") or []) == 1


def test_cross_application_resume_does_not_rerun_expensive_prefix(tmp_path):
    """A worker takeover restores the durable prefix from the shared DB."""
    database_url = f"sqlite:///{tmp_path / 'resume.db'}"
    first = build_application(database_url, enable_qdrant=False)
    task = first.enqueue(
        "bilibili",
        "BV-resume",
        {
            "metadata": {"title": "resume"},
            "transcript": "股票600000基本面良好。",
            "offline_fixture": True,
        },
    )
    knowledge_runner = next(runner for runner in first._pipeline._stages if runner.name == "knowledge")  # noqa: SLF001

    def fail_after_expensive_prefix(_context):
        raise RuntimeError("fail once after ASR")

    knowledge_runner._stage.execute = fail_after_expensive_prefix  # noqa: SLF001
    failed = first.process_next("worker-one")
    assert failed["status"] == "FAILED"

    second = build_application(database_url, enable_qdrant=False)
    # Deliberately reverse the durable artifact map.  Active slot selection
    # must still follow checkpoint execution order, not dictionary/set order.
    original_load = second._artifact_repository.load_checkpoints  # noqa: SLF001

    def unordered_load(task_id, stage_versions=None):
        records, persisted = original_load(task_id, stage_versions)
        return records, dict(reversed(list(persisted.items())))

    second._artifact_repository.load_checkpoints = unordered_load  # noqa: SLF001
    restored_task = second._tasks.get(task["task_id"])  # noqa: SLF001
    restored_context = PipelineContext(
        task_id=task["task_id"],
        source={"type": restored_task.source_type, "ref": restored_task.source_ref},
        options=restored_task.options,
    )
    assert second._restore_resume_context(restored_context, restored_task)  # noqa: SLF001
    transcript_checkpoints = [
        record for record in restored_context.checkpoints if record.stage in {"asr", "transcript_postprocess"}
    ]
    assert transcript_checkpoints
    latest_transcript_id = transcript_checkpoints[-1].output_artifact_ids[-1]
    assert restored_context.artifacts.transcript.artifact_id == latest_transcript_id
    assert latest_transcript_id in restored_context.restored_artifacts

    for stage_name in ("download", "frame", "asr", "ocr", "vision"):
        runner = next(runner for runner in second._pipeline._stages if runner.name == stage_name)  # noqa: SLF001

        def must_not_run(_context, *, _stage_name=stage_name):
            raise AssertionError(f"resumed expensive stage reran: {_stage_name}")

        runner._stage.execute = must_not_run  # noqa: SLF001

    resumed = second.process_next("worker-two")
    assert resumed["status"] == "SUCCEEDED", resumed
    assert resumed["task_id"] == task["task_id"]

    clean = build_application(f"sqlite:///{tmp_path / 'clean.db'}", enable_qdrant=False)
    clean.enqueue(
        "bilibili",
        "BV-resume",
        {
            "metadata": {"title": "resume"},
            "transcript": "股票600000基本面良好。",
            "offline_fixture": True,
        },
    )
    clean_result = clean.process_next("clean-worker")
    assert clean_result["status"] == "SUCCEEDED", clean_result
    assert resumed["video_id"] == clean_result["video_id"]
    assert resumed["summary"] == clean_result["summary"]


def test_resume_checkpoint_integrity_rejects_tamper_missing_and_version_drift(tmp_path):
    from stock_content.adapters.postgres.repositories.artifact_repository import ArtifactIntegrityError

    mutations = (
        ("tamper", ArtifactIntegrityError),
        ("missing", ArtifactIntegrityError),
        ("version", CheckpointValidationError),
    )
    for mutation, expected in mutations:
        application, (task, result) = _run(tmp_path / mutation, f"frame-{mutation}")
        assert result["status"] == "SUCCEEDED"
        repository = application._pipeline._stages[0]._artifact_repository  # noqa: SLF001
        with repository._sessions.begin() as session:  # noqa: SLF001
            if mutation == "tamper":
                row = session.scalar(select(ContentArtifactRow).where(ContentArtifactRow.artifact_type == "source"))
                row.payload = {**row.payload, "raw_content_hash": "tampered"}
            elif mutation == "missing":
                row = session.scalar(select(ContentArtifactRow).where(ContentArtifactRow.artifact_type == "source"))
                session.delete(row)
            else:
                checkpoint = session.scalar(
                    select(ContentStageCheckpointRow).where(
                        ContentStageCheckpointRow.task_id == task["task_id"],
                        ContentStageCheckpointRow.stage == "download",
                    )
                )
                checkpoint.stage_version = "drifted"
                checkpoint.payload = {**checkpoint.payload, "stage_version": "drifted"}
        try:
            repository.load_checkpoints(task["task_id"], STAGE_VERSIONS)
        except expected:
            pass
        else:
            raise AssertionError(f"resume integrity mutation was accepted: {mutation}")

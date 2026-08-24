from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect, text

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentArtifactRow
from stock_content.adapters.postgres.repositories.artifact_repository import (
    ArtifactIntegrityError,
    SqlArtifactRepository,
)
from stock_content.adapters.postgres.repositories.claim_repository import SqlClaimRepository
from stock_content.api.dependencies import build_application
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import _config_hash_of, _producer_manifest
from stock_content.domain.artifacts import (
    ArtifactRegistry,
    FrameArtifact,
    MediaArtifact,
    SourceArtifact,
    artifact_id_of,
    artifact_identity_payload,
    deserialize_artifact,
    make_artifact_id,
    serialize_artifact,
)
from stock_content.domain.claims import FinancialClaim
from stock_content.domain.lineage import build_content_snapshot, compute_artifact_root_hash


def _source(**kwargs) -> SourceArtifact:
    payload = {"source_type": "fixture", "source_ref": "one", "raw_content_hash": "raw-1", **kwargs}
    artifact = SourceArtifact(
        artifact_id="source-pending", artifact_type="source", source_type=payload["source_type"],
        source_ref=payload["source_ref"],
        raw_content_hash=payload["raw_content_hash"],
        parent_artifact_ids=kwargs.pop("parent_artifact_ids", ("p-1",)),
    )
    return SourceArtifact(**{**artifact.__dict__, "artifact_id": artifact_id_of(artifact)})


def test_artifact_identity_excludes_id_and_created_at_but_includes_parents():
    first = _source()
    second = SourceArtifact(
        **{**first.__dict__, "artifact_id": "source-other", "created_at": datetime(2030, 1, 1, tzinfo=UTC)}
    )
    assert first.content_hash == second.content_hash
    assert make_artifact_id("source", artifact_identity_payload(first)) == make_artifact_id(
        "source", artifact_identity_payload(second)
    )
    assert _source(parent_artifact_ids=("p-2",)).content_hash != first.content_hash
    assert deserialize_artifact(serialize_artifact(first)).content_hash == first.content_hash


def test_artifact_repository_is_idempotent_and_rejects_conflict(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'artifact.db'}")
    db.create_schema()
    repo = SqlArtifactRepository(db.session_factory)
    source = _source()
    repo.put(source)
    later = SourceArtifact(**{**source.__dict__, "created_at": datetime(2030, 1, 1, tzinfo=UTC)})
    repo.put(later)
    with pytest.raises(ArtifactIntegrityError):
        repo.put(SourceArtifact(**{**source.__dict__, "source_ref": "changed"}))
    assert repo.verify(source.artifact_id)
    assert repo.lineage(source.artifact_id)["artifact"]["artifact_id"] == source.artifact_id


def test_artifact_lineage_recurses_in_stable_parent_order_and_fails_closed(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'artifact-lineage.db'}")
    db.create_schema()
    repo = SqlArtifactRepository(db.session_factory)
    source = _source(parent_artifact_ids=())
    media_raw = MediaArtifact(
        artifact_id="media-pending", artifact_type="media", source_artifact_id=source.artifact_id,
        media_uri="fixture://media", parent_artifact_ids=(source.artifact_id,)
    )
    media = MediaArtifact(**{**media_raw.__dict__, "artifact_id": artifact_id_of(media_raw)})
    frame_raw = FrameArtifact(
        artifact_id="frame-pending", artifact_type="frame", media_artifact_id=media.artifact_id,
        timestamp_ms=1, image_hash="frame", parent_artifact_ids=(media.artifact_id,)
    )
    frame = FrameArtifact(**{**frame_raw.__dict__, "artifact_id": artifact_id_of(frame_raw)})
    for artifact in (source, media, frame):
        repo.put(artifact)
    lineage = repo.lineage(frame.artifact_id)
    assert lineage["lineage_complete"] is True
    assert lineage["parents"][0]["artifact_id"] == media.artifact_id
    assert lineage["parents"][0]["parents"][0]["artifact_id"] == source.artifact_id

    missing_raw = SourceArtifact(
        artifact_id="source-missing", artifact_type="source", source_type="fixture",
        source_ref="missing", raw_content_hash="raw-missing", parent_artifact_ids=("missing-parent",)
    )
    missing = SourceArtifact(**{**missing_raw.__dict__, "artifact_id": "source-missing"})
    repo.put(missing)
    broken = repo.lineage(missing.artifact_id)
    assert broken["lineage_complete"] is False
    assert broken["lineage"] is None
    assert "missing-parent" in broken["lineage_errors"][0]


def test_artifact_lineage_cycle_is_detected_without_partial_graph(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'artifact-cycle.db'}")
    db.create_schema()
    repo = SqlArtifactRepository(db.session_factory)
    first = SourceArtifact(
        artifact_id="cycle-a", artifact_type="source", source_type="fixture",
        source_ref="a", raw_content_hash="raw-a", parent_artifact_ids=("cycle-b",)
    )
    second = SourceArtifact(
        artifact_id="cycle-b", artifact_type="source", source_type="fixture",
        source_ref="b", raw_content_hash="raw-b", parent_artifact_ids=("cycle-a",)
    )
    repo.put(first)
    repo.put(second)
    broken = repo.lineage(first.artifact_id)
    assert broken["lineage_complete"] is False
    assert broken["lineage"] is None
    assert "cycle" in broken["lineage_errors"][0]


def test_producer_manifest_contains_release_and_nested_provenance():
    context = PipelineContext(
        task_id="manifest-test", source={}, options={
            "code_sha": "sha-explicit", "container_digest": "sha256:container",
            "dependency_lock_hash": "lock-explicit", "asr_model": "asr-x",
            "pipeline_config": {"threshold": 0.5}, "entity_alias_version": "alias.v2",
            "producer_manifest": {"code_sha": "manifest-sha", "models": {"llm": "llm-explicit"}},
        }
    )
    manifest = _producer_manifest(context)
    assert manifest["code_sha"] == "sha-explicit"
    assert manifest["container_digest"] == "sha256:container"
    assert manifest["python_lock_hash"] == "lock-explicit"
    assert manifest["models"]["llm"] == "llm-explicit"
    assert manifest["models"]["asr"] == "asr-x"
    assert manifest["prompts"]["extraction"] == "extraction.v1"
    assert manifest["configs"]["entity_alias_version"] == "alias.v2"
    assert manifest["configs"]["config_hash"]


def test_producer_manifest_falls_back_to_environment_and_pipeline_config(monkeypatch):
    monkeypatch.setenv("CONTENT_GIT_COMMIT", "environment-sha")
    context = PipelineContext(
        task_id="manifest-fallback-test",
        source={},
        options={"pipeline_config": {"threshold": 0.5}},
    )

    manifest = _producer_manifest(context)

    assert manifest["code_sha"] == "environment-sha"
    assert manifest["configs"]["config_hash"] == _config_hash_of({"threshold": 0.5})


def test_snapshot_service_keeps_top_level_and_manifest_provenance_identical():
    from stock_content.application.snapshot_service import SnapshotService

    snapshot = SnapshotService().record_from_artifacts(
        source_type="fixture",
        source_ref="provenance",
        source_content_hash="raw",
        artifact_ids={"source": "source-provenance"},
        code_sha="option-sha",
        config_hash="option-config",
        producer_manifest={
            "code_sha": "nested-sha",
            "configs": {"config_hash": "nested-config"},
        },
    )

    assert snapshot.code_sha == "option-sha"
    assert snapshot.config_hash == "option-config"
    assert snapshot.producer_manifest["code_sha"] == snapshot.code_sha
    assert snapshot.producer_manifest["configs"]["config_hash"] == snapshot.config_hash


def test_snapshot_manifest_models_match_effective_model_versions(tmp_path):
    application = build_application(f"sqlite:///{tmp_path / 'manifest-snapshot.db'}", enable_qdrant=False)
    application.enqueue(
        "bilibili",
        "BVmanifest",
        {
            "metadata": {"title": "manifest"},
            "transcript": "600000 收入增长。",
            "offline_fixture": True,
            "code_sha": "pipeline-sha",
            "config_hash": "pipeline-config",
            "pipeline_config": {"threshold": 0.5},
        },
    )
    result = application.process_next("manifest-worker")
    assert result["status"] == "SUCCEEDED"
    snapshot = application._snapshots.get(result["content_snapshot_id"])  # noqa: SLF001
    assert snapshot is not None
    models = snapshot.producer_manifest["models"]
    assert snapshot.model_versions["asr_model"] == models["asr"]
    assert snapshot.model_versions["asr_model_version"] == models["asr_version"]
    assert snapshot.model_versions["ocr_model"] == models["ocr"] == "fixture"
    assert snapshot.model_versions["ocr_model_version"] == models["ocr_version"] == "1"
    assert snapshot.model_versions["vision_model"] == models["vision"]
    assert snapshot.model_versions["llm_model"] == models["llm"]
    assert snapshot.model_versions["embedding_model"] == models["embedding"]
    assert snapshot.code_sha == snapshot.producer_manifest["code_sha"] == "pipeline-sha"
    assert snapshot.config_hash == snapshot.producer_manifest["configs"]["config_hash"] == "pipeline-config"


def test_artifact_repository_rejects_object_and_persisted_hash_tampering(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'artifact-integrity.db'}")
    db.create_schema()
    repo = SqlArtifactRepository(db.session_factory)
    source = _source()
    repo.put(source)

    with pytest.raises(ArtifactIntegrityError):
        repo.put(SourceArtifact(**{**source.__dict__, "content_hash": "tampered"}))
    forged_id = source.artifact_id[:-1] + ("0" if source.artifact_id[-1] != "0" else "1")
    with pytest.raises(ArtifactIntegrityError):
        repo.put(SourceArtifact(**{**source.__dict__, "artifact_id": forged_id}))

    with repo._sessions.begin() as session:  # noqa: SLF001 - deliberate storage tampering test
        row = session.get(ContentArtifactRow, source.artifact_id)
        row.payload = {**row.payload, "content_hash": "tampered"}
    with pytest.raises(ArtifactIntegrityError):
        repo.verify(source.artifact_id)

    with repo._sessions.begin() as session:  # noqa: SLF001 - deliberate storage tampering test
        row = session.get(ContentArtifactRow, source.artifact_id)
        row.payload = {**row.payload, "source_ref": "changed", "content_hash": source.content_hash}
    with pytest.raises(ArtifactIntegrityError):
        repo.verify(source.artifact_id)

    with repo._sessions.begin() as session:  # noqa: SLF001 - deliberate storage tampering test
        row = session.get(ContentArtifactRow, source.artifact_id)
        row.payload = {**row.payload, "source_ref": "one", "artifact_id": "source-tampered"}
    with pytest.raises(ArtifactIntegrityError):
        repo.verify(source.artifact_id)


def test_claim_semantic_id_ignores_evidence_and_roundtrips(tmp_path):
    kwargs = dict(
        claim_type="INFERENCE", subject_type="EQUITY", subject_id="600519.SH", predicate="outlook", value="up",
        source_confidence=0.8, extractor_confidence=0.9,
    )
    first = FinancialClaim(**kwargs, evidence_refs=["ev-1"])
    second = FinancialClaim(**kwargs, evidence_refs=["ev-2"])
    assert first.claim_id == second.claim_id
    db = Database(f"sqlite:///{tmp_path / 'claim.db'}")
    db.create_schema()
    repo = SqlClaimRepository(db.session_factory)
    repo.save(first)
    repo.save(second)
    assert repo.get(first.claim_id).evidence_refs == ["ev-1", "ev-2"]
    assert repo.evidence(first.claim_id) == ["ev-1", "ev-2"]


def test_snapshot_root_and_superseding_identity_are_immutable():
    assert compute_artifact_root_hash({"a": "1", "b": "2"}) == compute_artifact_root_hash({"b": "2", "a": "1"})
    first = build_content_snapshot(
        source_type="fixture", source_ref="one", source_content_hash="raw", artifact_ids={"source": "s"}, code_sha="sha"
    )
    second = build_content_snapshot(
        source_type="fixture", source_ref="one", source_content_hash="raw",
        artifact_ids={"source": "s", "verification": "v"}, code_sha="sha", kind="REFRESH",
        supersedes_snapshot_id=first.content_snapshot_id,
    )
    assert first.content_snapshot_id != second.content_snapshot_id
    assert second.supersedes_snapshot_id == first.content_snapshot_id


def test_visual_registry_ids_are_stable_and_part_of_root():
    registry = ArtifactRegistry(source=_source())
    frame = FrameArtifact(
        artifact_id="frame-1", artifact_type="frame", media_artifact_id="media-1", timestamp_ms=100, image_hash="a"
    )
    registry.add("frames", frame)
    first = registry.artifact_ids()
    changed = FrameArtifact(
        artifact_id="frame-2", artifact_type="frame", media_artifact_id="media-1", timestamp_ms=100, image_hash="b"
    )
    registry = ArtifactRegistry(source=_source())
    registry.add("frames", changed)
    assert first["frames:0"] != registry.artifact_ids()["frames:0"]
    assert compute_artifact_root_hash(first) != compute_artifact_root_hash(registry.artifact_ids())


def test_create_schema_upgrades_legacy_snapshot_and_claim_tables(tmp_path):
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE content_snapshot (
                content_snapshot_id VARCHAR(80) PRIMARY KEY, source_type VARCHAR(32), source_ref TEXT,
                source_content_hash VARCHAR(64), identity JSON, artifact_ids JSON,
                quant_market_snapshot_ids JSON, pipeline_version VARCHAR(40), schema_version VARCHAR(40),
                code_sha VARCHAR(64), config_hash VARCHAR(64), created_at TIMESTAMP
            )
        """))
        connection.execute(text("""
            CREATE TABLE financial_claim (
                claim_id VARCHAR(80) PRIMARY KEY, claim_type VARCHAR(40), fact_category VARCHAR(20),
                subject_type VARCHAR(40), subject_id VARCHAR(128), predicate VARCHAR(255), value JSON,
                unit VARCHAR(40), fact_time TIMESTAMP, published_at TIMESTAMP, evidence_refs JSON,
                source_confidence FLOAT, extractor_confidence FLOAT, created_at TIMESTAMP
            )
        """))
    Database(url).create_schema()
    columns = {
        table: {item["name"] for item in inspect(Database(url).engine).get_columns(table)}
        for table in ("content_snapshot", "financial_claim")
    }
    assert {"source_artifact_id", "artifact_root_hash", "snapshot_kind", "producer_manifest"} <= columns[
        "content_snapshot"
    ]
    assert {"currency", "period_start", "extraction_model_id", "claim_schema_version", "payload"} <= columns[
        "financial_claim"
    ]

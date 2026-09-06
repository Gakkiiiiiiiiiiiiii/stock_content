from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentArtifactRow
from stock_content.adapters.postgres.repositories.artifact_repository import (
    ArtifactIntegrityError,
    SqlArtifactRepository,
)
from stock_content.application.pipeline import PipelineContext
from stock_content.application.service import ContentApplication
from stock_content.application.stages import ASRStage, DownloadStage, TranscriptPostprocessStage
from stock_content.domain.artifacts import SourceArtifact, artifact_id_of
from stock_content.domain.bitemporal_query import FormalContentSignalQueryV2
from stock_content.domain.governance_evidence import redact_pii, validate_governance_evidence
from stock_content.domain.lineage import build_content_snapshot
from stock_content.domain.retention_policy import RetentionPolicy
from stock_content.domain.source_policy import allow_source, policy_for_source
from stock_content.domain.transcript_postprocessor import TranscriptPostprocessor


def _governed_context() -> PipelineContext:
    policy = policy_for_source("bilibili")
    return PipelineContext(
        "governed-source",
        source={"type": "bilibili", "ref": "BV-governed"},
        options={
            "transcript": "contact a@example.com or 13800138000",
            "source_artifact_metadata_required": True,
            "source_policy_version": policy.policy_version,
            "retention_class": policy.retention_class,
            "access_classification": policy.access_classification.value,
        },
    )


def _source_without_governance() -> SourceArtifact:
    source = SourceArtifact(
        artifact_id="source-pending",
        artifact_type="source",
        source_type="bilibili",
        source_ref="BV-formal",
        source_content_hash="a" * 64,
        raw_content_hash="a" * 64,
    )
    return replace(source, artifact_id=artifact_id_of(source))


def _formal_query(snapshot_id: str) -> FormalContentSignalQueryV2:
    instant = datetime(2026, 1, 2, tzinfo=UTC)
    return FormalContentSignalQueryV2(
        contract="content-factor-signal.v5.1",
        request_id="governance-evidence",
        symbols=["600000.SH"],
        business_as_of=instant,
        knowledge_as_of=instant,
        availability_as_of=instant,
        content_snapshot_id=snapshot_id,
        pit_mode="PUBLIC_STRICT",
        min_support=1,
    )


def test_governed_source_evidence_is_bound_to_immutable_source_artifact():
    context = _governed_context()
    DownloadStage({}).execute(context)
    source = context.artifacts.source
    assert source is not None
    evidence = validate_governance_evidence(source.source_metadata, source_type="bilibili")
    assert evidence["license_class"] == "public_terms"
    assert evidence["robots_or_terms_reference"].endswith("robots.txt")
    with pytest.raises(ValueError, match="retention_class"):
        validate_governance_evidence({**source.source_metadata, "retention_class": "tampered"})

    changed_metadata = dict(source.source_metadata)
    changed_metadata["governance_evidence"] = {
        **evidence,
        "license_class": "tampered-license",
    }
    changed = SourceArtifact(
        artifact_id="source-pending",
        artifact_type="source",
        source_type=source.source_type,
        source_ref=source.source_ref,
        source_content_hash=source.source_content_hash,
        raw_content_hash=source.raw_content_hash,
        raw_content_length=source.raw_content_length,
        raw_storage_uri=source.raw_storage_uri,
        source_identity_hash=source.source_identity_hash,
        source_version_id=source.source_version_id,
        source_metadata=changed_metadata,
    )
    assert changed.content_hash != source.content_hash


def test_unlicensed_or_disallowed_source_is_denied():
    policy = policy_for_source("bilibili")
    assert not allow_source(policy, "formal_publish")
    with pytest.raises(ValueError, match="not governed"):
        policy_for_source("unlicensed-feed")


def test_retention_expiry_normalizes_naive_inputs():
    policy = RetentionPolicy("short", timedelta(days=1), derived_retain_for=timedelta(hours=6))
    created_at = datetime(2026, 1, 1)
    assert policy.is_expired(created_at, now=datetime(2026, 1, 2))
    assert policy.derived_expires_at(created_at) == datetime(2026, 1, 1, 6, tzinfo=UTC)


def test_pii_is_redacted_before_the_transcript_artifact_is_created():
    context = _governed_context()
    ASRStage(SimpleNamespace()).execute(context)
    TranscriptPostprocessStage(TranscriptPostprocessor()).execute(context)
    transcript = context.artifacts.transcript
    assert transcript is not None
    text = transcript.segments[0].text
    assert "a@example.com" not in text and "13800138000" not in text
    assert "[REDACTED:EMAIL]" in text and "[REDACTED:CN_MOBILE]" in text
    assert context.state.quality_warnings == ["PII_REDACTED:CN_MOBILE", "PII_REDACTED:EMAIL"]
    assert redact_pii("11010519491231002X").detected_types == ("CN_NATIONAL_ID",)


def test_artifact_repository_detects_governance_evidence_tampering(tmp_path):
    context = _governed_context()
    DownloadStage({}).execute(context)
    source = context.artifacts.source
    assert source is not None
    database = Database(f"sqlite:///{tmp_path / 'governance-integrity.db'}")
    database.create_schema()
    repository = SqlArtifactRepository(database.session_factory)
    repository.put(source)
    with database.session_factory.begin() as session:
        row = session.get(ContentArtifactRow, source.artifact_id)
        assert row is not None
        payload = dict(row.payload)
        payload["source_metadata"] = dict(payload["source_metadata"])
        payload["source_metadata"]["governance_evidence"] = dict(
            payload["source_metadata"]["governance_evidence"]
        )
        payload["source_metadata"]["governance_evidence"]["retention_class"] = "tampered"
        row.payload = payload
    with pytest.raises(ArtifactIntegrityError):
        repository.get(source.artifact_id)


def test_formal_release_fails_closed_when_source_governance_evidence_is_missing(tmp_path):
    source = _source_without_governance()
    database = Database(f"sqlite:///{tmp_path / 'formal-governance.db'}")
    database.create_schema()
    artifacts = SqlArtifactRepository(database.session_factory)
    artifacts.put(source)
    snapshot = build_content_snapshot(
        source_type="bilibili",
        source_ref=source.source_ref,
        source_content_hash=source.raw_content_hash,
        source_artifact_id=source.artifact_id,
        artifact_ids={"source": source.artifact_id},
        code_sha="governance-test",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    application = ContentApplication.__new__(ContentApplication)
    application._artifact_repository = artifacts
    with pytest.raises(ValueError, match="governance evidence"):
        application._validate_formal_snapshot(snapshot, _formal_query(snapshot.content_snapshot_id))

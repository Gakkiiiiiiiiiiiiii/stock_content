from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import (
    ClaimArtifactMemberRow,
    ClaimOccurrenceEvidenceRow,
    ClaimOccurrenceRow,
    ContentArtifactRow,
    ContentSnapshotRow,
    FinancialClaimRow,
    SourceArtifactMetadataRow,
)
from stock_content.adapters.postgres.repositories.claim_event_repository import ClaimStateEventRepository
from stock_content.adapters.postgres.repositories.knowledge_bundle_repository import (
    PostgresKnowledgeBundleAuthority,
    PostgresKnowledgeBundleRepository,
)
from stock_content.application.knowledge_bundle_service import BundleProducerMetadata, KnowledgeBundleService
from stock_content.domain.artifacts import EvidenceItem
from stock_content.domain.claim_state_event import ClaimStateEvent
from stock_content.domain.knowledge_bundle import KnowledgeBundleRequest, canonical_json


def _at(day: int) -> datetime:
    return datetime(2026, 9, day, tzinfo=UTC)


def _request(day: int = 3) -> KnowledgeBundleRequest:
    return KnowledgeBundleRequest(
        content_snapshot_id="snapshot-1",
        query="收入",
        symbol="600000",
        business_as_of=_at(day),
        knowledge_as_of=_at(day),
        availability_as_of=_at(day),
        minimum_support_status="SOURCE_SUPPORTED",
        max_items=10,
    )


def _authority_with_snapshot(tmp_path, *, events: bool = True):
    database = Database(f"sqlite:///{tmp_path / 'bundle-authority.db'}")
    database.create_schema()
    with database.session_factory.begin() as session:
        session.add_all(
            [
                ContentArtifactRow(
                    artifact_id="source-1", artifact_type="source", content_hash="a" * 64,
                    payload={
                        "canonical_url": "https://example.test/video",
                        "source_identity_hash": "source-identity",
                        "source_version_id": "source-version-1",
                    },
                ),
                SourceArtifactMetadataRow(
                    artifact_id="source-1", source_policy_version="source-policy.v1", retention_class="raw_media",
                    access_classification="PUBLIC", source_content_hash="d" * 64, content_size=1,
                    mime_type="video/mp4", canonical_url="https://example.test/video", source_type="bilibili",
                    source_id="fixture-video", source_identity_hash="source-identity",
                    source_version_id="source-version-1",
                    source_available_from=_at(1), pipeline_version="pipeline-test",
                ),
                ContentArtifactRow(
                    artifact_id="claims-1", artifact_type="claims", content_hash="b" * 64, payload={},
                ),
                ContentArtifactRow(
                    artifact_id="evidence-1", artifact_type="evidence", content_hash="c" * 64,
                    # Persist the real EvidenceItem shape: it deliberately
                    # has no per-item content_hash.  The SQL Bundle adapter
                    # must hash the canonical public quote instead.
                    payload={"evidences": [asdict(EvidenceItem(
                        evidence_id="evidence-1", source_type="transcript", source_artifact_id="transcript-1",
                        start_ms=1, end_ms=2, locator={"segment_id": "segment-1"}, normalized_text="收入增长",
                    ))]},
                ),
                ContentSnapshotRow(
                    content_snapshot_id="snapshot-1", source_type="bilibili", source_ref="BV1fixture",
                    source_content_hash="d" * 64, artifact_ids={"claims": "claims-1", "evidence": "evidence-1"},
                    source_artifact_id="source-1", created_at=_at(1),
                ),
                FinancialClaimRow(
                    claim_id="claim-1", claim_type="FACT", fact_category="FACT", subject_type="EQUITY",
                    subject_id="600000", predicate="revenue_growth", value=20, unit="percent",
                    source_confidence=0.9, extractor_confidence=0.9, extraction_model_id="fixture",
                    extraction_prompt_version="fixture", source_support_status="UNSUPPORTED",
                    normalized_statement="收入增长20%", grounding_status="GROUNDED",
                    fact_time=_at(1),
                    claim_schema_version="claim.atomic.v1", legacy_grounding_incomplete=False,
                    legacy_history_incomplete=False,
                ),
                ClaimArtifactMemberRow(member_id="member-1", artifact_id="claims-1", claim_id="claim-1"),
                ClaimOccurrenceRow(
                    occurrence_id="occurrence-1", claim_id="claim-1", source_artifact_id="source-1",
                    transcript_artifact_id="transcript-1", semantic_segment_id="segment-1",
                    assertion_locator_hash="locator", ingested_at=_at(1), extraction_completed_at=_at(1),
                    snapshot_committed_at=_at(1), available_from=_at(1),
                    source_support_status="UNSUPPORTED", source_confidence=0.9, extractor_confidence=0.9,
                    primary_quote="收入增长", normalized_statement="收入增长20%", grounding_status="GROUNDED",
                    claim_schema_version="claim.atomic.v1", legacy_grounding_incomplete=False,
                ),
                ClaimOccurrenceEvidenceRow(
                    occurrence_id="occurrence-1", evidence_id="evidence-1", evidence_role="PRIMARY", ordinal=0,
                ),
            ]
        )
    if events:
        ledger = ClaimStateEventRepository(database.session_factory)
        verification = ClaimStateEvent(
            claim_id="claim-1", event_type="VERIFICATION_INITIAL",
            payload={"snapshot_id": "snapshot-1", "occurrence_id": "occurrence-1", "support_status": "SOURCE_SUPPORTED",
                     "verification_status": "SOURCE_VERIFIED", "available_from": "2026-09-01T00:00:00Z"},
            known_from=_at(1), source_available_from=_at(1),
        )
        active = ClaimStateEvent(
            claim_id="claim-1", event_type="LIFECYCLE", payload={"status": "ACTIVE", "artifact_id": "life-active"},
            known_from=_at(1), business_valid_from=_at(1), source_available_from=_at(1),
            previous_event_hash=verification.event_hash,
        )
        ledger.append(verification)
        ledger.append(active)
        return database, active
    return database, None


def test_sql_bundle_authority_uses_historical_status_not_current_rows(tmp_path):
    database, active = _authority_with_snapshot(tmp_path)
    authority = PostgresKnowledgeBundleAuthority(database.session_factory)

    initial = authority.read_bundle_source(_request())
    assert initial is not None
    assert initial["items"][0]["support_status"] == "SOURCE_SUPPORTED"
    assert initial["items"][0]["verification"]["status"] == "SOURCE_VERIFIED"
    assert initial["items"][0]["lifecycle_status"] == "ACTIVE"

    withdrawn = ClaimStateEvent(
        claim_id="claim-1", event_type="LIFECYCLE", payload={"status": "RETRACTED", "artifact_id": "life-retracted"},
        known_from=_at(4), business_valid_from=_at(1), source_available_from=_at(4),
        previous_event_hash=active.event_hash,
    )
    ClaimStateEventRepository(database.session_factory).append(withdrawn)

    # A later retraction cannot rewrite the old bundle; it does exclude a
    # request whose knowledge and availability clocks can see it.
    assert authority.read_bundle_source(_request(3))["items"]
    assert authority.read_bundle_source(_request(4))["items"] == []


def test_sql_bundle_authority_fails_closed_without_claim_history(tmp_path):
    database, _ = _authority_with_snapshot(tmp_path, events=False)
    with pytest.raises(ValueError, match="HISTORICAL_CLAIM_AUTHORITY_MISSING"):
        PostgresKnowledgeBundleAuthority(database.session_factory).read_bundle_source(_request())


def test_production_sql_evidence_item_bundle_uses_consumer_canonical_quote_hash(tmp_path):
    """Exercise the production SQL authority and persistence path, not a fake authority.

    The consumer owns a separate c14n implementation.  This probe serializes
    a real EvidenceItem through SQL then checks the emitted hash against that
    implementation in a fresh process; it does not import consumer runtime
    code into Content production modules.
    """
    database, _ = _authority_with_snapshot(tmp_path)
    service = KnowledgeBundleService(
        PostgresKnowledgeBundleAuthority(database.session_factory),
        PostgresKnowledgeBundleRepository(database.session_factory),
        BundleProducerMetadata(
            "stock_content", "test", "content-test-sha", "pipeline-test",
            "sha256:EBFD13B78622C3846890438A4FB3CB858278F571FDAB247CDD72EF18CA211621",
        ),
    )
    bundle = service.create(_request())
    citation = bundle["items"][0]["evidence"][0]
    expected = "sha256:" + hashlib.sha256(canonical_json("收入增长").encode("utf-8")).hexdigest()
    assert citation["quote"] == "收入增长"
    assert citation["quote_hash"] == expected

    bundle_path = tmp_path / "production-bundle.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    agent_root = Path(__file__).resolve().parents[2] / "stock_agent-EPIC-043"
    command = (
        "import hashlib,json,sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from app.application.knowledge_conclusion.bundle_validator import canonical_json; "
        "payload=json.load(open(sys.argv[2], encoding='utf-8')); "
        "citation=payload['items'][0]['evidence'][0]; "
        "expected='sha256:'+hashlib.sha256(canonical_json(citation['quote'])).hexdigest(); "
        "raise SystemExit(0 if citation['quote_hash'] == expected else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(agent_root), str(bundle_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

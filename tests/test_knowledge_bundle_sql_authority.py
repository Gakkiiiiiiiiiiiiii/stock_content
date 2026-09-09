from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

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
    _v2_semantics,
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
                    artifact_id="transcript-1", artifact_type="transcript", content_hash="t" * 64, payload={},
                ),
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


def test_v2_sql_authority_keeps_review_blocked_extracted_row_for_quality_only(tmp_path):
    database, active = _authority_with_snapshot(tmp_path)
    with database.session_factory.begin() as session:
        claim = session.get(FinancialClaimRow, "claim-1")
        claim.payload = {
            "bundle_v2": {
                "claim_nature": "PRESCRIPTIVE_RISK_LIMIT",
                "primary_domain": "PORTFOLIO_RISK_MANAGEMENT",
                "attribution": {"attributed": True, "source_label": "slide; ASR differs materially"},
                "source_grade": "SOURCE_ASSERTION",
                "detail": {"explanation": "ASR and OCR contain materially different numeric limits."},
                "temporal": {
                    "kind": "UNKNOWN", "start": None, "end": None, "as_of": None,
                    "rule": None, "label": None, "precision": "UNKNOWN", "explicitly_unknown": True,
                },
                "occurrence_review": {
                    "status": "HUMAN_REVIEW_REQUIRED",
                    "reason_codes": ["ASR_OCR_NUMERIC_CONFLICT"],
                },
                "external_truth_status": "NOT_CHECKED",
            }
        }
    extracted = ClaimStateEvent(
        claim_id="claim-1", event_type="LIFECYCLE",
        payload={"status": "EXTRACTED", "artifact_id": "life-review-blocked"},
        known_from=_at(4), business_valid_from=_at(1), source_available_from=_at(4),
        previous_event_hash=active.event_hash,
    )
    ClaimStateEventRepository(database.session_factory).append(extracted)
    authority = PostgresKnowledgeBundleAuthority(database.session_factory)

    v1 = authority.read_bundle_source(_request(4))
    v2 = authority.read_bundle_source(replace(_request(4), contract_version="content-knowledge-bundle.v2"))

    assert v1["items"] == []
    assert len(v2["items"]) == 1
    assert v2["items"][0]["lifecycle_status"] == "EXTRACTED"
    assert v2["items"][0]["occurrence_review"]["status"] == "HUMAN_REVIEW_REQUIRED"


def test_sql_bundle_authority_fails_closed_without_claim_history(tmp_path):
    database, _ = _authority_with_snapshot(tmp_path, events=False)
    with pytest.raises(ValueError, match="HISTORICAL_CLAIM_AUTHORITY_MISSING"):
        PostgresKnowledgeBundleAuthority(database.session_factory).read_bundle_source(_request())


def test_v2_sql_projection_normalizes_legacy_bare_year_without_mutating_its_snapshot_payload():
    raw_temporal = {
        "kind": "FORECAST_TARGET", "start": "2030", "end": None, "as_of": None,
        "rule": None, "label": "2030", "precision": "YEAR", "explicitly_unknown": False,
    }
    semantic = _v2_semantics(
        SimpleNamespace(
            payload={
                "bundle_v2": {
                    "claim_nature": "FORECAST",
                    "primary_domain": "INFORMATION_INFRASTRUCTURE_POLICY",
                    "attribution": {"attributed": True, "source_label": "source_speaker"},
                    "source_grade": "SOURCE_ASSERTION",
                    "detail": {"explanation": "目标对应信息基础设施建设需求。"},
                    "temporal": raw_temporal,
                    "external_truth_status": "NOT_CHECKED",
                }
            },
            fact_category="FORECAST",
            claim_type="FORECAST",
            grounding_reason_codes=[],
            normalized_statement="信息基础设施投资预计到2030年达到三万亿元",
            predicate="investment_target",
        ),
        SimpleNamespace(provenance={}),
    )
    assert semantic["temporal"]["start"] == "2030-01-01T00:00:00Z"
    assert semantic["temporal"]["end"] == "2030-12-31T23:59:59.999999Z"
    assert raw_temporal["start"] == "2030"


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


@pytest.mark.parametrize(
    ("claim_nature", "source_label", "expected_visual"),
    [
        ("ATTRIBUTED_SECONDARY_POLICY_REPORT", "displayed secondary policy page", True),
        ("ATTRIBUTED_SECONDARY_MACRO_FACT_REPORT", "displayed secondary macro page", True),
        # A speaker forecast/thesis may be adjacent to a news page, but it is
        # not a proposition about what that page reported (KU10/KU09).
        ("SOURCE_FORECAST", "displayed secondary page", False),
        ("SOURCE_INTERPRETIVE_CAUSAL_THESIS", "speaker interpretation", False),
    ],
)
def test_v2_sql_projection_adds_only_owned_displayed_secondary_page_evidence(
    tmp_path, claim_nature, source_label, expected_visual
):
    database, _ = _authority_with_snapshot(tmp_path)
    with database.session_factory.begin() as session:
        snapshot = session.get(ContentSnapshotRow, "snapshot-1")
        claim = session.get(FinancialClaimRow, "claim-1")
        occurrence = session.get(ClaimOccurrenceRow, "occurrence-1")
        occurrence.semantic_segment_id = "segment-1"
        claim.normalized_statement = "2030年目标9800 EFLOPS"
        claim.payload = {
            "bundle_v2": {
                "claim_nature": claim_nature,
                "primary_domain": "INFORMATION_INFRASTRUCTURE_POLICY",
                "attribution": {"attributed": True, "source_label": source_label},
                "source_grade": "SECONDARY",
                "detail": {"explanation": "展示页面写有2030和9800。"},
                "temporal": {
                    "kind": "FORECAST_TARGET", "start": "2030", "end": None,
                    "as_of": None, "rule": None, "label": "2030", "precision": "YEAR",
                    "explicitly_unknown": False,
                },
                "external_truth_status": "NOT_CHECKED",
            }
        }
        session.add_all(
            [
                ContentArtifactRow(
                    artifact_id="frame-page", artifact_type="frame", content_hash="f" * 64,
                    payload={"frame_id": "frame-page", "timestamp_ms": 2},
                ),
                ContentArtifactRow(
                    artifact_id="vision-page", artifact_type="vision", content_hash="v" * 64,
                    payload={
                        "frame_artifact_id": "frame-page", "frame_id": "frame-page", "timestamp_ms": 2,
                        "semantic_segment_ids": ["segment-1"], "labels": ["secondary-news-page"],
                        "label": "displayed secondary page", "model_name": "terra", "model_version": "1",
                    },
                ),
                ContentArtifactRow(
                    artifact_id="ocr-page", artifact_type="ocr", content_hash="o" * 64,
                    payload={
                        "frame_artifact_id": "frame-page", "frame_id": "frame-page", "timestamp_ms": 2,
                        "text": "2030年目标9800 EFLOPS", "bbox": [0, 0, 1, 1],
                        "engine": "paddleocr", "engine_version": "3.7.0",
                    },
                ),
            ]
        )
        snapshot.artifact_ids = {
            **snapshot.artifact_ids,
            "frames:0": "frame-page",
            "vision:0": "vision-page",
        }
    request = replace(_request(), contract_version="content-knowledge-bundle.v2")
    item = PostgresKnowledgeBundleAuthority(database.session_factory).read_bundle_source(request)["items"][0]
    modalities = [entry["modality"] for entry in item["evidence"]]
    assert ("frame" in modalities) is expected_visual
    assert ("ocr" in modalities) is expected_visual
    assert ("vision" in modalities) is expected_visual
    assert item["source_grade"] == "SECONDARY"
    assert item["external_truth_status"] == "NOT_CHECKED"

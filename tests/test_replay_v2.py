"""Replay V2 golden integrity and structured error coverage."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import InMemorySnapshotStore, SnapshotService
from stock_content.domain.artifacts import (
    ArtifactBase,
    ClaimArtifact,
    EvidenceArtifact,
    EvidenceItem,
    VerificationArtifact,
)
from stock_content.domain.claims import FinancialClaim, VerificationResult


class ArtifactRepo:
    def __init__(self, artifacts):
        self.items = {item.artifact_id: item for item in artifacts}

    def get(self, artifact_id):
        return self.items.get(artifact_id)

    def verify(self, artifact_id):
        if artifact_id not in self.items:
            raise KeyError(artifact_id)
        return True


class ClaimRepo:
    def __init__(self, claims):
        self.items = {item.claim_id: item for item in claims}

    def get(self, claim_id):
        return self.items.get(claim_id)


class SignalRows:
    def __init__(self, rows):
        self.rows = rows

    def list_for_snapshot(self, snapshot_id):
        return self.rows


def _snapshot(artifact_ids, *, store=None):
    service = SnapshotService(store or InMemorySnapshotStore())
    return service, service.record_from_artifacts(
        source_type="fixture",
        source_ref="replay",
        source_content_hash="raw-hash",
        artifact_ids=artifact_ids,
        source_artifact_id=artifact_ids.get("source", ""),
        code_sha="test-sha",
    )


def test_verify_lineage_walks_all_parent_edges_and_detects_cycle():
    first = ArtifactBase(artifact_id="artifact-a", artifact_type="source", parent_artifact_ids=("artifact-b",))
    second = ArtifactBase(artifact_id="artifact-b", artifact_type="media", parent_artifact_ids=("artifact-a",))
    snapshots, snapshot = _snapshot({"source": first.artifact_id})
    result = ReplayService(
        snapshots, artifact_repository=ArtifactRepo([first, second])
    ).replay(snapshot.content_snapshot_id)
    assert result["error"] == "REPLAY_LINEAGE_CYCLE"


def test_verify_lineage_reports_missing_artifact_with_stable_code():
    snapshots, snapshot = _snapshot({"source": "missing-source"})
    result = ReplayService(snapshots, artifact_repository=ArtifactRepo([])).replay(snapshot.content_snapshot_id)
    assert result["error"] == "REPLAY_ARTIFACT_MISSING"


def test_verify_lineage_checks_evidence_claim_and_verification_edges():
    evidence = EvidenceArtifact(
        artifact_id="evidence-1", artifact_type="evidence", evidences=(EvidenceItem("e-1", "TRANSCRIPT"),)
    )
    claim = FinancialClaim(
        claim_type="FINANCIAL_METRIC", subject_type="EQUITY", subject_id="600000",
        predicate="revenue", value=1, evidence_refs=["e-missing"], source_confidence=1, extractor_confidence=1,
    )
    claims = ClaimArtifact(
        artifact_id="claims-1", artifact_type="claims", evidence_artifact_id=evidence.artifact_id,
        claims=[claim.claim_id], parent_artifact_ids=(evidence.artifact_id,),
    )
    verification = VerificationArtifact(
        artifact_id="verification-1", artifact_type="verification", claim_artifact_id=claims.artifact_id,
        results=[VerificationResult(claim_id="claim-missing", status="VERIFICATION_PENDING")],
        parent_artifact_ids=(claims.artifact_id,),
    )
    snapshots, snapshot = _snapshot({"source": evidence.artifact_id, "claims": claims.artifact_id,
                                     "verification": verification.artifact_id})
    repo = ArtifactRepo([evidence, claims, verification])
    result = ReplayService(snapshots, artifact_repository=repo, claim_repository=ClaimRepo([claim])).replay(
        snapshot.content_snapshot_id
    )
    assert result["error"] == "REPLAY_LINEAGE_REFERENCE_MISSING"


def test_verify_lineage_checks_verification_claim_reference():
    evidence = EvidenceArtifact(
        artifact_id="evidence-2", artifact_type="evidence", evidences=(EvidenceItem("e-2", "TRANSCRIPT"),)
    )
    claim = FinancialClaim(
        claim_type="FINANCIAL_METRIC", subject_type="EQUITY", subject_id="600000",
        predicate="revenue", value=1, evidence_refs=["e-2"], source_confidence=1, extractor_confidence=1,
    )
    claims = ClaimArtifact(
        artifact_id="claims-2", artifact_type="claims", evidence_artifact_id=evidence.artifact_id,
        claims=[claim.claim_id], parent_artifact_ids=(evidence.artifact_id,),
    )
    verification = VerificationArtifact(
        artifact_id="verification-2", artifact_type="verification", claim_artifact_id=claims.artifact_id,
        results=[VerificationResult(claim_id="claim-missing", status="VERIFICATION_PENDING")],
        parent_artifact_ids=(claims.artifact_id,),
    )
    snapshots, snapshot = _snapshot({"source": evidence.artifact_id, "claims": claims.artifact_id,
                                     "verification": verification.artifact_id})
    result = ReplayService(
        snapshots,
        artifact_repository=ArtifactRepo([evidence, claims, verification]),
        claim_repository=ClaimRepo([claim]),
    ).replay(snapshot.content_snapshot_id)
    assert result["error"] == "REPLAY_LINEAGE_REFERENCE_MISSING"


def test_verify_lineage_checks_signal_snapshot_claim_and_verification_refs():
    source = ArtifactBase(artifact_id="source-signal", artifact_type="source")
    claims = ClaimArtifact(artifact_id="claims-signal", artifact_type="claims", claims=["claim-1"],
                           parent_artifact_ids=(source.artifact_id,))
    verification = VerificationArtifact(artifact_id="verification-signal", artifact_type="verification",
                                         claim_artifact_id=claims.artifact_id, results=[],
                                         parent_artifact_ids=(claims.artifact_id,))
    snapshots, snapshot = _snapshot({"source": source.artifact_id, "claims": claims.artifact_id,
                                     "verification": verification.artifact_id})
    row = SimpleNamespace(payload={"content_snapshot_id": snapshot.content_snapshot_id,
                                   "claim_id": "claim-missing",
                                   "verification_artifact_id": verification.artifact_id})
    result = ReplayService(
        snapshots,
        artifact_repository=ArtifactRepo([source, claims, verification]),
        signal_outbox=SignalRows([row]),
    ).replay(snapshot.content_snapshot_id)
    assert result["error"] == "REPLAY_LINEAGE_REFERENCE_MISSING"


def test_identity_mismatch_is_fail_closed_for_reprocess():
    source = ArtifactBase(artifact_id="source-1", artifact_type="source")
    store = InMemorySnapshotStore()
    snapshots, snapshot = _snapshot({"source": source.artifact_id}, store=store)
    store._snapshots.pop(snapshot.content_snapshot_id)
    store._snapshots["cs-tampered"] = replace(snapshot, content_snapshot_id="cs-tampered")
    result = ReplayService(snapshots, artifact_repository=ArtifactRepo([source]), pipeline=object()).replay(
        "cs-tampered", mode="REPROCESS"
    )
    assert result["error"] == "REPLAY_IDENTITY_MISMATCH"

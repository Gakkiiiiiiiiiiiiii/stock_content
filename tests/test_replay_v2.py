"""Replay V2 golden integrity and structured error coverage."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from stock_content.application.pipeline import PipelineContext
from stock_content.application.replay_service import ReplayService
from stock_content.application.snapshot_service import InMemorySnapshotStore, SnapshotService
from stock_content.application.stages import DownloadStage, ResolveSourceStage
from stock_content.domain.artifacts import (
    ArtifactBase,
    ClaimArtifact,
    EvidenceArtifact,
    EvidenceItem,
    FrameArtifact,
    OCRArtifact,
    SourceArtifact,
    TranscriptVisualCrosscheckArtifact,
    TranscriptVisualCrosscheckRecord,
    VerificationArtifact,
    VisionArtifact,
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


class ReplayArtifactRepo(ArtifactRepo):
    def find_task_options_for_snapshot(self, _artifact_ids):
        return {}


class _UnavailableXiaoeResolver:
    def __init__(self, error: str):
        self.resolve_calls = 0
        self.error = error

    def resolve_materialization(self, *_args, **_kwargs):
        self.resolve_calls += 1
        raise RuntimeError(self.error)


class _SealedMediaPipeline:
    def __init__(self, adapter, snapshot_id):
        self._adapter = adapter
        self._snapshot_id = snapshot_id
        self.context = None

    def process(self, context):
        ResolveSourceStage({"xiaoe": self._adapter}).execute(context)
        DownloadStage({"xiaoe": self._adapter}).execute(context)
        context.state.content_snapshot_id = self._snapshot_id
        self.context = context
        return context


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


def _cs_8b_multimodal_snapshot(*, visual_source_id="frame-8b", include_visual_source=True):
    """The persisted cs-8b shape: visual evidence is admitted by crosscheck.

    The EvidenceArtifact itself predates copying selected visual parents into
    its parent list.  The sealed crosscheck carries the Frame -> OCR/Vision
    graph and the admitted relation instead.
    """
    source = ArtifactBase(artifact_id="source-8b", artifact_type="source")
    media = ArtifactBase(
        artifact_id="media-8b", artifact_type="media", parent_artifact_ids=(source.artifact_id,)
    )
    transcript = ArtifactBase(
        artifact_id="transcript-8b", artifact_type="transcript", parent_artifact_ids=(media.artifact_id,)
    )
    semantic = ArtifactBase(
        artifact_id="semantic-8b", artifact_type="semantic_segments", parent_artifact_ids=(transcript.artifact_id,)
    )
    frame = FrameArtifact(
        artifact_id="frame-8b", artifact_type="frame", media_artifact_id=media.artifact_id,
        frame_id="frame-8b", timestamp_ms=8_805, image_hash="frame-image",
        parent_artifact_ids=(media.artifact_id,),
    )
    ocr = OCRArtifact(
        artifact_id="ocr-8b", artifact_type="ocr", frame_artifact_id=frame.artifact_id,
        frame_id=frame.frame_id, timestamp_ms=frame.timestamp_ms, image_hash=frame.image_hash,
        text="审计证据", parent_artifact_ids=(frame.artifact_id,),
    )
    vision = VisionArtifact(
        artifact_id="vision-8b", artifact_type="vision", frame_artifact_id=frame.artifact_id,
        frame_id=frame.frame_id, timestamp_ms=frame.timestamp_ms, image_hash=frame.image_hash,
        label="审计画面", labels=["审计画面"], parent_artifact_ids=(frame.artifact_id,),
    )
    crosscheck = TranscriptVisualCrosscheckArtifact(
        artifact_id="crosscheck-8b", artifact_type="transcript_visual_crosscheck",
        transcript_artifact_id=transcript.artifact_id, semantic_segment_artifact_id=semantic.artifact_id,
        crosscheck_version="crosscheck.v1", eligible_frame_ids=(frame.frame_id,),
        relations=(TranscriptVisualCrosscheckRecord(
            frame_id=frame.frame_id, frame_artifact_id=frame.artifact_id,
            timestamp_ms=frame.timestamp_ms, relation="SUPPORTS",
        ),),
        parent_artifact_ids=(
            transcript.artifact_id, semantic.artifact_id, frame.artifact_id, ocr.artifact_id, vision.artifact_id,
        ),
    )
    evidence = EvidenceArtifact(
        artifact_id="evidence-8b", artifact_type="evidence", transcript_artifact_id=transcript.artifact_id,
        source_artifact_ids=(transcript.artifact_id,),
        parent_artifact_ids=(transcript.artifact_id, semantic.artifact_id),
        evidences=(
            EvidenceItem("evidence-asr-8b", "ASR", source_artifact_id=transcript.artifact_id),
            EvidenceItem("evidence-frame-8b", "FRAME", source_artifact_id=visual_source_id),
            EvidenceItem("evidence-ocr-8b", "OCR", source_artifact_id=ocr.artifact_id),
            EvidenceItem("evidence-vision-8b", "VISION", source_artifact_id=vision.artifact_id),
        ),
    )
    artifacts = [source, media, transcript, semantic, frame, ocr, vision, crosscheck, evidence]
    artifact_ids = {
        "source": source.artifact_id,
        "media": media.artifact_id,
        "transcript": transcript.artifact_id,
        "semantic_segments": semantic.artifact_id,
        "frames:0": frame.artifact_id,
        "ocr:0": ocr.artifact_id,
        "vision:0": vision.artifact_id,
        "transcript_visual_crosscheck": crosscheck.artifact_id,
        "evidence": evidence.artifact_id,
    }
    if include_visual_source and visual_source_id == "frame-outside-8b":
        outside = FrameArtifact(
            artifact_id=visual_source_id, artifact_type="frame", media_artifact_id=media.artifact_id,
            frame_id="frame-outside-8b", timestamp_ms=9_999, image_hash="outside-image",
            parent_artifact_ids=(media.artifact_id,),
        )
        artifacts.append(outside)
        artifact_ids["frames:1"] = outside.artifact_id
    snapshots, snapshot = _snapshot(artifact_ids)
    return snapshots, snapshot, ArtifactRepo(artifacts)


def _sealed_media_replay(
    tmp_path: Path,
    monkeypatch,
    *,
    uri: Path | None = None,
    content_hash: str | None = None,
    resolver_error: str = "SOURCE_SESSION_EXPIRED",
):
    root = tmp_path / "private-raw"
    root.mkdir()
    media = root / "source.mp4"
    media.write_bytes(b"sealed-media")
    source_path = uri or media
    raw_hash = content_hash or "d82dafc3595aed96bbfaef5899dd166d6888ac6d065e32e53353660ec88e2e60"
    source = SourceArtifact(
        artifact_id="source-sealed", artifact_type="source", source_type="xiaoe", source_ref="p_1/v_1",
        source_content_hash=raw_hash, raw_content_hash=raw_hash, raw_content_length=12,
        raw_storage_uri=str(source_path), source_metadata={
            "canonical_url": "https://a.xiaoeknow.com/p/course/video/v_1?product_id=p_1",
        },
    )
    snapshots = SnapshotService()
    snapshot = snapshots.record_from_artifacts(
        source_type="xiaoe", source_ref="p_1/v_1", source_content_hash=raw_hash,
        artifact_ids={"source": source.artifact_id}, source_artifact_id=source.artifact_id, code_sha="test-sha",
    )
    monkeypatch.setenv("CONTENT_RAW_STORAGE_DIR", str(root))
    adapter = _UnavailableXiaoeResolver(resolver_error)
    pipeline = _SealedMediaPipeline(adapter, snapshot.content_snapshot_id)
    replay = ReplayService(
        snapshots, artifact_repository=ReplayArtifactRepo([source]), pipeline=pipeline,
    )
    return replay, snapshot, source, media, adapter, pipeline


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


def test_cs_8b_shape_accepts_sealed_frame_ocr_and_vision_evidence_for_replay():
    snapshots, snapshot, artifacts = _cs_8b_multimodal_snapshot()

    result = ReplayService(snapshots, artifact_repository=artifacts).replay(snapshot.content_snapshot_id)

    assert result["identity_match"] is True
    assert result["artifact_validation"]["checked"] is True


@pytest.mark.parametrize(
    ("visual_source_id", "include_visual_source", "expected"),
    [
        ("frame-outside-8b", True, "REPLAY_LINEAGE_REFERENCE_INVALID"),
        ("frame-unknown-8b", False, "REPLAY_LINEAGE_REFERENCE_MISSING"),
    ],
)
def test_cs_8b_shape_keeps_unadmitted_or_unknown_visual_evidence_fail_closed(
    visual_source_id, include_visual_source, expected
):
    snapshots, snapshot, artifacts = _cs_8b_multimodal_snapshot(
        visual_source_id=visual_source_id, include_visual_source=include_visual_source
    )

    result = ReplayService(snapshots, artifact_repository=artifacts).replay(snapshot.content_snapshot_id)

    assert result["error"] == expected


@pytest.mark.parametrize("resolver_error", ["SOURCE_PAGE_RESOLVER_DISABLED", "SOURCE_SESSION_EXPIRED"])
def test_migration_replay_reuses_sealed_media_without_xiaoe_resolution(tmp_path, monkeypatch, resolver_error):
    replay, snapshot, _source, media, adapter, pipeline = _sealed_media_replay(
        tmp_path, monkeypatch, resolver_error=resolver_error
    )

    result = replay.replay(
        snapshot.content_snapshot_id, mode="MIGRATION_REPLAY", pipeline_version="pipeline.v4.043.audit"
    )

    assert "error" not in result
    assert adapter.resolve_calls == 0
    assert pipeline.context is not None
    assert pipeline.context.runtime.video_path == media.resolve()


@pytest.mark.parametrize("failure", ["outside", "hash", "missing"])
def test_migration_replay_rejects_unsealed_or_invalid_raw_media(tmp_path, monkeypatch, failure):
    if failure == "outside":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"sealed-media")
        replay, snapshot, *_rest = _sealed_media_replay(tmp_path, monkeypatch, uri=outside)
    elif failure == "hash":
        replay, snapshot, *_rest = _sealed_media_replay(
            tmp_path, monkeypatch, content_hash="0" * 64
        )
    else:
        replay, snapshot, _source, media, *_rest = _sealed_media_replay(tmp_path, monkeypatch)
        media.unlink()
    result = replay.replay(
        snapshot.content_snapshot_id, mode="MIGRATION_REPLAY", pipeline_version="pipeline.v4.043.audit"
    )

    assert result["error"] == "REPLAY_INPUT_UNAVAILABLE"


def test_replay_ignores_request_override_of_a_sealed_media_path(tmp_path, monkeypatch):
    replay, snapshot, _source, media, adapter, pipeline = _sealed_media_replay(tmp_path, monkeypatch)
    attacker_path = tmp_path / "attacker.mp4"
    attacker_path.write_bytes(b"attacker-bytes")

    result = replay.replay(
        snapshot.content_snapshot_id,
        mode="MIGRATION_REPLAY",
        pipeline_version="pipeline.v4.043.audit",
        overrides={"replay_raw_storage_uri": str(attacker_path), "replay_expected_raw_hash": "0" * 64},
    )

    assert "error" not in result
    assert adapter.resolve_calls == 0
    assert pipeline.context.runtime.video_path == media.resolve()


def test_normal_source_resolution_still_calls_the_source_adapter():
    class Adapter:
        def __init__(self):
            self.calls = 0

        def resolve(self, source_ref):
            self.calls += 1
            return {"source_ref": source_ref}

    adapter = Adapter()
    context = PipelineContext(task_id="normal-resolution", source={"type": "xiaoe", "ref": "p_1/v_1"})
    ResolveSourceStage({"xiaoe": adapter}).execute(context)

    assert adapter.calls == 1
    assert context.state.metadata == {"source_ref": "p_1/v_1"}


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

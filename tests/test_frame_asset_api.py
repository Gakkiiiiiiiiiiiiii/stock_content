"""Private sealed-frame HTTP boundary; no media pipeline or database required."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient

from stock_content.api.main import create_app
from stock_content.api.security import ServiceAuthorizer
from stock_content.application.snapshot_service import SnapshotIntegrityError, SnapshotService
from stock_content.domain.artifacts import (
    ClaimOccurrenceArtifact,
    FrameArtifact,
    KnowledgeVisualEvidenceArtifact,
    MediaArtifact,
    OCRArtifact,
    SemanticSegmentArtifact,
    SourceArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    TranscriptVisualCrosscheckArtifact,
    TranscriptVisualCrosscheckRecord,
    VisionArtifact,
    artifact_id_of,
)
from stock_content.domain.semantic_segment import SemanticSegment


class _Artifacts:
    def __init__(self, *items):
        self.items = {item.artifact_id: item for item in items}

    def get(self, artifact_id):
        return self.items.get(artifact_id)

    def verify(self, artifact_id):
        return artifact_id in self.items


class _Service:
    def __init__(self, snapshots, artifacts, occurrences):
        self._snapshots = snapshots
        self._artifact_repository = artifacts
        self._occurrence_repository = occurrences
        self._tasks = None


class _Occurrences:
    def __init__(self, *items):
        self.items = {item.occurrence_id: item for item in items}

    def get(self, occurrence_id):
        return self.items.get(occurrence_id)


def _sealed(item):
    return replace(item, artifact_id=artifact_id_of(item))


def _fixture(
    tmp_path, monkeypatch, *, crosscheck_relation="SUPPORTS_DISPLAYED_MENTION",
    ocr_summary="蓝晓科技", crosscheck_parent_ocr=True, gap_mode="", ambiguous_foreign_timestamp=False,
    wrong_semantic_transcript=False, wrong_crosscheck_transcript=False, crosscheck_parent_transcript=True,
):
    root = tmp_path / "raw"
    frames = root / "frames"
    frames.mkdir(parents=True)
    image = b"\x89PNG\r\n\x1a\n" + b"sealed fixture bytes"
    digest = hashlib.sha256(image).hexdigest()
    path = frames / f"{digest}.png"
    path.write_bytes(image)
    monkeypatch.setenv("CONTENT_RAW_STORAGE_DIR", str(root))
    source = _sealed(SourceArtifact(
        artifact_id="pending", artifact_type="source", source_type="xiaoe",
        source_ref="lesson-1", source_content_hash="a" * 64,
    ))
    media = _sealed(MediaArtifact(
        artifact_id="pending", artifact_type="media", source_artifact_id=source.artifact_id,
        parent_artifact_ids=(source.artifact_id,),
    ))
    transcript = _sealed(TranscriptArtifact(
        artifact_id="pending", artifact_type="transcript", media_artifact_id=media.artifact_id,
        parent_artifact_ids=(media.artifact_id,),
        segments=[TranscriptSegmentItem(
            segment_index=0, start_seconds=0.0, end_seconds=2.0, text="蓝晓科技",
        )],
    ))
    frame = _sealed(FrameArtifact(
        artifact_id="pending", artifact_type="frame", media_artifact_id=media.artifact_id,
        frame_id="frame-1", timestamp_ms=1200, image_hash=digest, storage_ref=str(path),
        evidence_window_ids=("window-1", "window-2"),
        producer_stage="knowledge_frame", planner_version="knowledge-frame.v2",
        planner_request_id="request-1", parent_artifact_ids=(media.artifact_id,),
    ))
    semantic_id = "semseg-fixture"
    semantic = _sealed(SemanticSegmentArtifact(
        artifact_id="pending", artifact_type="semantic_segments",
        transcript_artifact_id="transcript-foreign" if wrong_semantic_transcript else transcript.artifact_id,
        parent_artifact_ids=(transcript.artifact_id,),
        segments=[SemanticSegment(
            semantic_segment_id=semantic_id, transcript_artifact_id=transcript.artifact_id,
            segment_index=0, start_segment_index=0, end_segment_index=0,
            start_segment_id="segment-1", end_segment_id="segment-1", start_ms=0, end_ms=2000,
        )],
    ))
    scoped_relations = [] if gap_mode in {"missing", "mixed_missing"} else [
        TranscriptVisualCrosscheckRecord(
            frame_id=frame.frame_id, frame_artifact_id=frame.artifact_id, timestamp_ms=1200,
            semantic_segment_ids=(semantic_id,), evidence_window_ids=("window-1",),
            relation="UNKNOWN" if gap_mode == "ambiguous" else crosscheck_relation,
        )
    ]
    if gap_mode == "ambiguous":
        scoped_relations.append(TranscriptVisualCrosscheckRecord(
            frame_id=frame.frame_id, frame_artifact_id=frame.artifact_id,
            timestamp_ms=1300 if ambiguous_foreign_timestamp else 1200,
            semantic_segment_ids=(semantic_id,), evidence_window_ids=("window-1",), relation="UNRELATED",
        ))
    scoped_relations.append(TranscriptVisualCrosscheckRecord(
        frame_id=frame.frame_id, frame_artifact_id=frame.artifact_id, timestamp_ms=1200,
        semantic_segment_ids=(semantic_id,), evidence_window_ids=("window-2",),
        relation="CONTRADICTS" if gap_mode == "mixed_missing" else "UNRELATED",
    ))
    crosscheck = _sealed(TranscriptVisualCrosscheckArtifact(
        artifact_id="pending", artifact_type="transcript_visual_crosscheck",
        transcript_artifact_id="transcript-foreign" if wrong_crosscheck_transcript else transcript.artifact_id,
        semantic_segment_artifact_id=semantic.artifact_id,
        relations=tuple(scoped_relations),
        eligible_frame_ids=(frame.frame_id,) if gap_mode == "mixed_missing" else (),
    ))
    ocr = _sealed(OCRArtifact(
        artifact_id="pending", artifact_type="ocr", frame_artifact_id=frame.artifact_id,
        frame_id=frame.frame_id, timestamp_ms=1200, image_hash=digest,
        evidence_window_ids=("window-1", "window-2"), text="蓝晓科技", confidence_score=0.93,
        engine="paddleocr", engine_version="3.7", parent_artifact_ids=(frame.artifact_id,),
    ))
    vision = _sealed(VisionArtifact(
        artifact_id="pending", artifact_type="vision", frame_artifact_id=frame.artifact_id,
        frame_id=frame.frame_id, timestamp_ms=1200, image_hash=digest,
        evidence_window_ids=("window-1", "window-2"), label="company list", confidence_score=0.81,
        model_name="vision", model_version="1", parent_artifact_ids=(frame.artifact_id,),
    ))
    crosscheck = _sealed(replace(
        crosscheck,
        parent_artifact_ids=(
            frame.artifact_id, semantic.artifact_id, vision.artifact_id,
            *((ocr.artifact_id,) if crosscheck_parent_ocr else ()),
            *((transcript.artifact_id,) if crosscheck_parent_transcript else ()),
        ),
        artifact_id="pending", content_hash="",
    ))
    def entry(relation):
        return {
            "frame_id": frame.frame_id, "frame_artifact_id": frame.artifact_id,
            "frame_artifact_hash": f"sha256:{frame.content_hash}", "timestamp_ms": 1200,
            "image_hash": digest, "relation": relation,
            "ocr": [{"artifact_id": ocr.artifact_id, "artifact_hash": f"sha256:{ocr.content_hash}",
                     "summary": ocr_summary, "confidence": 0.93,
                     "model": {"name": "paddleocr", "version": "3.7", "confidence": 0.93}}],
            "vision": [{"artifact_id": vision.artifact_id, "artifact_hash": f"sha256:{vision.content_hash}",
                        "summary": "company list", "confidence": 0.81,
                        "model": {"name": "vision", "version": "1", "confidence": 0.81}}],
        }
    first_window = {
        "evidence_window_id": "window-1", "frames": [entry("UNKNOWN" if gap_mode else "SUPPORTS_DISPLAYED_MENTION")],
    }
    if gap_mode:
        first_window.update({
            "status": "GAP", "reason": "CROSSCHECK_SCOPE_AMBIGUOUS" if gap_mode == "ambiguous"
            else "CROSSCHECK_MISSING",
        })
    first_packet = {
        "knowledge_id": "knowledge-1", "occurrence_id": "knowledge-1",
        "status": "HUMAN_REVIEW_REQUIRED" if gap_mode == "mixed_missing" else "GAP" if gap_mode else "AVAILABLE",
        "windows": [first_window],
    }
    if gap_mode == "mixed_missing":
        first_packet["windows"].append({
            "evidence_window_id": "window-2", "status": "HUMAN_REVIEW_REQUIRED",
            "reason": "CROSSCHECK_CONTRADICTS", "frames": [entry("CONTRADICTS")],
        })
        packets = [first_packet]
    else:
        packets = [first_packet, {
            "knowledge_id": "knowledge-2", "occurrence_id": "knowledge-2", "windows": [
                {"evidence_window_id": "window-2", "frames": [entry("UNRELATED")]}
            ],
        }]
    packet = _sealed(KnowledgeVisualEvidenceArtifact(
        artifact_id="pending", artifact_type="knowledge_visual_evidence",
        parent_artifact_ids=(frame.artifact_id, crosscheck.artifact_id, ocr.artifact_id, vision.artifact_id),
        occurrence_packets=packets,
    ))
    occurrences = _Occurrences(*(
        SimpleNamespace(occurrence_id=item["occurrence_id"], semantic_segment_id=semantic_id,
                        source_artifact_id=source.artifact_id, transcript_artifact_id=transcript.artifact_id,
                        provenance={"visual_evidence": item})
        for item in packets
    ))
    occurrence_artifact = _sealed(ClaimOccurrenceArtifact(
        artifact_id="pending", artifact_type="occurrences",
        semantic_segment_artifact_id=semantic.artifact_id,
        occurrence_ids=[item["occurrence_id"] for item in packets],
        parent_artifact_ids=(semantic.artifact_id, crosscheck.artifact_id, packet.artifact_id),
    ))
    artifacts = _Artifacts(
        source, media, transcript, frame, semantic, crosscheck, ocr, vision, packet, occurrence_artifact
    )
    snapshots = SnapshotService()
    snapshot = snapshots.record_from_artifacts(
        source_type="XIAOE", source_ref="lesson-1", source_content_hash="a" * 64,
        artifact_ids={"source": source.artifact_id, "media": media.artifact_id, "frames:0": frame.artifact_id,
                      "transcript": transcript.artifact_id,
                      "knowledge_visual_evidence": packet.artifact_id,
                      "semantic_segments": semantic.artifact_id,
                      "transcript_visual_crosscheck": crosscheck.artifact_id,
                      "ocr:0": ocr.artifact_id, "vision:0": vision.artifact_id,
                      "occurrences": occurrence_artifact.artifact_id},
    )
    token = tmp_path / "token"
    token.write_text("frame-secret", encoding="utf-8")
    service = _Service(snapshots, artifacts, occurrences)
    client = TestClient(create_app(service, authorizer=ServiceAuthorizer((token,), ("stock_agent",))))
    url = f"/api/v1/content-snapshots/{snapshot.content_snapshot_id}/frames/{frame.frame_id}"
    headers = {"Authorization": "Bearer frame-secret", "X-Caller-Service": "stock_agent"}
    return client, url, headers, service, frame, packet, path, image


def test_frame_asset_private_scoped_metadata_and_binary(tmp_path, monkeypatch):
    client, url, headers, _service, frame, _packet, path, image = _fixture(tmp_path, monkeypatch)
    assert client.get(url).status_code == 401
    wrong = {"Authorization": "Bearer wrong", "X-Caller-Service": "stock_agent"}
    assert client.get(url, headers=wrong).status_code == 401
    assert client.get(url, headers={**headers, "X-Caller-Service": "unknown"}).status_code == 403
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    data = response.json()["data"]
    assert response.json()["contract_version"] == "content.frame-asset.v1"
    assert data["frame_id"] == frame.frame_id
    assert data["timestamp_ms"] == 1200
    assert data["image_hash"] == frame.image_hash
    assert [(s["knowledge_id"], s["evidence_window_id"], s["relation"]) for s in data["scopes"]] == [
        ("knowledge-1", "window-1", "SUPPORTS_DISPLAYED_MENTION"),
        ("knowledge-2", "window-2", "UNRELATED"),
    ]
    assert data["scopes"][0]["ocr"] == [{"summary": "蓝晓科技", "confidence": 0.93}]
    assert str(path) not in response.text and "storage_ref" not in response.text
    binary = client.get(url + "/image", headers=headers)
    assert binary.status_code == 200 and binary.content == image
    assert binary.headers["content-type"] == "image/png"
    assert binary.headers["cache-control"] == "no-store"
    assert binary.headers["x-content-type-options"] == "nosniff"


def test_frame_asset_cross_snapshot_missing_and_tamper(tmp_path, monkeypatch):
    client, url, headers, service, frame, packet, path, _image = _fixture(tmp_path, monkeypatch)
    assert client.get(url.replace(frame.frame_id, "frame-other"), headers=headers).status_code == 404
    assert client.get(url.replace("/frames/", "-other/frames/"), headers=headers).status_code == 404
    path.write_bytes(b"\x89PNG\r\n\x1a\nchanged")
    response = client.get(url + "/image", headers=headers)
    assert response.status_code == 409
    assert str(path) not in response.text and "storage_ref" not in response.text
    path.unlink()
    assert client.get(url, headers=headers).status_code == 409
    service._artifact_repository.items.pop(packet.artifact_id)
    assert client.get(url, headers=headers).status_code == 409


def test_frame_asset_rejects_outside_and_symlink(tmp_path, monkeypatch):
    client, url, headers, service, frame, _packet, path, image = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / path.name
    outside.write_bytes(image)
    service._artifact_repository.items[frame.artifact_id] = replace(frame, storage_ref=str(outside))
    assert client.get(url, headers=headers).status_code == 409
    service._artifact_repository.items[frame.artifact_id] = replace(frame, storage_ref=str(path))
    path.unlink()
    try:
        path.symlink_to(outside)
    except (OSError, NotImplementedError):
        return  # Windows symlink privilege is an environment gate.
    assert client.get(url + "/image", headers=headers).status_code == 409


def test_frame_asset_rejects_snapshot_and_packet_drift(tmp_path, monkeypatch):
    client, url, headers, service, frame, packet, _path, _image = _fixture(tmp_path, monkeypatch)
    snapshot_id = url.split("/")[4]
    store = service._snapshots._store
    original = store._snapshots[snapshot_id]
    store._snapshots[snapshot_id] = replace(original, artifact_root_hash="0" * 64)
    assert client.get(url, headers=headers).status_code == 409
    store._snapshots[snapshot_id] = replace(original, producer_manifest={"code_sha": "tampered"})
    assert client.get(url, headers=headers).status_code == 409
    store._snapshots[snapshot_id] = original
    service._artifact_repository.items[packet.artifact_id] = replace(packet, parent_artifact_ids=())
    assert client.get(url, headers=headers).status_code == 409
    service._artifact_repository.items[packet.artifact_id] = packet
    service._artifact_repository.items[frame.artifact_id] = replace(frame, image_hash="0" * 64)
    assert client.get(url + "/image", headers=headers).status_code == 409


def test_frame_asset_requires_independent_scoped_lineage(tmp_path, monkeypatch):
    client, url, headers, service, _frame, packet, _path, _image = _fixture(tmp_path, monkeypatch)
    snapshot = service._snapshots.get(url.split("/")[4])
    for slot in ("occurrences", "semantic_segments", "transcript_visual_crosscheck", "ocr:0", "vision:0"):
        artifact_id = snapshot.artifact_ids[slot]
        removed = service._artifact_repository.items.pop(artifact_id)
        assert client.get(url, headers=headers).status_code == 409, slot
        service._artifact_repository.items[artifact_id] = removed
    removed = service._occurrence_repository.items.pop("knowledge-1")
    assert client.get(url, headers=headers).status_code == 409
    service._occurrence_repository.items["knowledge-1"] = removed
    service._occurrence_repository.items["knowledge-1"] = SimpleNamespace(
        occurrence_id="knowledge-1", semantic_segment_id="other-semantic",
        source_artifact_id=service._snapshots.get(url.split("/")[4]).source_artifact_id,
        transcript_artifact_id=service._snapshots.get(url.split("/")[4]).artifact_ids["transcript"],
        provenance={"visual_evidence": packet.occurrence_packets[0]},
    )
    assert client.get(url, headers=headers).status_code == 409


def test_frame_asset_rejects_scoped_relation_and_modality_drift(tmp_path, monkeypatch):
    client, url, headers, _service, _frame, _packet, _path, _image = _fixture(
        tmp_path, monkeypatch, crosscheck_relation="UNKNOWN"
    )
    assert client.get(url, headers=headers).status_code == 409
    client, url, headers, _service, _frame, _packet, _path, _image = _fixture(
        tmp_path / "third", monkeypatch, crosscheck_parent_ocr=False
    )
    assert client.get(url, headers=headers).status_code == 409
    client, url, headers, _service, _frame, _packet, _path, _image = _fixture(
        tmp_path / "second", monkeypatch, ocr_summary="different OCR"
    )
    assert client.get(url, headers=headers).status_code == 409


def test_frame_asset_corrupt_snapshot_store_is_redacted(tmp_path, monkeypatch):
    client, url, headers, service, _frame, _packet, path, _image = _fixture(tmp_path, monkeypatch)
    def corrupt(_snapshot_id):
        raise SnapshotIntegrityError(f"corrupt row at {path}")
    monkeypatch.setattr(service._snapshots, "get", corrupt)
    response = client.get(url, headers=headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "FRAME_ASSET_INTEGRITY_ERROR"
    assert str(path) not in response.text


def test_frame_asset_bounded_read(tmp_path, monkeypatch):
    client, url, headers, _service, _frame, _packet, _path, _image = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("stock_content.application.frame_assets._MAX_IMAGE_BYTES", 8)
    response = client.get(url + "/image", headers=headers)
    assert response.status_code == 409
    assert response.headers["cache-control"] == "no-store"


def test_frame_asset_preserves_sealed_unknown_gap_without_support(tmp_path, monkeypatch):
    for mode in ("missing", "ambiguous", "mixed_missing"):
        client, url, headers, _service, _frame, _packet, _path, _image = _fixture(
            tmp_path / mode, monkeypatch, gap_mode=mode
        )
        response = client.get(url, headers=headers)
        assert response.status_code == 200, mode
        scopes = response.json()["data"]["scopes"]
        assert scopes[0]["relation"] == "UNKNOWN"
        assert not any(scope["relation"].startswith("SUPPORTS") for scope in scopes)
        if mode == "mixed_missing":
            assert scopes[1]["relation"] == "CONTRADICTS"


def test_frame_asset_malformed_storage_ref_is_redacted(tmp_path, monkeypatch):
    client, url, headers, service, frame, _packet, _path, _image = _fixture(tmp_path, monkeypatch)
    service._artifact_repository.items[frame.artifact_id] = replace(frame, storage_ref=None)
    response = client.get(url + "/image", headers=headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "FRAME_ASSET_INTEGRITY_ERROR"
    assert "storage_ref" not in response.text


def test_frame_asset_rejects_foreign_frame_inside_ambiguous_gap(tmp_path, monkeypatch):
    client, url, headers, _service, _frame, _packet, _path, _image = _fixture(
        tmp_path, monkeypatch, gap_mode="ambiguous", ambiguous_foreign_timestamp=True
    )
    assert client.get(url, headers=headers).status_code == 409


def test_frame_asset_requires_transcript_snapshot_and_parent_closure(tmp_path, monkeypatch):
    client, url, headers, service, _frame, _packet, _path, _image = _fixture(tmp_path, monkeypatch)
    snapshot = service._snapshots.get(url.split("/")[4])
    transcript_id = snapshot.artifact_ids["transcript"]
    transcript = service._artifact_repository.items.pop(transcript_id)
    assert client.get(url, headers=headers).status_code == 409
    service._artifact_repository.items[transcript_id] = replace(transcript, media_artifact_id="wrong")
    assert client.get(url, headers=headers).status_code == 409
    for kwargs in (
        {"wrong_semantic_transcript": True},
        {"wrong_crosscheck_transcript": True},
        {"crosscheck_parent_transcript": False},
    ):
        client, url, headers, _service, _frame, _packet, _path, _image = _fixture(
            tmp_path / str(len(kwargs)) / str(next(iter(kwargs))), monkeypatch, **kwargs
        )
        assert client.get(url, headers=headers).status_code == 409, kwargs


def test_frame_asset_occurrence_row_must_bind_active_source_and_transcript(tmp_path, monkeypatch):
    client, url, headers, service, _frame, _packet, _path, _image = _fixture(tmp_path, monkeypatch)
    row = service._occurrence_repository.items["knowledge-1"]
    for field in ("source_artifact_id", "transcript_artifact_id"):
        for value in (None, "foreign-artifact"):
            changed = dict(vars(row))
            changed[field] = value
            service._occurrence_repository.items["knowledge-1"] = SimpleNamespace(**changed)
            response = client.get(url, headers=headers)
            assert response.status_code == 409, (field, value)
            assert response.json()["error"]["code"] == "FRAME_ASSET_INTEGRITY_ERROR"
    service._occurrence_repository.items["knowledge-1"] = row
    assert client.get(url, headers=headers).status_code == 200

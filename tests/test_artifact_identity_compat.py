"""Regression coverage for pre-EPIC-043 visual artifact identities."""
from __future__ import annotations

from copy import deepcopy

import pytest

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import ContentArtifactRow
from stock_content.adapters.postgres.repositories.artifact_repository import (
    ArtifactIntegrityError,
    SqlArtifactRepository,
)
from stock_content.domain.artifacts import (
    FrameArtifact,
    OCRArtifact,
    artifact_id_of,
    artifact_identity_payload,
    canonical_json,
    deserialize_artifact,
    serialize_artifact,
)

# These are byte-for-byte representative pre-EPIC-043 artifact.v1 JSON
# payloads.  Their hashes were produced by base 48a6ff8, before targeted-frame
# planner and GPU runtime provenance became identity inputs.
_LEGACY_FRAME_PAYLOAD = {
    "artifact_id": "frame-0fbb262edc8c5cf4beaf9673cf812fe6",
    "artifact_type": "frame",
    "schema_version": "artifact.v1",
    "created_at": "2026-09-08 00:00:00+00:00",
    "producer_stage": "frame",
    "producer_version": "1.0.0",
    "parent_artifact_ids": ["media-legacy"],
    "content_hash": "0fbb262edc8c5cf4beaf9673cf812fe66b1ed62b485d9aebb79132e418571b5f",
    "media_artifact_id": "media-legacy",
    "frame_id": "frame-legacy",
    "timestamp_ms": 1234,
    "image_hash": "a" * 64,
    "storage_ref": "private://frames/frame-legacy.png",
    "extraction_reason": "legacy-interval",
    "semantic_segment_ids": ["segment-1"],
    "evidence_window_ids": ["window-1"],
    "planner_version": "targeted.v1",
}

_LEGACY_OCR_PAYLOAD = {
    "artifact_id": "ocr-9eb3c5bb9a4b4aceed9087e205d63533",
    "artifact_type": "ocr",
    "schema_version": "artifact.v1",
    "created_at": "2026-09-08 00:00:00+00:00",
    "producer_stage": "ocr",
    "producer_version": "1.0.0",
    "parent_artifact_ids": ["frame-legacy"],
    "content_hash": "9eb3c5bb9a4b4aceed9087e205d63533b0c291099d287ecd99025b8f774832c0",
    "frame_artifact_id": "frame-legacy-artifact",
    "frame_id": "frame-legacy",
    "timestamp_ms": 1234,
    "image_hash": "b" * 64,
    "semantic_segment_ids": ["segment-1"],
    "evidence_window_ids": ["window-1"],
    "text": "旧版识别文本",
    "bbox": [[1, 2, 3, 4]],
    "confidence_score": 0.9,
    "blocks": [{"text": "旧版"}],
    "engine": "paddleocr",
    "engine_version": "2.7",
}


@pytest.mark.parametrize("payload", [_LEGACY_FRAME_PAYLOAD, _LEGACY_OCR_PAYLOAD])
def test_pre_epic043_visual_payload_rehydrates_with_its_exact_original_identity(payload):
    artifact = deserialize_artifact(deepcopy(payload))

    expected_identity = {
        key: value for key, value in payload.items() if key not in {"artifact_id", "created_at", "content_hash"}
    }
    assert artifact_identity_payload(artifact) == expected_identity
    assert artifact.content_hash == payload["content_hash"]
    assert artifact_id_of(artifact) == payload["artifact_id"]
    # A replay must not silently turn a historical row into a current one.
    assert canonical_json(serialize_artifact(artifact)) == canonical_json(payload)


def test_new_visual_provenance_fields_are_identity_bound():
    frame = FrameArtifact(
        artifact_id="frame-pending",
        artifact_type="frame",
        media_artifact_id="media-current",
        frame_id="frame-current",
        timestamp_ms=1000,
        image_hash="c" * 64,
        storage_ref="private://frames/current.png",
        planner_request_id="plan-request-1",
    )
    changed_frame = FrameArtifact(**{**frame.__dict__, "planner_request_id": "plan-request-2", "content_hash": ""})
    assert frame.content_hash != changed_frame.content_hash

    ocr = OCRArtifact(
        artifact_id="ocr-pending",
        artifact_type="ocr",
        frame_artifact_id="frame-current-artifact",
        frame_id="frame-current",
        image_hash="c" * 64,
        text="营收增长",
        requested_device="gpu:0",
        actual_device="gpu:0",
        runtime_identity={"paddle": "3.3.0", "cuda": "12.9"},
    )
    changed_ocr = OCRArtifact(**{**ocr.__dict__, "actual_device": "cpu", "content_hash": ""})
    assert ocr.content_hash != changed_ocr.content_hash


def test_repository_accepts_legacy_payload_but_rejects_new_identity_mutation(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'artifact-identity.db'}")
    database.create_schema()
    repository = SqlArtifactRepository(database.session_factory)

    legacy = deserialize_artifact(deepcopy(_LEGACY_FRAME_PAYLOAD))
    repository.put(legacy)
    assert repository.verify(legacy.artifact_id)

    raw = FrameArtifact(
        artifact_id="frame-pending",
        artifact_type="frame",
        media_artifact_id="media-current",
        frame_id="frame-current",
        timestamp_ms=1000,
        image_hash="c" * 64,
        planner_request_id="plan-request-1",
    )
    current = FrameArtifact(**{**raw.__dict__, "artifact_id": artifact_id_of(raw)})
    repository.put(current)
    with repository._sessions.begin() as session:  # noqa: SLF001 - deliberate persisted-payload attack
        row = session.get(ContentArtifactRow, current.artifact_id)
        row.payload = {**row.payload, "planner_request_id": "plan-request-tampered"}
    with pytest.raises(ArtifactIntegrityError):
        repository.verify(current.artifact_id)

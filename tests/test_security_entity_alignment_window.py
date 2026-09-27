"""Deterministic window-local entity recall; never claim support."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from stock_content.application.pipeline import PipelineContext
from stock_content.application.replay.errors import ReplayIntegrityError
from stock_content.application.replay.integrity import ReplayIntegrityMixin
from stock_content.application.stages import SecurityEntityAlignmentStage
from stock_content.domain.artifacts import (
    FrameArtifact,
    OCRArtifact,
    TranscriptArtifact,
    TranscriptSegmentItem,
    TranscriptVisualCrosscheckArtifact,
    TranscriptVisualCrosscheckRecord,
    VisionArtifact,
    deserialize_artifact,
)
from stock_content.domain.knowledge_evidence_window import KnowledgeEvidenceWindow
from stock_content.domain.knowledge_frame_plan import evidence_window_id


def _context(*, ocr_text: str = "蓝晓科技 300487", asr_text: str = "蓝小科技"):
    context = PipelineContext(task_id="entity-window", source={})
    transcript = TranscriptArtifact(
        artifact_id="transcript-a", artifact_type="transcript", media_artifact_id="media-a",
        asr_model="fixture", asr_model_version="1",
        segments=[TranscriptSegmentItem(
            segment_index=0, start_seconds=8, end_seconds=10, text=asr_text,
            raw_text=asr_text, normalized_text=asr_text, source_artifact_id="asr-a",
        )],
    )
    window = KnowledgeEvidenceWindow(
        semantic_segment_id="semantic-a", start_ms=8_000, end_ms=10_000, center_ms=9_000,
        transcript_segment_ids=(transcript.segments[0].segment_id,), high_signals=(),
    )
    window_id = evidence_window_id(window)
    frame = FrameArtifact(
        artifact_id="frame-a", artifact_type="frame", frame_id="f-a", timestamp_ms=9_000,
        image_hash="image-a", semantic_segment_ids=("semantic-a",), evidence_window_ids=(window_id,),
    )
    ocr = OCRArtifact(
        artifact_id="ocr-a", artifact_type="ocr", frame_artifact_id=frame.artifact_id,
        frame_id=frame.frame_id, timestamp_ms=frame.timestamp_ms, image_hash=frame.image_hash,
        evidence_window_ids=(window_id,), text=ocr_text, bbox=[[1, 2], [3, 4]],
        confidence_score=0.99, engine="paddle", engine_version="3",
    )
    # UNKNOWN is not claim support, but it independently seals which ASR
    # segment belongs to this frame/window for audit-only alignment.
    crosscheck = TranscriptVisualCrosscheckArtifact(
        artifact_id="crosscheck-a", artifact_type="transcript_visual_crosscheck",
        transcript_artifact_id=transcript.artifact_id,
        relations=(TranscriptVisualCrosscheckRecord(
            frame_id=frame.frame_id, frame_artifact_id=frame.artifact_id,
            timestamp_ms=frame.timestamp_ms, semantic_segment_ids=("semantic-a",),
            evidence_window_ids=(window_id,), transcript_segment_ids=(transcript.segments[0].segment_id,),
            relation="UNKNOWN",
        ),),
    )
    context.artifacts.transcript = transcript
    context.artifacts.frames = [frame]
    context.artifacts.ocr = [ocr]
    context.artifacts.transcript_visual_crosscheck = crosscheck
    context.state.knowledge_evidence_windows = [window]
    return context, window_id


def test_pending_correction_is_window_scoped_and_does_not_infer_ocr_ticker():
    context, window_id = _context()
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    aligned = next(item for item in artifact.security_mentions if item.record_type == "ALIGNED_MENTION")
    assert aligned.evidence_window_id == window_id
    assert aligned.frame_id == "f-a" and aligned.ocr_artifact_id == "ocr-a"
    assert aligned.relation == "ENTITY_CORRECTION_PENDING"
    assert aligned.ticker == "" and aligned.asr_ticker == ""
    assert aligned.canonical_display == "蓝小科技"
    assert aligned.review_status == "HUMAN_REVIEW_REQUIRED"
    assert aligned.authorization_status == "NOT_AUTHORIZED"
    assert context.artifacts.transcript_visual_crosscheck.eligible_frame_ids == ()
    assert artifact.displayed_target_candidates[0].ticker == "300487"  # raw OCR observation only
    restored = deserialize_artifact(artifact.to_dict())
    assert restored.content_hash == artifact.content_hash


def test_exact_security_display_is_a_mention_only_and_not_claim_support():
    context, _ = _context(asr_text="蓝晓科技")
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    aligned = next(item for item in artifact.security_mentions if item.record_type == "ALIGNED_MENTION")
    assert aligned.relation == "SUPPORTS_DISPLAYED_MENTION"
    assert aligned.ticker == "300487" and aligned.relation_strength == "HIGH"
    assert context.artifacts.transcript_visual_crosscheck.eligible_frame_ids == ()


def test_replay_rejects_cross_window_entity_rebinding():
    context, window_id = _context()
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    mapping = {
        "transcript": context.artifacts.transcript.artifact_id,
        "transcript_visual_crosscheck": context.artifacts.transcript_visual_crosscheck.artifact_id,
        "security_entity_alignment": artifact.artifact_id,
    }
    loaded = {item.artifact_id: item for item in context.artifacts.artifacts()}
    ReplayIntegrityMixin._validate_security_entity_alignment(mapping, loaded)
    bad = replace(artifact, displayed_target_candidates=tuple(
        replace(item, evidence_window_id="other-window") for item in artifact.displayed_target_candidates
    ))
    with pytest.raises(ReplayIntegrityError, match="window/frame/OCR"):
        ReplayIntegrityMixin._validate_security_entity_alignment(mapping, {**loaded, artifact.artifact_id: bad})
    assert window_id != "other-window"


def test_replay_rejects_aligned_mention_swapped_to_other_existing_asr_segment():
    context, _ = _context()
    original = context.artifacts.transcript
    other = TranscriptSegmentItem(
        segment_index=1, start_seconds=20, end_seconds=22, text="华发科技",
        raw_text="华发科技", normalized_text="华发科技", source_artifact_id="asr-b",
        media_artifact_id=original.media_artifact_id,
        asr_model=original.asr_model, asr_model_version=original.asr_model_version,
    )
    context.artifacts.transcript = replace(original, segments=[*original.segments, other])
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    mapping = {
        "transcript": context.artifacts.transcript.artifact_id,
        "transcript_visual_crosscheck": context.artifacts.transcript_visual_crosscheck.artifact_id,
        "security_entity_alignment": artifact.artifact_id,
    }
    loaded = {item.artifact_id: item for item in context.artifacts.artifacts()}
    ReplayIntegrityMixin._validate_security_entity_alignment(mapping, loaded)
    bad = replace(artifact, security_mentions=tuple(
        replace(item, asr_segment_id=other.segment_id, source_artifact_id="asr-b")
        if item.record_type == "ALIGNED_MENTION" else item
        for item in artifact.security_mentions
    ))
    with pytest.raises(ReplayIntegrityError, match="outside sealed window"):
        ReplayIntegrityMixin._validate_security_entity_alignment(mapping, {**loaded, artifact.artifact_id: bad})


def test_ambiguous_ocr_names_keep_ticker_as_unpaired_observation():
    context, window_id = _context(ocr_text="蓝晓科技 华发科技 300487", asr_text="蓝晓科技")
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    candidates = artifact.displayed_target_candidates
    assert {(item.ocr_name, item.ticker) for item in candidates} == {
        ("蓝晓科技", ""), ("华发科技", ""), ("", "300487"),
    }
    assert all(item.evidence_window_id == window_id for item in candidates)
    assert all(item.relation == "ENTITY_CORRECTION_PENDING" for item in candidates)
    code = next(item for item in candidates if item.ticker)
    assert code.correction_trace[0]["type"] == "UNPAIRED_DISPLAYED_CODE"
    assert not any(item.ticker == "300487" for item in artifact.security_mentions
                   if item.record_type == "ALIGNED_MENTION")


def test_another_window_never_receives_first_window_asr_name():
    context, first_window_id = _context()
    other = KnowledgeEvidenceWindow(
        semantic_segment_id="semantic-b", start_ms=20_000, end_ms=22_000, center_ms=21_000,
        transcript_segment_ids=(), high_signals=(),
    )
    other_window_id = evidence_window_id(other)
    context.state.knowledge_evidence_windows.append(other)
    context.artifacts.frames.append(FrameArtifact(
        artifact_id="frame-b", artifact_type="frame", frame_id="f-b", timestamp_ms=21_000,
        image_hash="image-b", semantic_segment_ids=("semantic-b",), evidence_window_ids=(other_window_id,),
    ))
    context.artifacts.ocr.append(OCRArtifact(
        artifact_id="ocr-b", artifact_type="ocr", frame_artifact_id="frame-b",
        frame_id="f-b", timestamp_ms=21_000, image_hash="image-b",
        evidence_window_ids=(other_window_id,), text="蓝晓科技 300487", confidence_score=0.99,
    ))
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    assert {item.evidence_window_id for item in artifact.displayed_target_candidates} == {
        first_window_id, other_window_id,
    }
    aligned_windows = {item.evidence_window_id for item in artifact.security_mentions
                       if item.record_type == "ALIGNED_MENTION"}
    assert aligned_windows == {
        first_window_id,
    }


@pytest.mark.parametrize("reason,relations", [
    ("CROSSCHECK_MISSING", ()),
    ("CROSSCHECK_SCOPE_AMBIGUOUS", (TranscriptVisualCrosscheckRecord(
        frame_id="f-a", frame_artifact_id="frame-a", timestamp_ms=9_000,
        semantic_segment_ids=("wrong-semantic",), evidence_window_ids=("window-a",),
        relation="UNRELATED",
    ),)),
])
def test_replay_accepts_only_unknown_gap_when_scoped_crosscheck_is_absent(reason, relations):
    frame = FrameArtifact(
        artifact_id="frame-a", artifact_type="frame", frame_id="f-a", timestamp_ms=9_000,
        image_hash="image-a", evidence_window_ids=("window-a",),
    )
    crosscheck = TranscriptVisualCrosscheckArtifact(
        artifact_id="crosscheck-a", artifact_type="transcript_visual_crosscheck", relations=relations,
    )
    packet = {"status": "GAP"}
    entry = {
        "frame_id": frame.frame_id, "frame_artifact_id": frame.artifact_id,
        "frame_artifact_hash": f"sha256:{frame.content_hash}", "timestamp_ms": frame.timestamp_ms,
        "image_hash": frame.image_hash, "relation": "UNKNOWN", "ocr": [], "vision": [],
    }
    kwargs = dict(
        packet=packet, evidence_window_id="window-a", semantic_segment_id="semantic-a",
        crosscheck_artifact=crosscheck, visual_parent_ids={frame.artifact_id},
        loaded={frame.artifact_id: frame}, artifact_id="visual-packet-a",
        window_status="GAP", window_reason=reason,
    )
    ReplayIntegrityMixin._validate_visual_packet_frame(frame=entry, **kwargs)
    with pytest.raises(ReplayIntegrityError, match="relation is not uniquely sealed"):
        ReplayIntegrityMixin._validate_visual_packet_frame(frame={**entry, "relation": "SUPPORTS"}, **kwargs)
    with pytest.raises(ReplayIntegrityError, match="relation is not uniquely sealed"):
        ReplayIntegrityMixin._validate_visual_packet_frame(
            frame=entry,
            **{**kwargs, "packet": {"status": "HUMAN_REVIEW_REQUIRED", "windows": [{"status": "GAP"}]}},
        )
    with pytest.raises(ReplayIntegrityError, match="relation is not uniquely sealed"):
        ReplayIntegrityMixin._validate_visual_packet_frame(
            frame=entry, **{**kwargs, "packet": {"status": "AVAILABLE"}}
        )


def test_replay_accepts_missing_crosscheck_gap_inside_mixed_human_review_packet():
    frame = FrameArtifact(
        artifact_id="mixed-frame", artifact_type="frame", frame_id="mixed-f", timestamp_ms=9_000,
        image_hash="mixed-image", evidence_window_ids=("gap-window",),
    )
    conflict_frame = FrameArtifact(
        artifact_id="conflict-frame", artifact_type="frame", frame_id="conflict-f", timestamp_ms=12_000,
        image_hash="conflict-image", evidence_window_ids=("conflict-window",),
    )
    crosscheck = TranscriptVisualCrosscheckArtifact(
        artifact_id="mixed-crosscheck", artifact_type="transcript_visual_crosscheck",
        relations=(TranscriptVisualCrosscheckRecord(
            frame_id=conflict_frame.frame_id, frame_artifact_id=conflict_frame.artifact_id,
            timestamp_ms=conflict_frame.timestamp_ms, semantic_segment_ids=("semantic-a",),
            evidence_window_ids=("conflict-window",), relation="CONTRADICTS",
        ),),
        eligible_frame_ids=(conflict_frame.frame_id,),
    )
    packet = {
        "status": "HUMAN_REVIEW_REQUIRED",
        "windows": [
            {"evidence_window_id": "gap-window", "status": "GAP", "reason": "CROSSCHECK_MISSING"},
            {"evidence_window_id": "conflict-window", "status": "HUMAN_REVIEW_REQUIRED",
             "reason": "CROSSCHECK_CONTRADICTS"},
        ],
    }
    entry = {
        "frame_id": frame.frame_id, "frame_artifact_id": frame.artifact_id,
        "frame_artifact_hash": f"sha256:{frame.content_hash}", "timestamp_ms": frame.timestamp_ms,
        "image_hash": frame.image_hash, "relation": "UNKNOWN", "ocr": [], "vision": [],
    }
    conflict_entry = {
        "frame_id": conflict_frame.frame_id, "frame_artifact_id": conflict_frame.artifact_id,
        "frame_artifact_hash": f"sha256:{conflict_frame.content_hash}",
        "timestamp_ms": conflict_frame.timestamp_ms, "image_hash": conflict_frame.image_hash,
        "relation": "CONTRADICTS", "ocr": [], "vision": [],
    }
    kwargs = dict(
        packet=packet, evidence_window_id="gap-window", semantic_segment_id="semantic-a",
        crosscheck_artifact=crosscheck,
        visual_parent_ids={frame.artifact_id, conflict_frame.artifact_id},
        loaded={frame.artifact_id: frame, conflict_frame.artifact_id: conflict_frame},
        artifact_id="mixed-packet", window_status="GAP", window_reason="CROSSCHECK_MISSING",
    )
    ReplayIntegrityMixin._validate_visual_packet_frame(frame=entry, **kwargs)
    ReplayIntegrityMixin._validate_visual_packet_frame(
        frame=conflict_entry,
        **{**kwargs, "evidence_window_id": "conflict-window", "window_status": "HUMAN_REVIEW_REQUIRED",
           "window_reason": "CROSSCHECK_CONTRADICTS"},
    )
    with pytest.raises(ReplayIntegrityError, match="relation is not uniquely sealed"):
        ReplayIntegrityMixin._validate_visual_packet_frame(frame={**entry, "relation": "SUPPORTS"}, **kwargs)


def test_sealed_sep19_asr_excerpt_preserves_ambiguous_name_without_guessing_security_code():
    # Copied verbatim from sealed transcript segment 962, not from any old
    # conclusion or inferred ticker. This portable fixture needs no media.
    fixture = json.loads((Path(__file__).parent / "fixtures" / "sep19_security_excerpt.json").read_text(
        encoding="utf-8"
    ))
    assert fixture["source_transcript_sha256"] == (
        "107abe600dd947ffc1391d629ce777f2d28444e2c1d6d8c70a023468ce4a7cc5"
    )
    context = PipelineContext(task_id="sealed-asr-excerpt", source={})
    transcript = TranscriptArtifact(
        artifact_id="sealed-transcript-excerpt", artifact_type="transcript", media_artifact_id="sealed-media",
        asr_model="faster-whisper", asr_model_version="small",
        segments=[TranscriptSegmentItem(
            segment_index=fixture["segment_index"], start_seconds=fixture["start_seconds"],
            end_seconds=fixture["end_seconds"], text=fixture["normalized_text"],
            raw_text=fixture["raw_text"], normalized_text=fixture["normalized_text"],
            correction_records=fixture["correction_records"],
        )],
    )
    context.artifacts.transcript = transcript
    context.artifacts.transcript_visual_crosscheck = TranscriptVisualCrosscheckArtifact(
        artifact_id="sealed-crosscheck-excerpt", artifact_type="transcript_visual_crosscheck",
        transcript_artifact_id=transcript.artifact_id,
    )
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    assert any(item.asr_normalized_name == "蓝小科技" and item.ticker == ""
               and item.relation == "ENTITY_CORRECTION_PENDING" for item in artifact.security_mentions)
    assert artifact.displayed_target_candidates == ()


def test_vision_only_observed_entity_and_ticker_are_separate_pending_window_observations():
    context, window_id = _context()
    context.artifacts.ocr = []
    frame = context.artifacts.frames[0]
    context.artifacts.vision = [VisionArtifact(
        artifact_id="vision-a", artifact_type="vision", frame_artifact_id=frame.artifact_id,
        frame_id=frame.frame_id, timestamp_ms=frame.timestamp_ms, image_hash=frame.image_hash,
        evidence_window_ids=(window_id,), payload={
            "observed_entities": ["蓝晓科技"], "observed_tickers": ["300487"],
        },
    )]
    artifact = SecurityEntityAlignmentStage().execute(context).produced_artifacts[0]
    displayed = artifact.displayed_target_candidates
    assert {(item.ocr_name, item.ticker) for item in displayed} == {("蓝晓科技", ""), ("", "300487")}
    assert all(item.relation == "ENTITY_CORRECTION_PENDING" and not item.ocr_artifact_id for item in displayed)
    mapping = {
        "transcript": context.artifacts.transcript.artifact_id,
        "transcript_visual_crosscheck": context.artifacts.transcript_visual_crosscheck.artifact_id,
        "security_entity_alignment": artifact.artifact_id,
    }
    ReplayIntegrityMixin._validate_security_entity_alignment(
        mapping, {item.artifact_id: item for item in context.artifacts.artifacts()}
    )

"""Relational persistence for authoritative semantic segments."""

from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import SemanticSegmentRow
from stock_content.domain.semantic_segment import SemanticSegment, SemanticSegmentArtifact


class SemanticSegmentRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def save(
        self,
        segment: SemanticSegment | SemanticSegmentArtifact,
        *,
        artifact_id: str | None = None,
        video_id: str | None = None,
    ) -> SemanticSegment | SemanticSegmentArtifact:
        if isinstance(segment, SemanticSegmentArtifact):
            _validate_video_id(video_id)
            for item in segment.segments:
                self.save(
                    _item_to_segment(segment, item),
                    artifact_id=segment.artifact_id,
                    video_id=video_id,
                )
            return segment
        return self._save_segment(segment, artifact_id=artifact_id, video_id=video_id)

    def _save_segment(
        self,
        segment: SemanticSegment,
        *,
        artifact_id: str | None = None,
        video_id: str | None = None,
    ) -> SemanticSegment:
        _validate_video_id(video_id)
        if len(segment.semantic_segment_id) > 64:
            raise ValueError("semantic_segment_id exceeds the 64-character contract")
        values = asdict(segment)
        row_values = {**values, "video_id": video_id, "artifact_id": artifact_id}
        with self._sessions.begin() as session:
            row = session.get(SemanticSegmentRow, segment.semantic_segment_id)
            if row is not None:
                if row.video_id != video_id or _row_payload(row) != values:
                    raise ValueError(
                        f"semantic segment id {segment.semantic_segment_id} already stores different payload"
                    )
            else:
                session.add(SemanticSegmentRow(**row_values))
        return segment

    put = save

    def get(self, semantic_segment_id: str) -> SemanticSegment | None:
        with self._sessions() as session:
            row = session.get(SemanticSegmentRow, semantic_segment_id)
            return _to_segment(row) if row is not None else None

    def list_for_transcript(self, transcript_artifact_id: str) -> list[SemanticSegment]:
        with self._sessions() as session:
            rows = session.scalars(
                select(SemanticSegmentRow)
                .where(SemanticSegmentRow.transcript_artifact_id == transcript_artifact_id)
                .order_by(SemanticSegmentRow.segment_index, SemanticSegmentRow.semantic_segment_id)
            ).all()
        return [_to_segment(row) for row in rows]


def _item_to_segment(artifact, item) -> SemanticSegment:
    values = dict(vars(item))
    values["transcript_artifact_id"] = artifact.transcript_artifact_id
    values["model_id"] = values.get("model_id") or artifact.model_id
    values["prompt_version"] = values.get("prompt_version") or artifact.prompt_version
    return SemanticSegment(**values)


def _row_payload(row: SemanticSegmentRow) -> dict:
    return {
        "semantic_segment_id": row.semantic_segment_id,
        "transcript_artifact_id": row.transcript_artifact_id,
        "segment_index": row.segment_index,
        "start_segment_index": row.start_segment_index,
        "end_segment_index": row.end_segment_index,
        "start_segment_id": row.start_segment_id,
        "end_segment_id": row.end_segment_id,
        "start_ms": row.start_ms,
        "end_ms": row.end_ms,
        "topic": row.topic,
        "subject": row.subject,
        "segment_type": row.segment_type,
        "model_id": row.model_id,
        "prompt_version": row.prompt_version,
        "confidence": row.confidence,
    }


def _to_segment(row: SemanticSegmentRow) -> SemanticSegment:
    values = _row_payload(row)
    values.pop("artifact_id", None)
    return SemanticSegment(**values)


def _validate_video_id(video_id: str | None) -> None:
    if not video_id or not video_id.strip():
        raise ValueError("semantic segment persistence requires an authoritative video_id")
    if len(video_id) > 64:
        raise ValueError("video_id exceeds the 64-character contract")


__all__ = ["SemanticSegmentRepository"]

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import OcrEvidenceRow, TemporalWindowRow, VideoFrameRow, VisionEvidenceRow


class PostgresMultimodalRepository:
    """The durable authority for pipeline-produced frame and temporal evidence."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def replace(
        self, video_id: str, frames: list[dict], ocr: list[dict], vision: list[dict], windows: list[dict]
    ) -> None:
        with self._sessions.begin() as session:
            self.replace_in_session(session, video_id, frames, ocr, vision, windows)

    def replace_in_session(
        self, session, video_id: str, frames: list[dict], ocr: list[dict], vision: list[dict], windows: list[dict]
    ) -> None:
            frame_ids = session.scalars(select(VideoFrameRow.frame_id).where(VideoFrameRow.video_id == video_id)).all()
            if frame_ids:
                session.execute(delete(OcrEvidenceRow).where(OcrEvidenceRow.frame_id.in_(frame_ids)))
                session.execute(delete(VisionEvidenceRow).where(VisionEvidenceRow.frame_id.in_(frame_ids)))
            session.execute(delete(VideoFrameRow).where(VideoFrameRow.video_id == video_id))
            session.execute(delete(TemporalWindowRow).where(TemporalWindowRow.video_id == video_id))
            for frame in frames:
                session.add(
                    VideoFrameRow(
                        frame_id=str(frame["frame_id"]),
                        video_id=video_id,
                        timestamp_ms=int(frame.get("timestamp_ms") or 0),
                        image_hash=str(frame.get("image_hash") or ""),
                        extraction_reason=str(frame.get("trigger_source") or "INTERVAL"),
                        storage_ref=str(frame.get("image_path") or frame.get("storage_ref") or ""),
                    )
                )
            # VideoFrameRow and OcrEvidenceRow are linked by a scalar FK but
            # intentionally have no ORM relationship.  Flush the parent rows
            # before adding children so PostgreSQL cannot order two pending
            # INSERTs arbitrarily when a later repository operation autoflushes.
            session.flush()
            for item in ocr:
                frame_id = str(item.get("frame_id") or "")
                if frame_id:
                    session.add(
                        OcrEvidenceRow(
                            frame_id=frame_id,
                            timestamp_ms=int(item.get("timestamp_ms") or 0),
                            text=str(item.get("evidence_text") or item.get("text") or ""),
                            bbox=_bbox_json(item.get("bbox")),
                            confidence=item.get("confidence_score") or item.get("confidence"),
                            ocr_engine=str(item.get("ocr_engine") or "unknown"),
                            engine_version=item.get("ocr_engine_version"),
                        )
                    )
            for item in vision:
                frame_id = str(item.get("frame_id") or "")
                if frame_id:
                    session.add(
                        VisionEvidenceRow(
                            frame_id=frame_id,
                            timestamp_ms=int(item.get("timestamp_ms") or 0),
                            label=str(item.get("label") or "FRAME_ANALYSIS"),
                            payload=dict(item),
                            confidence=item.get("confidence_score"),
                            model_name=item.get("model"),
                            model_version=item.get("model_version"),
                        )
                    )
            for index, window in enumerate(windows):
                key = str(
                    window.get("window_id") or hashlib.sha256(f"{video_id}:{index}:{window}".encode()).hexdigest()[:32]
                )
                session.add(
                    TemporalWindowRow(
                        window_id=key,
                        video_id=video_id,
                        start_ms=int(window.get("start_ms") or 0),
                        end_ms=int(window.get("end_ms") or 0),
                        transcript=str(window.get("transcript_text") or ""),
                        speaker_ids=list(window.get("speaker_ids") or []),
                        frame_ids=[str(value) for value in window.get("frame_ids") or []],
                        ocr_items=list(window.get("ocr_evidence") or []),
                        vision_items=list(window.get("visual_evidence") or []),
                    )
                )

    def list_frames(self, video_id: str) -> list[dict]:
        with self._sessions() as session:
            return [
                {
                    "frame_id": row.frame_id,
                    "timestamp_ms": row.timestamp_ms,
                    "image_hash": row.image_hash,
                    "storage_ref": row.storage_ref,
                }
                for row in session.scalars(
                    select(VideoFrameRow).where(VideoFrameRow.video_id == video_id).order_by(VideoFrameRow.timestamp_ms)
                )
            ]

    def list_ocr(self, video_id: str) -> list[dict]:
        with self._sessions() as session:
            return [
                {
                    "frame_id": row.frame_id,
                    "timestamp_ms": row.timestamp_ms,
                    "text": row.text,
                    "confidence": row.confidence,
                }
                for row in session.scalars(
                    select(OcrEvidenceRow).join(VideoFrameRow).where(VideoFrameRow.video_id == video_id)
                )
            ]
    def list_vision(self, video_id: str) -> list[dict]:
        with self._sessions() as session:
            return [
                {"frame_id": row.frame_id, "timestamp_ms": row.timestamp_ms, "label": row.label, "payload": row.payload}
                for row in session.scalars(
                    select(VisionEvidenceRow).join(VideoFrameRow).where(VideoFrameRow.video_id == video_id)
                )
            ]

    def list_temporal_windows(self, video_id: str) -> list[dict]:
        with self._sessions() as session:
            return [
                {
                    "window_id": row.window_id,
                    "start_ms": row.start_ms,
                    "end_ms": row.end_ms,
                    "transcript": row.transcript,
                }
                for row in session.scalars(
                    select(TemporalWindowRow)
                    .where(TemporalWindowRow.video_id == video_id)
                    .order_by(TemporalWindowRow.start_ms)
                )
            ]


def _bbox_json(value: object) -> dict[str, Any] | list[Any]:
    """Preserve OCR geometry in the JSON shape emitted by its engine.

    PaddleOCR commonly returns a four-point polygon (a nested list), while
    older adapters may use a named mapping.  The former must not be coerced
    through ``dict(...)``: that both rejects ordinary polygons and would lose
    their ordered geometry.  Unknown shapes are represented as an empty
    object, matching the historical missing-bbox projection.
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return {}

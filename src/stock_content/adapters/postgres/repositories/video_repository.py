from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import VideoAssetRow, VideoSegmentRow
from stock_content.domain.models import TranscriptSegment, VideoAsset


class PostgresVideoRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def upsert(self, video: VideoAsset, segments: list[TranscriptSegment]) -> None:
        with self._sessions.begin() as session:
            row = session.get(VideoAssetRow, video.video_id)
            values = vars(video)
            if row is None:
                row = VideoAssetRow(**values)
                session.add(row)
            else:
                for name, value in values.items():
                    setattr(row, name, value)
            session.execute(delete(VideoSegmentRow).where(VideoSegmentRow.video_id == video.video_id))
            session.add_all(VideoSegmentRow(video_id=video.video_id, **vars(segment)) for segment in segments)

    def get(self, video_id: str) -> dict | None:
        with self._sessions() as session:
            row = session.get(VideoAssetRow, video_id)
            if row is None:
                return None
            segments = session.scalars(
                select(VideoSegmentRow)
                .where(VideoSegmentRow.video_id == video_id)
                .order_by(VideoSegmentRow.segment_index)
            ).all()
            return {
                "video_id": row.video_id,
                "source_type": row.source_type,
                "source_ref": row.source_ref,
                "title": row.title,
                "author": row.author,
                "duration_seconds": row.duration_seconds,
                "transcript_text": row.transcript_text,
                "source_hash": row.source_hash,
                "segments": [
                    {
                        "segment_index": item.segment_index,
                        "start_seconds": item.start_seconds,
                        "end_seconds": item.end_seconds,
                        "text": item.text,
                        "confidence": item.confidence,
                    }
                    for item in segments
                ],
            }

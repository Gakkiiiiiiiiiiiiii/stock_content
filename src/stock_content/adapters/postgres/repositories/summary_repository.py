from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import VideoSummaryRow
from stock_content.domain.models import VideoSummary


class PostgresSummaryRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def upsert(self, summary: VideoSummary) -> None:
        with self._sessions.begin() as session:
            row = session.get(VideoSummaryRow, summary.video_id)
            if row is None:
                session.add(VideoSummaryRow(**vars(summary)))
            else:
                row.core_summary = summary.core_summary
                row.markdown = summary.markdown
                row.confidence = summary.confidence

    def get(self, video_id: str) -> dict | None:
        with self._sessions() as session:
            row = session.get(VideoSummaryRow, video_id)
            if row is None:
                return None
            return {
                "video_id": row.video_id,
                "core_summary": row.core_summary,
                "markdown": row.markdown,
                "confidence": row.confidence,
            }

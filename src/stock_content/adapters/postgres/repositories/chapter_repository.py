from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import VideoChapterRow
from stock_content.domain.models import VideoChapter


class PostgresChapterRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def replace_for_video(self, video_id: str, chapters: list[VideoChapter]) -> None:
        with self._sessions.begin() as session:
            session.execute(delete(VideoChapterRow).where(VideoChapterRow.video_id == video_id))
            session.add_all(VideoChapterRow(video_id=video_id, **vars(chapter)) for chapter in chapters)

    def list_for_video(self, video_id: str) -> list[dict]:
        with self._sessions() as session:
            rows = session.scalars(
                select(VideoChapterRow)
                .where(VideoChapterRow.video_id == video_id)
                .order_by(VideoChapterRow.chapter_index)
            ).all()
            return [
                {
                    "chapter_id": row.chapter_id,
                    "chapter_index": row.chapter_index,
                    "title": row.title,
                    "start_seconds": row.start_seconds,
                    "end_seconds": row.end_seconds,
                    "summary": row.summary,
                }
                for row in rows
            ]

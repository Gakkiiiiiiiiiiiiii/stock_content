from __future__ import annotations

from sqlalchemy import delete
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

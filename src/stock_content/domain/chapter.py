from __future__ import annotations

import re
from uuid import uuid4

from stock_content.domain.models import TranscriptSegment, VideoChapter


class ChapterSegmenter:
    """Deterministic chapter segmentation suitable for independent regression tests."""

    def __init__(self, target_seconds: float = 300.0) -> None:
        self.target_seconds = target_seconds

    @staticmethod
    def _title(text: str, index: int) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip("，。,. ")
        return cleaned[:28] or f"章节 {index + 1}"

    def segment(self, segments: list[TranscriptSegment]) -> list[VideoChapter]:
        if not segments:
            return []
        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        group_start = segments[0].start_seconds
        for item in segments:
            if current and item.end_seconds - group_start > self.target_seconds:
                groups.append(current)
                current = []
                group_start = item.start_seconds
            current.append(item)
        if current:
            groups.append(current)
        chapters = []
        for index, group in enumerate(groups):
            text = " ".join(item.text for item in group)
            chapters.append(
                VideoChapter(
                    chapter_id=uuid4().hex,
                    chapter_index=index,
                    title=self._title(group[0].text, index),
                    summary=text[:240],
                    start_seconds=group[0].start_seconds,
                    end_seconds=group[-1].end_seconds,
                )
            )
        return chapters

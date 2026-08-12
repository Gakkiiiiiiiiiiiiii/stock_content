from __future__ import annotations

from stock_content.domain.models import KnowledgeUnit, VideoAsset, VideoChapter, VideoSummary


class SummaryGenerator:
    def generate(
        self,
        video: VideoAsset,
        chapters: list[VideoChapter],
        units: list[KnowledgeUnit],
    ) -> VideoSummary:
        points = [unit.statement for unit in units[:8]]
        core = points[0] if points else (chapters[0].summary if chapters else video.transcript_text[:240])
        chapter_lines = "\n".join(f"- {chapter.title}: {chapter.summary}" for chapter in chapters)
        knowledge_lines = "\n".join(f"- [{unit.kind}] {unit.statement}" for unit in units[:20])
        markdown = (
            f"# {video.title}\n\n## 核心摘要\n\n{core}\n\n"
            f"## 章节\n\n{chapter_lines or '- 无'}\n\n## 知识单元\n\n{knowledge_lines or '- 无'}\n"
        )
        confidence = min(0.95, 0.5 + len(units) * 0.03)
        return VideoSummary(video_id=video.video_id, core_summary=core, markdown=markdown, confidence=confidence)

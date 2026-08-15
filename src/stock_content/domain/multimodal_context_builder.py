from __future__ import annotations

import os
from pathlib import Path


class MultimodalContextBuilder:
    def __init__(self, transcript_window_seconds: int | None = None, max_items: int | None = None) -> None:
        self.transcript_window_seconds = int(os.getenv("VIDEO_VISUAL_TRANSCRIPT_WINDOW_SECONDS", str(transcript_window_seconds or 12)))
        self.max_items = int(os.getenv("VIDEO_VISUAL_MAX_CONTEXT_ITEMS", str(max_items or 10)))

    def build(self, transcript: dict, frame_insights: list[dict]) -> dict:
        if not frame_insights:
            return {"items": [], "outline": "", "evidence_segments": []}
        segments = transcript.get("segments") or []
        items: list[dict] = []
        evidence_segments: list[dict] = []
        for insight in frame_insights[: self.max_items]:
            timestamp_ms = int(insight.get("timestamp_ms") or 0)
            related_segments = self._collect_related_segments(segments=segments, timestamp_ms=timestamp_ms)
            related_text = " ".join(segment.get("text", "") for segment in related_segments if str(segment.get("text") or "").strip()).strip()
            item = {
                "timestamp_ms": timestamp_ms,
                "image_path": insight.get("image_path"),
                "trigger_source": insight.get("trigger_source"),
                "ocr_text": str(insight.get("ocr_text") or "").strip(),
                "visual_summary": str(insight.get("visual_summary") or "").strip(),
                "related_text": related_text,
                "themes": list(insight.get("themes") or []),
                "symbols": list(insight.get("symbols") or []),
                "confidence_score": insight.get("confidence_score"),
            }
            items.append(item)
            evidence_segments.append(
                {
                    "type": "visual",
                    "start_ms": timestamp_ms,
                    "end_ms": timestamp_ms,
                    "text": item["visual_summary"] or item["ocr_text"] or related_text,
                    "image_path": item["image_path"],
                    "ocr_text": item["ocr_text"],
                    "visual_summary": item["visual_summary"],
                }
            )
        outline_parts = []
        for index, item in enumerate(items, start=1):
            outline_parts.append(
                "\n".join(
                    [
                        f"[视觉锚点 {index} | {item['timestamp_ms']} ms]",
                        f"画面文字：{item['ocr_text'] or '未成功识别'}",
                        f"画面解读：{item['visual_summary'] or '未生成视觉解读'}",
                        f"关联口播：{item['related_text'] or '未找到强相关口播'}",
                        f"图像证据：{Path(str(item['image_path'] or '')).name or 'unknown'}",
                    ]
                )
            )
        return {
            "items": items,
            "outline": "\n\n".join(outline_parts).strip(),
            "evidence_segments": evidence_segments,
        }

    def _collect_related_segments(self, segments: list[dict], timestamp_ms: int) -> list[dict]:
        window_ms = self.transcript_window_seconds * 1000
        related = []
        for segment in segments:
            start_ms = int(segment.get("start_ms") or 0)
            end_ms = int(segment.get("end_ms") or start_ms)
            if start_ms - window_ms <= timestamp_ms <= end_ms + window_ms:
                related.append(segment)
        return related[:4]


from __future__ import annotations

import os
import re


class TemporalWindowBuilder:
    def __init__(
        self,
        target_duration_seconds: int | None = None,
        min_duration_seconds: int | None = None,
        max_duration_seconds: int | None = None,
        overlap_seconds: int | None = None,
    ) -> None:
        self.target_duration_ms = (
            int(os.getenv("VIDEO_KNOWLEDGE_WINDOW_TARGET_SECONDS", str(target_duration_seconds or 90))) * 1000
        )
        self.min_duration_ms = (
            int(os.getenv("VIDEO_KNOWLEDGE_WINDOW_MIN_SECONDS", str(min_duration_seconds or 25))) * 1000
        )
        self.max_duration_ms = (
            int(os.getenv("VIDEO_KNOWLEDGE_WINDOW_MAX_SECONDS", str(max_duration_seconds or 150))) * 1000
        )
        self.overlap_ms = int(os.getenv("VIDEO_KNOWLEDGE_WINDOW_OVERLAP_SECONDS", str(overlap_seconds or 8))) * 1000

    def build(self, transcript: dict, frame_insights: list[dict] | None = None) -> list[dict]:
        segments = [segment for segment in transcript.get("segments") or [] if str(segment.get("text") or "").strip()]
        if not segments:
            return []
        frames = sorted(frame_insights or [], key=lambda item: int(item.get("timestamp_ms") or 0))
        windows: list[dict] = []
        current: list[dict] = []
        current_start: int | None = None
        current_end: int | None = None

        def flush() -> None:
            nonlocal current, current_start, current_end
            if not current:
                return
            windows.append(self._build_window(len(windows), current, current_start or 0, current_end or 0, frames))
            if self.overlap_ms <= 0:
                current = []
                current_start = None
                current_end = None
                return
            overlap: list[dict] = []
            for segment in reversed(current):
                overlap.insert(0, segment)
                span = int(current_end or 0) - int(overlap[0].get("start_ms") or 0)
                if span >= self.overlap_ms:
                    break
            current = overlap
            current_start = int(current[0].get("start_ms") or 0) if current else None
            current_end = int(current[-1].get("end_ms") or 0) if current else None

        for segment in segments:
            start_ms = int(segment.get("start_ms") or 0)
            end_ms = int(segment.get("end_ms") or start_ms)
            text = str(segment.get("text") or "").strip()
            if current_start is None:
                current_start = start_ms
            projected_duration = end_ms - current_start
            should_split = bool(current) and (
                projected_duration >= self.max_duration_ms
                or (projected_duration >= self.target_duration_ms and self._looks_like_boundary(text))
                or projected_duration >= self.target_duration_ms * 1.35
            )
            if should_split:
                flush()
                if current_start is None:
                    current_start = start_ms
            current.append(segment)
            current_end = end_ms
        flush()
        return windows

    def _build_window(self, index: int, segments: list[dict], start_ms: int, end_ms: int, frames: list[dict]) -> dict:
        window_frames = [frame for frame in frames if start_ms <= int(frame.get("timestamp_ms") or 0) <= end_ms]
        transcript_text = " ".join(
            str(segment.get("text") or "").strip() for segment in segments if str(segment.get("text") or "").strip()
        )
        ocr_lines = self._dedup_lines(str(frame.get("ocr_text") or "") for frame in window_frames)
        ocr_blocks = [block for frame in window_frames for block in (frame.get("ocr_evidence") or {}).get("blocks", [])]
        visual_parts = [
            str(frame.get("visual_summary") or "").strip()
            for frame in window_frames
            if str(frame.get("visual_summary") or "").strip()
        ]
        entities = sorted({entity for frame in window_frames for entity in self._frame_entities(frame)})
        return {
            "window_index": index,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "segments": segments,
            "transcript_text": transcript_text,
            "ocr_text": " | ".join(ocr_lines[:8]),
            "ocr_blocks": ocr_blocks,
            "ocr_confidence_score": self._ocr_confidence(ocr_blocks),
            "visual_summary": " | ".join(visual_parts[:6]),
            "frame_refs": window_frames,
            "entities": entities,
            "speaker_labels": sorted(
                {
                    str(segment.get("speaker_label") or segment.get("speaker") or "")
                    for segment in segments
                    if segment.get("speaker_label") or segment.get("speaker")
                }
            ),
            "confidence_score": self._confidence(segments),
            "vision_confidence_score": self._confidence(window_frames),
        }

    @staticmethod
    def _ocr_confidence(blocks: list[dict]) -> float | None:
        scores = []
        for block in blocks:
            try:
                if block.get("score") is not None:
                    scores.append(float(block["score"]))
            except (TypeError, ValueError):
                continue
        return round(sum(scores) / len(scores), 4) if scores else None

    @staticmethod
    def _looks_like_boundary(text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return any(
            marker in normalized
            for marker in ("接下来", "下面", "再看", "然后看", "最后", "总结一下", "第二个", "第三个")
        )

    @staticmethod
    def _dedup_lines(values) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            for line in str(value or "").splitlines():
                key = " ".join(line.split()).strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(key)
        return result

    @staticmethod
    def _frame_entities(frame: dict) -> list[str]:
        # 多模态模型已判定该帧画面与口播无关：不采信其任何实体
        if frame.get("narration_aligned") is False:
            return []
        # 帧实体只取模型甄别的 symbols 和主画面描述里的代码；
        # 不对原始 OCR 做数字正则——其中行情软件侧边栏公告等内容与口播无关。
        entities = [str(item).strip() for item in frame.get("symbols") or [] if str(item).strip()]
        entities.extend(
            re.findall(r"\b\d{6}\b|\b\d{4}\.HK\b", str(frame.get("visual_summary") or ""), flags=re.IGNORECASE)
        )
        return entities

    @staticmethod
    def _confidence(items: list[dict]) -> float | None:
        """Aggregate one modality's measured confidence only.

        ASR quality and vision confidence are deliberately kept separate:
        ``confidence_score`` aggregates ASR segments, ``vision_confidence_score``
        aggregates vision frames.  Unknown must remain unknown.
        """
        scores: list[float] = []
        for item in items:
            try:
                value = item.get("confidence_score")
                if value is not None:
                    scores.append(float(value))
            except (TypeError, ValueError):
                continue
        return round(sum(scores) / len(scores), 4) if scores else None

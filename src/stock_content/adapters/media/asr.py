from __future__ import annotations

import os
from pathlib import Path

from stock_content.domain.models import TranscriptSegment


class FasterWhisperRecognizer:
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv("CONTENT_ASR_MODEL", "small")
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("install stock-content[media] to enable ASR") from exc
            self._model = WhisperModel(self._model_name, device="auto", compute_type="auto")
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[TranscriptSegment]:
        segments, _ = self._load().transcribe(str(audio_path), language=language, vad_filter=True)
        return [
            TranscriptSegment(
                segment_index=index,
                start_seconds=float(item.start),
                end_seconds=float(item.end),
                text=item.text.strip(),
                confidence=None,
            )
            for index, item in enumerate(segments)
            if item.text.strip()
        ]

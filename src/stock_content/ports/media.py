from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from stock_content.domain.models import TranscriptSegment


class SourceAdapter(Protocol):
    def resolve(self, source_ref: str) -> dict[str, Any]: ...

    def download(self, source_ref: str, target_dir: Path) -> Path: ...


class AudioExtractor(Protocol):
    def extract(self, video_path: Path, target_dir: Path) -> Path: ...


class SpeechRecognizer(Protocol):
    def transcribe(self, audio_path: Path, language: str | None = None) -> list[TranscriptSegment]: ...

from __future__ import annotations

import re
from typing import Protocol

from stock_content.domain.models import TranscriptSegment


class TranscriptTextConverter(Protocol):
    name: str
    version: str

    def convert(self, text: str) -> str: ...


class TranscriptPostprocessor:
    """Auditable normalisation that always preserves original ASR text."""

    _TRADITIONAL_HINTS = frozenset("體臺灣關藍曉設備彈萬與為於億傳儲兌冊劃務區協醫嚴壞壓壘壟壽夢夠奧")

    def __init__(self, converter: TranscriptTextConverter | None = None) -> None:
        self._converter = converter

    def process(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        for segment in segments:
            raw = segment.raw_text or segment.text
            whitespace_normalized = re.sub(r"\s+", " ", raw).strip()
            if self._converter is None:
                if any(char in self._TRADITIONAL_HINTS for char in whitespace_normalized):
                    raise ValueError("TRANSCRIPT_TRADITIONAL_CONVERTER_REQUIRED_HUMAN_REVIEW")
                normalized = whitespace_normalized
            else:
                normalized = self._converter.convert(whitespace_normalized)
            records = []
            if raw != whitespace_normalized:
                records.append({
                    "type": "WHITESPACE_NORMALIZATION", "before": raw, "after": whitespace_normalized,
                    "converter": "builtin-whitespace", "version": "whitespace.v1",
                })
            if whitespace_normalized != normalized:
                records.append({
                    "type": "TRADITIONAL_TO_SIMPLIFIED", "before": whitespace_normalized, "after": normalized,
                    "converter": self._converter.name, "version": self._converter.version,
                })
            segment.raw_text = raw
            segment.normalized_text = normalized
            segment.text = normalized
            segment.correction_records = records
        return segments

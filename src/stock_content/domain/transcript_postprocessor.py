from __future__ import annotations

import re

from stock_content.domain.models import TranscriptSegment


class TranscriptPostprocessor:
    """Auditable normalisation that always preserves original ASR text."""

    def process(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        for segment in segments:
            raw = segment.raw_text or segment.text
            normalized = re.sub(r"\s+", " ", raw).strip()
            segment.raw_text = raw
            segment.normalized_text = normalized
            segment.text = normalized
            segment.correction_records = (
                [{"type": "WHITESPACE_NORMALIZATION", "before": raw, "after": normalized}] if raw != normalized else []
            )
        return segments

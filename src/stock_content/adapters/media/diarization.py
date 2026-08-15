from __future__ import annotations

import os

from stock_content.domain.models import TranscriptSegment


class PyannoteDiarizer:
    """Optional pyannote adapter; a missing model yields explicit UNKNOWN."""

    def annotate(self, audio_path: str | None, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        self.last_status = "DISABLED_BY_CONFIG"
        if not audio_path:
            return segments
        try:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            diarization = pipeline(audio_path)
        except ImportError as exc:
            self.last_status = "UNAVAILABLE"
            if os.getenv("CONTENT_REQUIRE_DIARIZATION", "false").lower() in {"1", "true", "yes"}:
                raise RuntimeError("diarization dependency is unavailable") from exc
            return segments
        except Exception as exc:
            self.last_status = "FAILED"
            if os.getenv("CONTENT_REQUIRE_DIARIZATION", "false").lower() in {"1", "true", "yes"}:
                raise RuntimeError("diarization failed") from exc
            return segments
        for segment in segments:
            midpoint = (segment.start_seconds + segment.end_seconds) / 2
            labels = [
                label
                for turn, _, label in diarization.itertracks(yield_label=True)
                if turn.start <= midpoint <= turn.end
            ]
            if labels:
                segment.speaker_id = str(labels[0])
                segment.speaker_confidence = 1.0
        self.last_status = "SUCCEEDED" if any(item.speaker_id != "UNKNOWN" for item in segments) else "DEGRADED"
        return segments

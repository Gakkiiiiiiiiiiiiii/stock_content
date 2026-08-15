from __future__ import annotations

from stock_content.domain.models import TranscriptSegment


class PyannoteDiarizer:
    """Optional pyannote adapter; a missing model yields explicit UNKNOWN."""

    def annotate(self, audio_path: str | None, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if not audio_path:
            return segments
        try:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            diarization = pipeline(audio_path)
        except Exception:
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
        return segments

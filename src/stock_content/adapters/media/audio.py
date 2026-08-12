from __future__ import annotations

import subprocess
from pathlib import Path


class FfmpegAudioExtractor:
    def extract(self, video_path: Path, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / "audio.wav"
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"audio extraction failed: {completed.stderr[-1500:]}")
        return output

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


class FfmpegFrameExtractor:
    """Deterministic frame extraction shared by OCR and vision stages."""

    def __init__(self, interval_seconds: int = 30) -> None:
        self._interval_seconds = max(1, interval_seconds)

    def extract(self, video_path: Path, output_dir: Path, boundaries_ms: list[int] | None = None) -> list[dict]:
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        pattern = frame_dir / "frame_%06d.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps=1/{self._interval_seconds}",
                "-q:v",
                "3",
                str(pattern),
            ],
            check=True,
            capture_output=True,
        )
        frames: list[dict] = []
        seen_hashes: set[str] = set()
        for index, path in enumerate(sorted(frame_dir.glob("frame_*.jpg"))):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            timestamp_ms = index * self._interval_seconds * 1000
            frames.append(
                {
                    "frame_id": f"frame_{digest[:24]}",
                    "timestamp_ms": timestamp_ms,
                    "image_path": str(path),
                    "image_hash": digest,
                    "trigger_source": "INTERVAL",
                }
            )
        for timestamp_ms in sorted(set(boundaries_ms or [])):
            target = frame_dir / f"boundary_{timestamp_ms}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(timestamp_ms / 1000),
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(target),
                ],
                check=True,
                capture_output=True,
            )
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                frames.append(
                    {
                        "frame_id": f"frame_{digest[:24]}",
                        "timestamp_ms": timestamp_ms,
                        "image_path": str(target),
                        "image_hash": digest,
                        "trigger_source": "CHAPTER_BOUNDARY",
                    }
                )
        return sorted(frames, key=lambda frame: (frame["timestamp_ms"], frame["frame_id"]))

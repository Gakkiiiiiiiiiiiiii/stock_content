from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Iterable


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

    def extract_targeted(
        self,
        video_path: Path,
        output_dir: Path,
        requests: Iterable[Any],
        *,
        existing_image_hashes: set[str] | None = None,
    ) -> list[dict]:
        """Extract planned single frames with exact, deterministic seek inputs.

        ``requests`` is intentionally duck-typed so the infrastructure adapter
        remains independent of the pure domain planner.  Each request must
        expose timestamp/reason/identity fields; no source URL or credential is
        accepted or persisted here.
        """
        frame_dir = output_dir / "knowledge_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        known_hashes = set(existing_image_hashes or ())
        frames: list[dict] = []
        ordered = sorted(
            requests,
            key=lambda item: (
                int(item.timestamp_ms),
                str(item.extraction_reason),
                tuple(item.semantic_segment_ids),
                tuple(item.evidence_window_ids),
            ),
        )
        for index, request in enumerate(ordered):
            timestamp_ms = int(request.timestamp_ms)
            target = frame_dir / f"knowledge_{timestamp_ms:012d}_{index:03d}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{timestamp_ms / 1000:.3f}",
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
            if digest in known_hashes:
                # One image byte sequence is enough evidence.  Skipping its
                # duplicate prevents non-deterministic frame multiplication.
                continue
            known_hashes.add(digest)
            frames.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "image_path": str(target),
                    "image_hash": digest,
                    "trigger_source": str(request.extraction_reason),
                    "extraction_reason": str(request.extraction_reason),
                    "semantic_segment_ids": list(request.semantic_segment_ids),
                    "evidence_window_ids": list(request.evidence_window_ids),
                    "planner_version": str(request.planner_version),
                }
            )
        return sorted(
            frames,
            key=lambda frame: (
                int(frame["timestamp_ms"]),
                str(frame["extraction_reason"]),
                tuple(frame["semantic_segment_ids"]),
            ),
        )

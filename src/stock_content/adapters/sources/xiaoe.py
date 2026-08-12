from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class XiaoeHlsSourceAdapter:
    def resolve(self, source_ref: str) -> dict[str, Any]:
        if not source_ref.startswith(("http://", "https://")):
            raise ValueError("invalid HLS URL")
        return {"source_ref": source_ref, "title": "Xiaoe course video", "author": None}

    def download(self, source_ref: str, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / "source.mp4"
        command = ["ffmpeg", "-nostdin", "-y", "-i", source_ref, "-c", "copy", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg HLS download failed: {completed.stderr[-1500:]}")
        return output

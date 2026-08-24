from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from stock_content.adapters.sources.security import download_hls_playlist, preflight_source_url


class XiaoeHlsSourceAdapter:
    def resolve(self, source_ref: str) -> dict[str, Any]:
        if not source_ref.startswith(("http://", "https://")):
            raise ValueError("invalid HLS URL")
        preflight_source_url(source_ref)
        return {"source_ref": source_ref, "title": "Xiaoe course video", "author": None}

    def download(self, source_ref: str, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / "source.mp4"
        checked_url = preflight_source_url(source_ref)
        local_playlist = download_hls_playlist(checked_url, target_dir)
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-protocol_whitelist",
            "file,crypto,data",
            "-safe",
            "0",
            "-i",
            local_playlist,
            "-c",
            "copy",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg HLS download failed: {completed.stderr[-1500:]}")
        return output

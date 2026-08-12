from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


class BilibiliSourceAdapter:
    """Bilibili adapter backed by yt-dlp, loaded only in the media worker."""

    @staticmethod
    def _url(source_ref: str) -> str:
        if source_ref.startswith(("http://", "https://")):
            return source_ref
        if not re.fullmatch(r"BV[0-9A-Za-z]+", source_ref):
            raise ValueError("invalid Bilibili URL or BV id")
        return f"https://www.bilibili.com/video/{source_ref}"

    @staticmethod
    def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-m", "yt_dlp", *arguments]
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"yt-dlp failed: {detail[-1500:]}") from exc

    def resolve(self, source_ref: str) -> dict[str, Any]:
        url = self._url(source_ref)
        completed = self._run(["--dump-single-json", "--skip-download", "--no-playlist", url])
        payload = json.loads(completed.stdout)
        return {
            "source_ref": url,
            "platform_id": payload.get("id"),
            "title": payload.get("title") or payload.get("id") or "Bilibili video",
            "author": payload.get("uploader"),
            "duration_seconds": payload.get("duration"),
            "published_at": payload.get("timestamp"),
        }

    def download(self, source_ref: str, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        template = str(target_dir / "source.%(ext)s")
        completed = self._run(
            ["--no-playlist", "--no-progress", "--print", "after_move:filepath", "-o", template, self._url(source_ref)]
        )
        path = Path(completed.stdout.strip().splitlines()[-1])
        if not path.exists():
            raise RuntimeError("yt-dlp completed without a media file")
        return path

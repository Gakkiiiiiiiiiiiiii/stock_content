from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from stock_content.adapters.credentials.file_secret_provider import FileSecretProvider, SecretUnavailable
from stock_content.adapters.sources.bilibili_materializer import BilibiliMaterializer, MaterializedMedia
from stock_content.adapters.sources.bilibili_resolver import BilibiliResolver, canonical_bilibili_url
from stock_content.adapters.sources.security import preflight_source_url, safe_download_url, validate_source_url
from stock_content.domain.source_materialization import SourceMaterialization

BILIBILI_ALLOWED_DOMAINS = frozenset(
    {"bilibili.com", "www.bilibili.com", "b23.tv", "bilibili.tv", "bilivideo.com", "biliapi.com"}
)


class BilibiliSourceAdapter:
    """Bilibili adapter backed by yt-dlp, loaded only in the media worker."""

    def __init__(self, *, credential_provider: FileSecretProvider | None = None) -> None:
        self._credential_provider = credential_provider

    @classmethod
    def from_environment(cls) -> "BilibiliSourceAdapter":
        reference = os.getenv("CONTENT_BILIBILI_CREDENTIAL_REF", "").strip()
        cookiefile = os.getenv("CONTENT_BILIBILI_COOKIEFILE", "").strip()
        provider = FileSecretProvider({reference: cookiefile}) if reference and cookiefile else None
        return cls(credential_provider=provider)

    @staticmethod
    def _url(source_ref: str) -> str:
        # Share the exact BV/AV/b23/page handling used by the materialization
        # resolver.  Redirect expansion remains there, behind URL policy.
        return canonical_bilibili_url(source_ref)[0]

    @staticmethod
    def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-m", "yt_dlp", *arguments]
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        except subprocess.CalledProcessError as exc:
            # Extractor diagnostics can echo a cookie-file path, redirect, or
            # signed media URL.  Persist/log only the stable public code.
            raise RuntimeError("BILIBILI_RESOLUTION_FAILED") from exc

    def resolve(self, source_ref: str) -> dict[str, Any]:
        url = self._url(source_ref)
        checked_url = preflight_source_url(url)
        completed = self._run([
            "--ignore-config", "--use-extractors", "Bilibili", "--dump-single-json", "--skip-download", "--no-playlist",
            checked_url,
        ])
        payload = json.loads(completed.stdout)
        return {
            "source_ref": url,
            "platform_id": payload.get("id"),
            "title": payload.get("title") or payload.get("id") or "Bilibili video",
            "author": payload.get("uploader"),
            "duration_seconds": payload.get("duration"),
            "published_at": payload.get("timestamp"),
        }

    def resolve_materialization(
        self, source_ref: str, *, part: int | None = None, credential_ref_hash: str | None = None
    ) -> SourceMaterialization:
        """New ephemeral resolution seam; legacy pipeline methods stay compatible."""
        def extract(arguments: list[str]) -> dict[str, Any]:
            return json.loads(self._run(arguments).stdout)

        cookiefile = None
        if credential_ref_hash:
            if self._credential_provider is None:
                raise RuntimeError("SOURCE_SESSION_EXPIRED")
            try:
                cookiefile = self._credential_provider.resolve_hash(credential_ref_hash)
            except SecretUnavailable as exc:
                raise RuntimeError(exc.code) from exc
        return BilibiliResolver(extractor=extract, cookiefile=cookiefile).resolve(source_ref, part=part)

    def materialize(
        self,
        materialization: SourceMaterialization,
        target_dir: Path,
        *,
        request_video: bool = True,
        expected_duration: float | None = None,
        expected_sha256: str | None = None,
        reresolve: Any | None = None,
    ) -> MaterializedMedia:
        """Register the safe materializer without placing locators in pipeline state."""
        return BilibiliMaterializer().materialize(
            materialization,
            target_dir,
            request_video=request_video,
            expected_duration=expected_duration,
            expected_sha256=expected_sha256,
            reresolve=reresolve,
        )

    @staticmethod
    def _media_streams(payload: dict[str, Any]) -> list[dict[str, Any]]:
        requested = payload.get("requested_formats") or payload.get("requested_downloads")
        if isinstance(requested, list) and requested:
            if any(not isinstance(item, dict) or not item.get("url") for item in requested):
                raise RuntimeError("yt-dlp returned an incomplete audio/video format set")
            return [item for item in requested if isinstance(item, dict)]
        direct = payload.get("url")
        if isinstance(direct, str) and direct:
            if payload.get("vcodec") == "none" or payload.get("acodec") == "none":
                raise RuntimeError("yt-dlp returned media without both audio and video")
            return [payload]
        formats = payload.get("formats")
        if isinstance(formats, list):
            candidates = [item for item in formats if isinstance(item, dict) and item.get("url")]
            combined = [
                item
                for item in candidates
                if item.get("vcodec") not in {None, "none"} and item.get("acodec") not in {None, "none"}
            ]
            combined.sort(
                key=lambda item: (
                    item.get("height") or 0,
                    item.get("tbr") or 0,
                ),
                reverse=True,
            )
            if combined:
                return [combined[0]]
            video = [item for item in candidates if item.get("vcodec") not in {None, "none"}]
            audio = [item for item in candidates if item.get("acodec") not in {None, "none"}]
            if video and audio:
                video.sort(key=lambda item: (item.get("height") or 0, item.get("tbr") or 0), reverse=True)
                audio.sort(key=lambda item: item.get("abr") or item.get("tbr") or 0, reverse=True)
                return [video[0], audio[0]]
        raise RuntimeError("yt-dlp did not expose a complete audio/video media set")

    @staticmethod
    def _headers(stream: dict[str, Any]) -> dict[str, str]:
        raw = stream.get("http_headers")
        if not isinstance(raw, dict):
            return {}
        allowed = {"user-agent", "referer", "accept", "origin"}
        headers = {str(key): str(value) for key, value in raw.items() if str(key).lower() in allowed}
        return headers

    def download(self, source_ref: str, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        checked_url = preflight_source_url(self._url(source_ref))
        metadata = json.loads(
            self._run([
                "--ignore-config",
                "--use-extractors",
                "Bilibili",
                "--dump-single-json",
                "--skip-download",
                "--no-playlist",
                checked_url,
            ]).stdout
        )
        streams = self._media_streams(metadata)
        for stream in streams:
            validate_source_url(str(stream["url"]), allowed_domains=BILIBILI_ALLOWED_DOMAINS)
        local_paths: list[Path] = []
        for index, stream in enumerate(streams):
            local = target_dir / f"source.stream{index}.media"
            safe_download_url(
                str(stream["url"]),
                local,
                allowed_domains=BILIBILI_ALLOWED_DOMAINS,
                headers=self._headers(stream),
            )
            local_paths.append(local)
        if len(local_paths) == 1:
            return local_paths[0]
        video_index = next(
            (index for index, stream in enumerate(streams) if stream.get("vcodec") not in {None, "none"}), None
        )
        audio_index = next(
            (index for index, stream in enumerate(streams) if stream.get("acodec") not in {None, "none"}), None
        )
        if video_index is None or audio_index is None or video_index == audio_index:
            raise RuntimeError("yt-dlp media set cannot be merged with both audio and video")
        output = target_dir / "source.mp4"
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(local_paths[video_index]),
                "-i",
                str(local_paths[audio_index]),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(f"ffmpeg Bilibili merge failed: {completed.stderr[-1500:]}")
        return output

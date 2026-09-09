"""Ephemeral, SSRF-safe Bilibili metadata and stream resolution."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from pydantic import SecretStr

from stock_content.adapters.sources.security import expand_safe_redirect, validate_source_url
from stock_content.domain.source_materialization import (
    MediaStream,
    ResolvedSource,
    SourceMaterialization,
    SubtitleTrack,
)

BILIBILI_ALLOWED_DOMAINS = frozenset(
    {"bilibili.com", "www.bilibili.com", "b23.tv", "bilibili.tv", "bilivideo.com", "biliapi.com"}
)
_BV = re.compile(r"^BV[0-9A-Za-z]+$", re.IGNORECASE)
_AV = re.compile(r"^(?:av)?([1-9][0-9]*)$", re.IGNORECASE)
_VIDEO_PATH = re.compile(r"^/video/(BV[0-9A-Za-z]+|av[1-9][0-9]*)/?$", re.IGNORECASE)
_SAFE_HEADERS = frozenset({"user-agent", "referer", "accept", "origin"})


class BilibiliResolutionError(RuntimeError):
    """Stable error that never embeds a locator, cookie, or extractor output."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_bilibili_url(
    source_ref: str, *, redirect_expander: Callable[[str], str] | None = None
) -> tuple[str, int | None]:
    """Return a public canonical Bilibili page URL and an optional requested part."""
    value = source_ref.strip()
    if _BV.fullmatch(value):
        return f"https://www.bilibili.com/video/BV{value[2:]}", None
    av_match = _AV.fullmatch(value)
    if av_match:
        return f"https://www.bilibili.com/video/av{av_match.group(1)}", None
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        raise BilibiliResolutionError("BILIBILI_SOURCE_INVALID")
    host = parsed.hostname.lower().rstrip(".")
    if host == "b23.tv":
        expander = redirect_expander or (
            lambda short_url: expand_safe_redirect(short_url, allowed_domains=BILIBILI_ALLOWED_DOMAINS)
        )
        expanded = expander(value)
        return canonical_bilibili_url(expanded, redirect_expander=redirect_expander)
    if host not in {"bilibili.com", "www.bilibili.com"}:
        # Keep policy failures distinct from malformed public identifiers.
        validate_source_url(value, allowed_domains=BILIBILI_ALLOWED_DOMAINS)
        raise BilibiliResolutionError("BILIBILI_SOURCE_INVALID")
    match = _VIDEO_PATH.fullmatch(parsed.path)
    if match is None:
        raise BilibiliResolutionError("BILIBILI_SOURCE_INVALID")
    query = parse_qs(parsed.query, keep_blank_values=True)
    parts = query.get("p", [])
    if len(parts) > 1 or (parts and (not parts[0].isdigit() or int(parts[0]) < 1)):
        raise BilibiliResolutionError("BILIBILI_PART_INVALID")
    part = int(parts[0]) if parts else None
    return f"https://www.bilibili.com/video/{match.group(1)}", part


def select_chinese_subtitle(tracks: list[SubtitleTrack]) -> tuple[SubtitleTrack | None, str]:
    """Apply the documented manual/automatic Chinese preference deterministically."""
    def chinese(track: SubtitleTrack) -> bool:
        return track.language.lower().replace("_", "-").startswith("zh")

    def preferred(track: SubtitleTrack) -> bool:
        return track.language.lower().replace("_", "-") in {"zh-hans", "zh-cn"}

    for source, exact, reason in (
        ("official", True, "manual_zh_hans_or_zh_cn"),
        ("official", False, "manual_other_chinese"),
        ("automatic", True, "automatic_zh_hans_or_zh_cn"),
        ("automatic", False, "automatic_other_chinese"),
    ):
        selected = next(
            (
                track
                for track in tracks
                if track.source == source and chinese(track) and (preferred(track) == exact)
            ),
            None,
        )
        if selected is not None:
            return selected, reason
    return None, "asr_required"


class BilibiliResolver:
    """Use yt-dlp only as the Bilibili extractor, retaining locators in memory."""

    def __init__(
        self,
        *,
        extractor: Callable[[list[str]], dict[str, Any]] | None = None,
        redirect_expander: Callable[[str], str] | None = None,
        cookiefile: Path | None = None,
    ) -> None:
        self._extractor = extractor or self._extract
        self._redirect_expander = redirect_expander
        # A path is deliberately the only credential surface accepted here.
        # Its content stays in yt-dlp and is never projected into a task or
        # materialization object.
        self._cookiefile = cookiefile

    @staticmethod
    def _extract(arguments: list[str]) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "yt_dlp", *arguments],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            payload = json.loads(completed.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
            raise BilibiliResolutionError("BILIBILI_RESOLUTION_FAILED") from exc
        if not isinstance(payload, dict):
            raise BilibiliResolutionError("BILIBILI_RESOLUTION_FAILED")
        return payload

    @staticmethod
    def _headers(value: object) -> dict[str, SecretStr]:
        if not isinstance(value, dict):
            return {}
        return {
            str(name): SecretStr(str(header))
            for name, header in value.items()
            if str(name).lower() in _SAFE_HEADERS
        }

    @classmethod
    def _subtitle_tracks(cls, payload: dict[str, Any]) -> list[SubtitleTrack]:
        tracks: list[SubtitleTrack] = []
        for source, key in (("official", "subtitles"), ("automatic", "automatic_captions")):
            raw = payload.get(key)
            if not isinstance(raw, dict):
                continue
            for language, entries in raw.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
                        continue
                    url = str(entry["url"])
                    validate_source_url(url, allowed_domains=BILIBILI_ALLOWED_DOMAINS)
                    tracks.append(SubtitleTrack(
                        track_id=str(entry.get("id") or f"{source}-{language}-{len(tracks)}"),
                        language=str(language), source=source, format=str(entry.get("ext") or "vtt"),
                        url=SecretStr(url), headers=cls._headers(entry.get("http_headers")),
                    ))
        return tracks

    @classmethod
    def _streams(cls, payload: dict[str, Any]) -> list[MediaStream]:
        entries = payload.get("requested_formats") or payload.get("requested_downloads")
        if not isinstance(entries, list) or not entries:
            entries = [payload] if payload.get("url") else []
        streams: list[MediaStream] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
                continue
            url = str(entry["url"])
            validate_source_url(url, allowed_domains=BILIBILI_ALLOWED_DOMAINS)
            kind = "muxed"
            if entry.get("vcodec") == "none":
                kind = "audio"
            elif entry.get("acodec") == "none":
                kind = "video"
            streams.append(MediaStream(
                stream_id=str(entry.get("format_id") or index), kind=kind, url=SecretStr(url),
                headers=cls._headers(entry.get("http_headers")),
                codec=str(entry.get("vcodec") or entry.get("acodec") or "") or None,
                bitrate=int(entry["tbr"]) if isinstance(entry.get("tbr"), (int, float)) else None,
                width=entry.get("width") if isinstance(entry.get("width"), int) else None,
                height=entry.get("height") if isinstance(entry.get("height"), int) else None,
            ))
        if not streams:
            raise BilibiliResolutionError("BILIBILI_MEDIA_UNAVAILABLE")
        return streams

    def resolve(self, source_ref: str, *, part: int | None = None) -> SourceMaterialization:
        canonical_url, requested_part = canonical_bilibili_url(source_ref, redirect_expander=self._redirect_expander)
        if part is not None and requested_part is not None and part != requested_part:
            raise BilibiliResolutionError("BILIBILI_PART_CONFLICT")
        selected_part = part if part is not None else requested_part
        validate_source_url(canonical_url, allowed_domains=BILIBILI_ALLOWED_DOMAINS)
        arguments = [
            "--ignore-config",
            "--use-extractors",
            "Bilibili",
            "--dump-single-json",
            "--skip-download",
            "--no-playlist",
            canonical_url,
        ]
        if self._cookiefile is not None:
            arguments[0:0] = ["--cookies", str(self._cookiefile)]
        if selected_part is not None:
            arguments.extend(["--playlist-items", str(selected_part)])
        payload = self._extractor(arguments)
        entries = payload.get("entries")
        if isinstance(entries, list):
            if not entries or not isinstance(entries[0], dict):
                raise BilibiliResolutionError("BILIBILI_PART_NOT_FOUND")
            payload = entries[0]
        reported = str(payload.get("webpage_url") or canonical_url)
        reported_url, _ = canonical_bilibili_url(reported, redirect_expander=self._redirect_expander)
        platform_id = str(payload.get("id") or reported_url.rsplit("/", 1)[-1])
        published_at = payload.get("timestamp")
        timestamp = datetime.fromtimestamp(published_at, UTC) if isinstance(published_at, (int, float)) else None
        public = ResolvedSource(
            source_type="bilibili", canonical_source_ref=reported_url, canonical_url=reported_url,
            source_identity_hash=hashlib.sha256(f"bilibili:{reported_url}".encode()).hexdigest(),
            platform_id=platform_id, part_id=str(selected_part) if selected_part is not None else None,
            title=str(payload.get("title") or platform_id), author=str(payload.get("uploader") or "") or None,
            published_at=timestamp,
            duration_seconds=float(payload["duration"])
            if isinstance(payload.get("duration"), (int, float)) and float(payload["duration"]) > 0
            else None,
        )
        return SourceMaterialization(
            public=public,
            streams=self._streams(payload),
            subtitles=self._subtitle_tracks(payload),
        )


__all__ = [
    "BILIBILI_ALLOWED_DOMAINS", "BilibiliResolutionError", "BilibiliResolver", "canonical_bilibili_url",
    "select_chinese_subtitle",
]

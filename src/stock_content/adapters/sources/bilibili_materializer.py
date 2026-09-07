"""Local Bilibili media materialization with one bounded locator refresh."""
from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_content.adapters.sources.bilibili_resolver import select_chinese_subtitle
from stock_content.adapters.sources.security import safe_download_url
from stock_content.domain.source_materialization import (
    MaterializedSubtitleCue,
    MaterializedSubtitleTrack,
    SourceMaterialization,
    SubtitleTrack,
)


class BilibiliMaterializationError(RuntimeError):
    """A stable, safe-to-log materialization failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class MaterializedMedia:
    path: Path
    sha256: str
    duration_seconds: float
    subtitle_metadata: dict[str, str | None]
    subtitle_tracks: tuple[MaterializedSubtitleTrack, ...] = field(default_factory=tuple)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BilibiliMaterializer:
    """Fetch ephemeral locators through the safe-fetch boundary only."""

    def __init__(
        self,
        *,
        downloader: Callable[..., str] = safe_download_url,
        probe: Callable[[Path], dict[str, Any]] | None = None,
        merger: Callable[[list[Path], Path], None] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._downloader = downloader
        self._probe = probe or self._ffprobe
        self._merger = merger or self._ffmpeg_merge
        self._now = now

    @staticmethod
    def _ffprobe(path: Path) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            payload = json.loads(result.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            raise BilibiliMaterializationError("MEDIA_PROBE_FAILED") from exc
        if not isinstance(payload, dict):
            raise BilibiliMaterializationError("MEDIA_PROBE_FAILED")
        return payload

    @staticmethod
    def _ffmpeg_merge(inputs: list[Path], output: Path) -> None:
        if len(inputs) != 2:
            raise BilibiliMaterializationError("BILIBILI_MEDIA_STREAMSET_INCOMPLETE")
        # Local files only: signed locators and headers never reach ffmpeg or
        # its process listing/logs.
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y", "-i", str(inputs[0]), "-i", str(inputs[1]),
                    "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(output),
                ],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise BilibiliMaterializationError("MEDIA_MERGE_FAILED") from exc
        if result.returncode != 0 or not output.is_file():
            raise BilibiliMaterializationError("MEDIA_MERGE_FAILED")

    @staticmethod
    def _expired(materialization: SourceMaterialization, now: datetime) -> bool:
        for stream in materialization.streams:
            if stream.expires_at is not None and stream.expires_at.astimezone(UTC) <= now:
                return True
        return False

    @staticmethod
    def _is_expiry_failure(exc: Exception) -> bool:
        text = str(exc).upper()
        return "401" in text or "403" in text or "EXPIRED" in text

    @staticmethod
    def _safe_headers(headers: dict[str, object]) -> dict[str, str]:
        # SecretStr unwrap happens only at the immediate HTTP boundary.  Cookie
        # and Authorization are never accepted from extractor metadata.
        result: dict[str, str] = {}
        for name, value in headers.items():
            if name.lower() in {"cookie", "authorization", "proxy-authorization"}:
                raise BilibiliMaterializationError("SOURCE_HEADER_UNSAFE")
            getter = getattr(value, "get_secret_value", None)
            result[name] = getter() if callable(getter) else str(value)
        return result

    def _download(self, materialization: SourceMaterialization, target_dir: Path) -> Path:
        if not materialization.streams:
            raise BilibiliMaterializationError("BILIBILI_MEDIA_UNAVAILABLE")
        paths: list[Path] = []
        for index, stream in enumerate(materialization.streams):
            local = target_dir / f"source.stream{index}.media"
            self._downloader(
                stream.url.get_secret_value(),
                local,
                headers=self._safe_headers(stream.headers),
            )
            if not local.is_file() or local.stat().st_size == 0:
                raise BilibiliMaterializationError("MEDIA_EMPTY")
            paths.append(local)
        if len(paths) == 1:
            return paths[0]
        video = next((index for index, item in enumerate(materialization.streams) if item.kind == "video"), None)
        audio = next((index for index, item in enumerate(materialization.streams) if item.kind == "audio"), None)
        if video is None or audio is None:
            raise BilibiliMaterializationError("BILIBILI_MEDIA_STREAMSET_INCOMPLETE")
        output = target_dir / "source.muxed.media"
        self._merger([paths[video], paths[audio]], output)
        return output

    @staticmethod
    def _validate_probe(payload: dict[str, Any], *, request_video: bool, expected_duration: float | None) -> float:
        try:
            format_info = payload["format"]
            duration = float(format_info["duration"])
            bitrate = int(float(format_info["bit_rate"]))
            streams = payload["streams"]
        except (KeyError, TypeError, ValueError) as exc:
            raise BilibiliMaterializationError("MEDIA_PROBE_INVALID") from exc
        if duration <= 0 or bitrate <= 0 or not isinstance(streams, list):
            raise BilibiliMaterializationError("MEDIA_PROBE_INVALID")
        if expected_duration is not None and abs(duration - expected_duration) > max(2.0, expected_duration * 0.05):
            raise BilibiliMaterializationError("MEDIA_DURATION_MISMATCH")
        audio = any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio" and str(stream.get("codec_name") or "")
            for stream in streams
        )
        video = any(
            isinstance(stream, dict) and stream.get("codec_type") == "video" and str(stream.get("codec_name") or "")
            for stream in streams
        )
        if not audio or (request_video and not video):
            raise BilibiliMaterializationError("MEDIA_TRACKS_INVALID")
        return duration

    @staticmethod
    def _normalize_text(value: object) -> str:
        # Subtitle text is evidence, so normalisation is deliberately small:
        # decode markup/entities and canonicalise whitespace/Unicode only.
        text = unicodedata.normalize("NFKC", html.unescape(str(value)))
        text = re.sub(r"<[^>]*>", "", text)
        return " ".join(text.split())

    @staticmethod
    def _timestamp(value: str) -> int:
        match = re.fullmatch(r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[,.](\d{1,3})", value.strip())
        if match is None:
            raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID")
        hours, minutes, seconds, millis = match.groups()
        minute_value, second_value = int(minutes), int(seconds)
        if minute_value >= 60 or second_value >= 60:
            raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID")
        return ((int(hours or 0) * 3600 + minute_value * 60 + second_value) * 1000) + int(millis.ljust(3, "0"))

    @classmethod
    def _text_cues(cls, value: str, subtitle_format: str) -> list[tuple[int, int, str]]:
        if subtitle_format == "json":
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID") from exc
            entries = payload.get("body") if isinstance(payload, dict) else payload
            if not isinstance(entries, list):
                raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID")
            result = []
            for item in entries:
                if not isinstance(item, dict):
                    raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID")
                try:
                    start = round(float(item.get("from", item.get("start"))) * 1000)
                    end = round(float(item.get("to", item.get("end"))) * 1000)
                except (TypeError, ValueError) as exc:
                    raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID") from exc
                result.append((start, end, str(item.get("content", item.get("text", "")))))
            return result
        if subtitle_format not in {"vtt", "srt"}:
            raise BilibiliMaterializationError("SUBTITLE_FORMAT_UNSUPPORTED")
        blocks = re.split(r"\r?\n\s*\r?\n", value.replace("\ufeff", ""))
        result: list[tuple[int, int, str]] = []
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines or lines[0].upper() == "WEBVTT" or lines[0].upper().startswith(("NOTE", "STYLE", "REGION")):
                continue
            timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
            if timing_index is None:
                raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID")
            parts = lines[timing_index].split("-->", 1)
            # VTT allows a settings suffix after the end timestamp.
            end_token = parts[1].strip().split(maxsplit=1)[0]
            start, end = cls._timestamp(parts[0]), cls._timestamp(end_token)
            text = "\n".join(lines[timing_index + 1 :])
            result.append((start, end, text))
        return result

    @classmethod
    def _subtitle_track(
        cls, subtitle: SubtitleTrack, target_dir: Path, downloader: Callable[..., str], duration_seconds: float
    ) -> MaterializedSubtitleTrack:
        subtitle_format = subtitle.format.lower().lstrip(".")
        if subtitle_format not in {"vtt", "srt", "json"}:
            raise BilibiliMaterializationError("SUBTITLE_FORMAT_UNSUPPORTED")
        raw = target_dir / f"subtitle.{subtitle.track_id}.{subtitle_format}"
        downloader(subtitle.url.get_secret_value(), raw, headers=cls._safe_headers(subtitle.headers))
        if not raw.is_file() or raw.stat().st_size == 0 or raw.stat().st_size > 4 * 1024 * 1024:
            raise BilibiliMaterializationError("SUBTITLE_EMPTY")
        try:
            contents = raw.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID") from exc
        parsed = cls._text_cues(contents, subtitle_format)
        if not parsed or len(parsed) > 10_000:
            raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID")
        cue_values: list[MaterializedSubtitleCue] = []
        previous_end = 0
        upper_bound = round(duration_seconds * 1000)
        for start, end, raw_text in parsed:
            normalized_text = cls._normalize_text(raw_text)
            if start < 0 or end <= start or end > upper_bound or start < previous_end or not normalized_text:
                raise BilibiliMaterializationError("SUBTITLE_CUES_INVALID")
            cue_hash = hashlib.sha256(
                json.dumps([start, end, raw_text, normalized_text], ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            cue_values.append(MaterializedSubtitleCue(start, end, raw_text, normalized_text, cue_hash))
            previous_end = end
        raw_hash = _sha256(raw)
        normalised = "\n".join(
            f"{cue.start_ms}\t{cue.end_ms}\t{cue.normalized_text}" for cue in cue_values
        ) + "\n"
        normalized_hash = hashlib.sha256(normalised.encode()).hexdigest()
        identity = json.dumps(
            [subtitle.track_id, subtitle.language, subtitle.source, raw_hash, normalized_hash], separators=(",", ":")
        )
        return MaterializedSubtitleTrack(
            track_id=subtitle.track_id,
            language=subtitle.language,
            source=subtitle.source,
            raw_sha256=raw_hash,
            normalized_sha256=normalized_hash,
            artifact_id="subtitle-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            cues=tuple(cue_values),
        )

    def _subtitle_metadata(
        self, materialization: SourceMaterialization, target_dir: Path, duration_seconds: float
    ) -> tuple[dict[str, str | None], tuple[MaterializedSubtitleTrack, ...]]:
        subtitle, reason = select_chinese_subtitle(materialization.subtitles)
        if subtitle is None:
            return ({
                "language": None,
                "type": "asr",
                "selection_reason": reason,
                "raw_sha256": None,
                "normalized_sha256": None,
            }, ())
        track = self._subtitle_track(subtitle, target_dir, self._downloader, duration_seconds)
        return ({
            "language": subtitle.language,
            "type": subtitle.source,
            "selection_reason": reason,
            "raw_sha256": track.raw_sha256,
            "normalized_sha256": track.normalized_sha256,
        }, (track,))

    def materialize(
        self,
        materialization: SourceMaterialization,
        target_dir: Path,
        *,
        request_video: bool = True,
        expected_duration: float | None = None,
        expected_sha256: str | None = None,
        reresolve: Callable[[], SourceMaterialization] | None = None,
    ) -> MaterializedMedia:
        """Materialize once, with one and only one safe re-resolution attempt."""
        target_dir.mkdir(parents=True, exist_ok=True)
        active = materialization
        for attempt in range(2):
            try:
                if self._expired(active, self._now()):
                    raise BilibiliMaterializationError("SOURCE_STREAM_EXPIRED")
                media = self._download(active, target_dir)
                digest = _sha256(media)
                if expected_sha256 is not None and digest != expected_sha256:
                    raise BilibiliMaterializationError("MEDIA_SHA256_MISMATCH")
                duration = self._validate_probe(
                    self._probe(media), request_video=request_video, expected_duration=expected_duration
                )
                subtitle_metadata, subtitle_tracks = self._subtitle_metadata(active, target_dir, duration)
                return MaterializedMedia(media, digest, duration, subtitle_metadata, subtitle_tracks)
            except Exception as exc:
                if attempt == 0 and reresolve is not None and self._is_expiry_failure(exc):
                    active = reresolve()
                    continue
                if isinstance(exc, BilibiliMaterializationError):
                    raise
                if self._is_expiry_failure(exc):
                    raise BilibiliMaterializationError("SOURCE_STREAM_EXPIRED") from exc
                raise BilibiliMaterializationError("MEDIA_DOWNLOAD_FAILED") from exc
        raise BilibiliMaterializationError("SOURCE_STREAM_EXPIRED")


__all__ = ["BilibiliMaterializationError", "BilibiliMaterializer", "MaterializedMedia"]

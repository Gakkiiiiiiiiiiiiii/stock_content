"""Safe local materialization for authorized Xiaoe HLS media."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from stock_content.adapters.sources.bilibili_materializer import MaterializedMedia
from stock_content.adapters.sources.dash_materializer import DashLocalizer, DashMaterializationError
from stock_content.adapters.sources.security import UnsafeSourceURL, download_hls_playlist, validate_source_url
from stock_content.adapters.sources.xiaoe_page import xiaoe_allowed_domains
from stock_content.domain.drm_policy import DrmPolicyError, require_supported_hls
from stock_content.domain.source_materialization import SourceMaterialization


class XiaoeMaterializationError(RuntimeError):
    """Stable failure code for legal/session/DRM-safe handling."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class XiaoeMaterializer:
    """Use only local playlists with a single permitted locator refresh."""

    def __init__(
        self,
        *,
        playlist_downloader: Callable[..., str] = download_hls_playlist,
        dash_localizer: DashLocalizer | None = None,
        ffmpeg: Callable[[list[str]], None] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._playlist_downloader = playlist_downloader
        self._dash_localizer = dash_localizer or DashLocalizer()
        self._ffmpeg = ffmpeg or self._run_ffmpeg
        self._now = now
        self._allowed_domains = xiaoe_allowed_domains()

    @staticmethod
    def _run_ffmpeg(arguments: list[str]) -> None:
        try:
            result = subprocess.run(arguments, capture_output=True, text=True)
        except OSError as exc:
            raise XiaoeMaterializationError("MEDIA_MATERIALIZATION_FAILED") from exc
        if result.returncode != 0:
            raise XiaoeMaterializationError("MEDIA_MATERIALIZATION_FAILED")

    @staticmethod
    def _safe_headers(headers: dict[str, object]) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, value in headers.items():
            if name.lower() in {"authorization", "cookie", "host", "proxy-authorization"}:
                raise XiaoeMaterializationError("SOURCE_HEADER_UNSAFE")
            getter = getattr(value, "get_secret_value", None)
            result[name] = getter() if callable(getter) else str(value)
        return result

    @staticmethod
    def _expired(materialization: SourceMaterialization, now: datetime) -> bool:
        return any(
            stream.expires_at is not None and stream.expires_at.astimezone(UTC) <= now
            for stream in materialization.streams
        )

    @staticmethod
    def _failure_code(exc: Exception, materialization: SourceMaterialization) -> str:
        if isinstance(exc, XiaoeMaterializationError):
            return exc.code
        text = str(exc).upper()
        if "404" in text or "NOT_FOUND" in text or "MEDIA_EMPTY" in text:
            return "SOURCE_MEDIA_NOT_FOUND"
        if "401" in text or "403" in text or "EXPIRED" in text:
            return "SOURCE_SESSION_EXPIRED" if materialization.credential_ref_hash else "SOURCE_SIGNED_URL_EXPIRED"
        return "MEDIA_MATERIALIZATION_FAILED"

    def _download(self, materialization: SourceMaterialization, target_dir: Path) -> Path:
        if len(materialization.streams) != 1:
            raise XiaoeMaterializationError("SOURCE_MEDIA_NOT_FOUND")
        stream = materialization.streams[0]
        if stream.kind == "dash":
            try:
                local_mpd = self._dash_localizer.materialize(
                    stream.url.get_secret_value(),
                    target_dir,
                    allowed_domains=self._allowed_domains,
                    headers=self._safe_headers(stream.headers),
                )
                output = target_dir / "source.mp4"
                self._ffmpeg(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-protocol_whitelist",
                        "file,crypto,data",
                        "-safe",
                        "0",
                        "-i",
                        str(local_mpd),
                        "-c",
                        "copy",
                        str(output),
                    ]
                )
                if not output.is_file() or output.stat().st_size == 0:
                    raise XiaoeMaterializationError("SOURCE_MEDIA_NOT_FOUND")
                return output
            finally:
                self._dash_localizer.cleanup(target_dir)
        if stream.kind != "hls":
            raise XiaoeMaterializationError("SOURCE_MEDIA_NOT_FOUND")
        playlist = self._playlist_downloader(
            stream.url.get_secret_value(),
            target_dir,
            allowed_domains=self._allowed_domains,
            headers=self._safe_headers(stream.headers),
            manifest_validator=require_supported_hls,
        )
        if not Path(playlist).is_file():
            raise XiaoeMaterializationError("SOURCE_MEDIA_NOT_FOUND")
        output = target_dir / "source.mp4"
        self._ffmpeg(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-protocol_whitelist",
                "file,crypto,data",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-c",
                "copy",
                str(output),
            ]
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise XiaoeMaterializationError("SOURCE_MEDIA_NOT_FOUND")
        return output

    def _validate_refresh(
        self, previous: SourceMaterialization, refreshed: SourceMaterialization
    ) -> SourceMaterialization:
        """Allow a refresh to replace only an ephemeral locator, never identity."""
        if (
            refreshed.public.source_type != previous.public.source_type
            or refreshed.public.canonical_source_ref != previous.public.canonical_source_ref
            or refreshed.public.source_identity_hash != previous.public.source_identity_hash
            or refreshed.public.platform_id != previous.public.platform_id
            or refreshed.public.part_id != previous.public.part_id
            or refreshed.credential_ref_hash != previous.credential_ref_hash
            or len(refreshed.streams) != 1
        ):
            raise XiaoeMaterializationError("SOURCE_LOCATOR_MISMATCH")
        stream = refreshed.streams[0]
        try:
            # DNS pinning and redirect policy are enforced again by the DASH
            # localizer on every fetch.  This preflight validates the new
            # secret locator's public host and URL form without retaining it.
            validate_source_url(
                stream.url.get_secret_value(), allowed_domains=self._allowed_domains, resolve_host=False
            )
        except UnsafeSourceURL as exc:
            raise XiaoeMaterializationError(exc.code) from exc
        return refreshed

    def _cleanup_refresh_artifacts(self, target_dir: Path) -> None:
        """Discard only our prior local DASH graph and partial output."""
        self._dash_localizer.cleanup(target_dir)
        output = target_dir / "source.mp4"
        try:
            root = target_dir.resolve()
            if output.is_file() and not output.is_symlink() and output.resolve().is_relative_to(root):
                output.unlink()
        except OSError:
            return

    def materialize(
        self,
        materialization: SourceMaterialization,
        target_dir: Path,
        *,
        reresolve: Callable[[], SourceMaterialization] | None = None,
        **_: object,
    ) -> MaterializedMedia:
        """Retry exactly once after an expired signed locator/session response."""
        target_dir.mkdir(parents=True, exist_ok=True)
        active = materialization
        for attempt in range(2):
            try:
                if self._expired(active, self._now()):
                    raise XiaoeMaterializationError(
                        "SOURCE_SESSION_EXPIRED" if active.credential_ref_hash else "SOURCE_SIGNED_URL_EXPIRED"
                    )
                path = self._download(active, target_dir)
                return MaterializedMedia(
                    path,
                    _sha256(path),
                    0.0,
                    {
                        "language": None,
                        "type": "asr",
                        "selection_reason": "asr_required",
                        "raw_sha256": None,
                        "normalized_sha256": None,
                    },
                )
            except DrmPolicyError as exc:
                raise XiaoeMaterializationError(exc.code) from exc
            except DashMaterializationError as exc:
                if attempt == 0 and exc.retryable_locator_expiry and reresolve:
                    self._cleanup_refresh_artifacts(target_dir)
                    try:
                        active = self._validate_refresh(active, reresolve())
                    except Exception as refresh_error:
                        raise XiaoeMaterializationError(self._failure_code(refresh_error, active)) from refresh_error
                    continue
                raise XiaoeMaterializationError(exc.code) from exc
            except Exception as exc:
                code = self._failure_code(exc, active)
                if attempt == 0 and code in {"SOURCE_SESSION_EXPIRED", "SOURCE_SIGNED_URL_EXPIRED"} and reresolve:
                    self._cleanup_refresh_artifacts(target_dir)
                    try:
                        active = self._validate_refresh(active, reresolve())
                    except Exception as refresh_error:
                        raise XiaoeMaterializationError(self._failure_code(refresh_error, active)) from refresh_error
                    continue
                raise XiaoeMaterializationError(code) from exc
        raise XiaoeMaterializationError("SOURCE_MEDIA_NOT_FOUND")


__all__ = ["XiaoeMaterializationError", "XiaoeMaterializer"]

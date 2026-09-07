"""An injectable, allowlisted Playwright session for authorized Xiaoe pages."""
from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr

from stock_content.adapters.sources.security import UnsafeSourceURL, validate_source_url

_SAFE_HEADERS = frozenset({"accept", "origin", "referer", "user-agent"})
_SUBTITLE_SUFFIXES = (".vtt", ".srt", ".ass", ".ttml")


class BrowserSessionError(RuntimeError):
    """Safe-to-log browser failure without locator, cookie, or page details."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CapturedResponse:
    url: SecretStr
    kind: str
    headers: dict[str, SecretStr]
    expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PageCapture:
    course_id: str
    lesson_id: str
    title: str
    author: str | None
    published_at: datetime | None
    media: CapturedResponse | None
    subtitles: tuple[CapturedResponse, ...] = ()
    drm_hint: str | None = None


class PlaywrightSession:
    """Capture authorized media responses while every route remains constrained.

    Playwright remains an optional runtime dependency.  Production wiring can
    inject this port; deterministic tests inject a small fake instead.
    """

    def __init__(
        self,
        *,
        allowed_domains: frozenset[str],
        page_url_for: Callable[[str], str],
        timeout_seconds: float = 60.0,
    ) -> None:
        self._allowed_domains = allowed_domains
        self._page_url_for = page_url_for
        self._timeout_ms = max(1, int(timeout_seconds * 1000))

    def capture(self, source_identity: str, storage_state: Path) -> PageCapture:
        page_url = self._page_url_for(source_identity)
        try:
            validate_source_url(page_url, allowed_domains=self._allowed_domains)
        except UnsafeSourceURL as exc:
            raise BrowserSessionError(exc.code) from exc
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserSessionError("SOURCE_BROWSER_UNAVAILABLE") from exc

        if not storage_state.is_file() or storage_state.is_symlink():
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        captured: list[CapturedResponse] = []
        subtitles: list[CapturedResponse] = []
        unsafe_route = False
        temporary = tempfile.TemporaryDirectory(prefix="xiaoe-state-")
        copied_state = Path(temporary.name) / "storage-state.json"
        browser = context = playwright = None
        try:
            shutil.copyfile(storage_state, copied_state)
            if os.name != "nt":
                copied_state.chmod(0o400)
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(storage_state=str(copied_state))
            page = context.new_page()

            def route_request(route) -> None:
                nonlocal unsafe_route
                try:
                    validate_source_url(route.request.url, allowed_domains=self._allowed_domains)
                except UnsafeSourceURL:
                    unsafe_route = True
                    route.abort()
                else:
                    route.continue_()

            def observe(response) -> None:
                value = response.url
                try:
                    validate_source_url(value, allowed_domains=self._allowed_domains)
                except UnsafeSourceURL:
                    return
                content_type = str(response.headers.get("content-type") or "").lower()
                path = urlsplit(value).path.lower()
                kind = "hls" if ".m3u8" in path or "mpegurl" in content_type else "dash" if (
                    path.endswith(".mpd") or "dash+xml" in content_type
                ) else "subtitle" if path.endswith(_SUBTITLE_SUFFIXES) else ""
                if not kind:
                    return
                request_headers = response.request.headers
                headers = {
                    name: SecretStr(str(value))
                    for name, value in request_headers.items()
                    if name.lower() in _SAFE_HEADERS
                }
                item = CapturedResponse(SecretStr(value), kind, headers)
                (subtitles if kind == "subtitle" else captured).append(item)

            context.route("**/*", route_request)
            page.on("response", observe)
            page.goto(page_url, wait_until="networkidle", timeout=self._timeout_ms)
            if unsafe_route:
                raise BrowserSessionError("SOURCE_DOMAIN_NOT_ALLOWLISTED")
            media = next((item for item in captured if item.kind == "hls"), None) or next(
                (item for item in captured if item.kind == "dash"), None
            )
            course_id, lesson_id = source_identity.split("/", 1)
            title = page.title().strip() or lesson_id
            return PageCapture(course_id, lesson_id, title, None, None, media, tuple(subtitles))
        except BrowserSessionError:
            raise
        except Exception as exc:
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc
        finally:
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
            if playwright is not None:
                playwright.stop()
            temporary.cleanup()


__all__ = ["BrowserSessionError", "CapturedResponse", "PageCapture", "PlaywrightSession"]

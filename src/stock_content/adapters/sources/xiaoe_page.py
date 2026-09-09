"""Authorized Xiaoe page resolution without persisting page/session locators."""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

from pydantic import SecretStr

from stock_content.adapters.browser.playwright_session import BrowserSessionError, PageCapture, PlaywrightSession
from stock_content.adapters.credentials.file_secret_provider import FileSecretProvider, SecretUnavailable
from stock_content.adapters.sources.security import UnsafeSourceURL, validate_source_url
from stock_content.domain.drm_policy import DrmPolicyError, require_supported_drm
from stock_content.domain.source_materialization import (
    MediaStream,
    ResolvedSource,
    SourceMaterialization,
    SubtitleTrack,
)
from stock_content.domain.source_url import canonical_public_source_url

_XIAOE_BASE_DOMAINS = frozenset({"xiaoe-tech.com", "m.xiaoe-tech.com"})
_IDENTITY = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


class XiaoeResolutionError(RuntimeError):
    """Stable source error; source locators and browser diagnostics stay hidden."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def xiaoe_allowed_domains(value: str | None = None) -> frozenset[str]:
    """Parse only explicit DNS suffixes, never URLs or wildcard patterns."""
    configured = value if value is not None else os.getenv("CONTENT_XIAOE_ALLOWED_DOMAINS", "")
    domains = set(_XIAOE_BASE_DOMAINS)
    for raw in configured.split(","):
        domain = raw.strip().lower().rstrip(".")
        if not domain:
            continue
        if "://" in domain or "/" in domain or any(part == "" for part in domain.split(".")):
            raise XiaoeResolutionError("SOURCE_DOMAIN_NOT_ALLOWLISTED")
        domains.add(domain)
    return frozenset(domains)


def _safe_headers(headers: dict[str, SecretStr]) -> dict[str, SecretStr]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in {"accept", "origin", "referer", "user-agent"}
    }


def _public_url(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


class XiaoePageResolver:
    """Resolve a stable course/lesson identity with a configured session port."""

    def __init__(
        self,
        *,
        credential_provider: FileSecretProvider,
        browser: PlaywrightSession,
        allowed_domains: frozenset[str] | None = None,
        public_url_for: Callable[[str], str] | None = None,
    ) -> None:
        self._credential_provider = credential_provider
        self._browser = browser
        self._allowed_domains = allowed_domains or xiaoe_allowed_domains()
        self._public_url_for = public_url_for

    def resolve(self, source_identity: str, *, credential_ref_hash: str | None) -> SourceMaterialization:
        try:
            storage_state = self._credential_provider.resolve_hash(credential_ref_hash)
            capture = self._browser.capture(source_identity, storage_state)
            return self._materialization(source_identity, capture, credential_ref_hash)
        except (SecretUnavailable, BrowserSessionError) as exc:
            raise XiaoeResolutionError(getattr(exc, "code", "SOURCE_SESSION_EXPIRED")) from exc
        except DrmPolicyError as exc:
            raise XiaoeResolutionError(exc.code) from exc

    def storage_state_for(self, credential_ref_hash: str | None):
        """Return the configured private state at the immediate worker boundary."""
        try:
            return self._credential_provider.resolve_hash(credential_ref_hash)
        except SecretUnavailable as exc:
            raise XiaoeResolutionError(exc.code) from exc

    def _materialization(
        self, source_identity: str, capture: PageCapture, credential_ref_hash: str | None
    ) -> SourceMaterialization:
        if capture.media is None:
            raise XiaoeResolutionError("SOURCE_MEDIA_NOT_FOUND")
        if capture.course_id + "/" + capture.lesson_id != source_identity:
            raise XiaoeResolutionError("SOURCE_MEDIA_NOT_FOUND")
        try:
            require_supported_drm(capture.drm_hint)
            media_url = capture.media.url.get_secret_value()
            validate_source_url(media_url, allowed_domains=self._allowed_domains)
        except UnsafeSourceURL as exc:
            raise XiaoeResolutionError(exc.code) from exc
        except DrmPolicyError as exc:
            raise XiaoeResolutionError(exc.code) from exc
        streams = [MediaStream(
            stream_id="authorized-media", kind=capture.media.kind, url=capture.media.url,
            headers=_safe_headers(capture.media.headers), expires_at=capture.media.expires_at,
        )]
        subtitles: list[SubtitleTrack] = []
        for index, subtitle in enumerate(capture.subtitles):
            try:
                validate_source_url(subtitle.url.get_secret_value(), allowed_domains=self._allowed_domains)
            except UnsafeSourceURL as exc:
                raise XiaoeResolutionError(exc.code) from exc
            subtitles.append(SubtitleTrack(
                track_id=f"subtitle-{index}", language="und", source="official", format="vtt",
                url=subtitle.url, headers=_safe_headers(subtitle.headers),
            ))
        canonical_url = (
            canonical_public_source_url("xiaoe", self._public_url_for(source_identity))
            if self._public_url_for is not None
            else None
        )
        public = ResolvedSource(
            source_type="xiaoe", canonical_source_ref=source_identity, canonical_url=canonical_url,
            source_identity_hash=hashlib.sha256(f"xiaoe:{source_identity}".encode()).hexdigest(),
            platform_id=capture.course_id, part_id=capture.lesson_id, title=capture.title,
            author=capture.author, published_at=capture.published_at, duration_seconds=capture.duration_seconds,
        )
        return SourceMaterialization(
            public=public, streams=streams, subtitles=subtitles, credential_ref_hash=credential_ref_hash,
            requires_reresolve=True,
        )


class XiaoeHlsResolver:
    """Keep a direct HLS locator only in the caller's runtime materialization."""

    def __init__(self, *, allowed_domains: frozenset[str] | None = None) -> None:
        self._allowed_domains = allowed_domains or xiaoe_allowed_domains()

    def resolve(self, source_url: str) -> SourceMaterialization:
        try:
            validate_source_url(source_url, allowed_domains=self._allowed_domains)
        except UnsafeSourceURL as exc:
            raise XiaoeResolutionError(exc.code) from exc
        public_url = _public_url(source_url)
        title = urlsplit(public_url).path.rsplit("/", 1)[-1] or "Xiaoe course video"
        public = ResolvedSource(
            source_type="xiaoe_hls", canonical_source_ref=public_url, canonical_url=public_url,
            source_identity_hash=hashlib.sha256(f"xiaoe_hls:{public_url}".encode()).hexdigest(),
            platform_id=hashlib.sha256(public_url.encode()).hexdigest()[:24], title=title,
        )
        return SourceMaterialization(
            public=public,
            streams=[MediaStream(stream_id="direct-hls", kind="hls", url=SecretStr(source_url))],
            subtitles=[], requires_reresolve=False,
        )


def _page_url_from_template(template: str, source_identity: str) -> str:
    values = source_identity.split("/", 1)
    if len(values) != 2 or not all(_IDENTITY.fullmatch(value) for value in values):
        raise XiaoeResolutionError("SOURCE_MEDIA_NOT_FOUND")
    course_id, lesson_id = values
    if not any(marker in template for marker in ("{source_ref}", "{course_id}", "{lesson_id}")):
        raise XiaoeResolutionError("SOURCE_MEDIA_NOT_FOUND")
    try:
        return template.format(source_ref=source_identity, course_id=course_id, lesson_id=lesson_id)
    except (KeyError, ValueError) as exc:
        raise XiaoeResolutionError("SOURCE_MEDIA_NOT_FOUND") from exc


def page_resolver_from_environment() -> XiaoePageResolver | None:
    """Build only when the operator has configured an explicit, legal session."""
    if os.getenv("CONTENT_XIAOE_PAGE_RESOLVER_ENABLED", "").lower() != "true":
        return None
    state_file = os.getenv("CONTENT_XIAOE_STORAGE_STATE_FILE")
    template = os.getenv("CONTENT_XIAOE_PAGE_URL_TEMPLATE")
    credential_ref = os.getenv("CONTENT_XIAOE_CREDENTIAL_REF", "xiaoe-storage-state")
    if not state_file or not template or not any(
        marker in template for marker in ("{source_ref}", "{course_id}", "{lesson_id}")
    ):
        return None
    domains = xiaoe_allowed_domains()
    timeout = float(os.getenv("CONTENT_XIAOE_PAGE_TIMEOUT_SECONDS", "60"))

    def page_url_for(source_ref: str) -> str:
        return _page_url_from_template(template, source_ref)

    return XiaoePageResolver(
        credential_provider=FileSecretProvider({credential_ref: state_file}),
        browser=PlaywrightSession(
            allowed_domains=domains, page_url_for=page_url_for,
            timeout_seconds=timeout,
        ),
        allowed_domains=domains, public_url_for=page_url_for,
    )


__all__ = [
    "XiaoeHlsResolver", "XiaoePageResolver", "XiaoeResolutionError", "page_resolver_from_environment",
    "xiaoe_allowed_domains", "_page_url_from_template",
]

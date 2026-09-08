"""SSRF-safe URL validation for source adapters."""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import math
import os
import re
import shutil
import socket
import ssl
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from stock_content.domain.drm_policy import require_supported_hls

DEFAULT_ALLOWED_DOMAINS = frozenset(
    {
        "bilibili.com",
        "www.bilibili.com",
        "b23.tv",
        "bilivideo.com",
        "biliapi.com",
        "xiaoe-tech.com",
        "m.xiaoe-tech.com",
    }
)


class UnsafeSourceURL(ValueError):
    """Stable, safe-to-log source URL policy failure."""

    def __init__(self, message: str, *, code: str = "SOURCE_URL_UNSAFE", url: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.url = url


class SourceDownloadHTTPError(RuntimeError):
    """A redacted HTTP response from the safe byte-download boundary.

    Callers may use ``status`` to make a narrowly scoped retry decision, but
    the exception deliberately has no URL, response body, or request headers.
    """

    def __init__(self, status: int) -> None:
        super().__init__(f"SOURCE_DOWNLOAD_HTTP_{status}")
        self.status = status


class HlsResourceLimitError(RuntimeError):
    """Redacted, deterministic HLS graph resource-limit failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _is_allowed_host(host: str, allowed_domains: set[str] | frozenset[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)


def _safe_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
        or not ip.is_global
    )


def validate_source_url(
    url: str,
    *,
    allowed_domains: set[str] | frozenset[str] = DEFAULT_ALLOWED_DOMAINS,
    resolve_host: bool = True,
) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise UnsafeSourceURL("source URL is malformed", code="SOURCE_URL_MALFORMED", url=url) from exc
    if parsed.scheme not in {"http", "https"} or not host:
        raise UnsafeSourceURL(
            "source URL must use HTTP(S) and include a hostname", code="SOURCE_URL_MALFORMED", url=url
        )
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeSourceURL("source URL userinfo is not allowed", code="SOURCE_URL_USERINFO", url=url)
    host = host.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeSourceURL("source hostname is malformed", code="SOURCE_URL_MALFORMED", url=url) from exc
    if not _is_allowed_host(host, allowed_domains):
        raise UnsafeSourceURL(
            f"source domain is not allowlisted: {host}", code="SOURCE_DOMAIN_NOT_ALLOWLISTED", url=url
        )
    if resolve_host:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(
                    host, port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM
                )
            }
        except OSError as exc:
            raise UnsafeSourceURL(
                f"source DNS resolution failed: {host}", code="SOURCE_DNS_RESOLUTION_FAILED", url=url
            ) from exc
        if not addresses or any(not _safe_ip(address) for address in addresses):
            raise UnsafeSourceURL(
                "source resolves to a private or local address", code="SOURCE_PRIVATE_ADDRESS", url=url
            )
    return url


def validate_redirect(url: str, **kwargs: object) -> str:
    """Redirects must pass the exact same checks as initial URLs."""
    try:
        return validate_source_url(url, **kwargs)
    except UnsafeSourceURL as exc:
        raise UnsafeSourceURL(
            "source redirect target failed URL policy",
            code="SOURCE_REDIRECT_UNSAFE",
            url=url,
        ) from exc


class _ValidatingRedirectHandler(HTTPRedirectHandler):
    """Validate each Location before urllib follows it.

    This handler is retained for the cheap metadata preflight path. Actual
    media bytes use the pinned connection path below, which does not delegate
    redirect handling to an external downloader.
    """

    def __init__(self, *, allowed_domains: set[str] | frozenset[str], resolve_host: bool) -> None:
        super().__init__()
        self.allowed_domains = allowed_domains
        self.resolve_host = resolve_host

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        target = urljoin(req.full_url, newurl)
        try:
            validate_redirect(
                target,
                allowed_domains=self.allowed_domains,
                resolve_host=self.resolve_host,
            )
        except UnsafeSourceURL as exc:
            raise UnsafeSourceURL(
                "source redirect target failed URL policy",
                code="SOURCE_REDIRECT_UNSAFE",
                url=target,
            ) from exc
        return super().redirect_request(req, fp, code, msg, headers, target)


def preflight_source_url(
    url: str,
    *,
    allowed_domains: set[str] | frozenset[str] = DEFAULT_ALLOWED_DOMAINS,
    resolve_host: bool = True,
    opener: Any | None = None,
    timeout: float = 10.0,
) -> str:
    """Validate an adapter URL and every HTTP redirect before downloading.

    The response body is intentionally not consumed. Callers must use
    ``safe_download_url`` (or ``download_hls_playlist``) for actual bytes;
    this function only establishes a checked URL/redirect chain. HTTP status
    failures still prove that the redirect policy was traversed.
    """

    validate_source_url(url, allowed_domains=allowed_domains, resolve_host=resolve_host)
    if opener is None:
        opener = build_opener(
            _ValidatingRedirectHandler(allowed_domains=allowed_domains, resolve_host=resolve_host)
        )
    request = Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "stock-content-source-preflight/1"},
        method="HEAD",
    )
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as exc:
        # HTTP status is not a URL safety failure.  urllib has already passed
        # each redirect through _ValidatingRedirectHandler before raising it.
        response = exc
    final_url = response.geturl() if hasattr(response, "geturl") else url
    validate_source_url(final_url, allowed_domains=allowed_domains, resolve_host=resolve_host)
    close = getattr(response, "close", None)
    if close is not None:
        close()
    return final_url


def _validated_endpoint(
    url: str,
    *,
    allowed_domains: set[str] | frozenset[str],
) -> tuple[Any, str, int, list[str]]:
    """Return a URL parse and a DNS-pinned, policy-checked endpoint."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise UnsafeSourceURL("source URL is malformed", code="SOURCE_URL_MALFORMED", url=url) from exc
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        raise UnsafeSourceURL(
            "source URL must use HTTP(S) and include a hostname", code="SOURCE_URL_MALFORMED", url=url
        )
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeSourceURL("source URL userinfo is not allowed", code="SOURCE_URL_USERINFO", url=url)
    host = host.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeSourceURL("source hostname is malformed", code="SOURCE_URL_MALFORMED", url=url) from exc
    if not _is_allowed_host(host, allowed_domains):
        raise UnsafeSourceURL(
            f"source domain is not allowlisted: {host}", code="SOURCE_DOMAIN_NOT_ALLOWLISTED", url=url
        )
    resolved_port = port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = list(
            dict.fromkeys(
                item[4][0]
                for item in socket.getaddrinfo(host, resolved_port, type=socket.SOCK_STREAM)
            )
        )
    except OSError as exc:
        raise UnsafeSourceURL(
            f"source DNS resolution failed: {host}", code="SOURCE_DNS_RESOLUTION_FAILED", url=url
        ) from exc
    if not addresses or any(not _safe_ip(address) for address in addresses):
        raise UnsafeSourceURL(
            "source resolves to a private or local address", code="SOURCE_PRIVATE_ADDRESS", url=url
        )
    return parsed, host, resolved_port, addresses


def _connect_pinned(
    host: str,
    port: int,
    addresses: list[str],
    *,
    timeout: float,
    tls: bool,
) -> socket.socket:
    """Connect only to an already validated address and optionally negotiate TLS."""
    last_error: OSError | None = None
    for address in addresses:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((address, port))
            if tls:
                context = ssl.create_default_context()
                return context.wrap_socket(sock, server_hostname=host)
            return sock
        except (OSError, ssl.SSLError) as exc:
            last_error = exc
            sock.close()
    raise OSError(f"could not connect to validated source endpoint: {host}") from last_error


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, *, addresses: list[str], timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._validated_addresses = addresses

    def connect(self) -> None:
        self.sock = _connect_pinned(
            self.host, self.port, self._validated_addresses, timeout=self.timeout, tls=False
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, *, addresses: list[str], timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())
        self._validated_addresses = addresses

    def connect(self) -> None:
        self.sock = _connect_pinned(
            self.host, self.port, self._validated_addresses, timeout=self.timeout, tls=True
        )


def _open_safe_response(
    url: str,
    *,
    allowed_domains: set[str] | frozenset[str],
    timeout: float,
    headers: dict[str, str] | None = None,
    cookie_jar: Any | None = None,
    max_redirects: int = 5,
) -> tuple[str, http.client.HTTPConnection, http.client.HTTPResponse]:
    """Open one safe response, manually traversing and validating redirects."""
    current = url
    for _ in range(max_redirects + 1):
        parsed, host, port, addresses = _validated_endpoint(current, allowed_domains=allowed_domains)
        connection_type = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
        connection = connection_type(host, port, addresses=addresses, timeout=timeout)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        try:
            request_headers = {"Host": parsed.netloc, "User-Agent": "stock-content-safe-fetch/1"}
            if headers:
                request_headers.update(headers)
            if cookie_jar is not None:
                cookie = cookie_jar.header_for(current)
                if cookie:
                    request_headers["Cookie"] = cookie
            connection.request(
                "GET",
                path,
                headers=request_headers,
            )
            response = connection.getresponse()
        except Exception:
            connection.close()
            raise
        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            response.close()
            connection.close()
            if not location:
                raise UnsafeSourceURL(
                    "source redirect did not include a Location", code="SOURCE_REDIRECT_UNSAFE", url=current
                )
            current = urljoin(current, location)
            # _validated_endpoint on the next iteration is the security boundary.
            continue
        return current, connection, response
    raise UnsafeSourceURL("source redirect chain is too long", code="SOURCE_REDIRECT_UNSAFE", url=current)


def expand_safe_redirect(
    url: str,
    *,
    allowed_domains: set[str] | frozenset[str] = DEFAULT_ALLOWED_DOMAINS,
    timeout: float = 10.0,
    max_redirects: int = 5,
) -> str:
    """Expand a public short link through the DNS-pinned redirect boundary.

    This intentionally exposes only the final public URL to the caller; no
    response body, cookies, authorization headers, or redirect diagnostics are
    retained.  ``_open_safe_response`` validates every hop and bounds the
    chain before any subsequent consumer sees the result.
    """
    final_url, connection, response = _open_safe_response(
        url,
        allowed_domains=allowed_domains,
        timeout=timeout,
        max_redirects=max_redirects,
    )
    try:
        if response.status < 200 or response.status >= 400:
            raise UnsafeSourceURL("source redirect expansion failed", code="SOURCE_REDIRECT_UNSAFE", url=url)
        return final_url
    finally:
        response.close()
        connection.close()


def safe_download_url(
    url: str,
    target: str | os.PathLike[str],
    *,
    allowed_domains: set[str] | frozenset[str] = DEFAULT_ALLOWED_DOMAINS,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
    headers: dict[str, str] | None = None,
    cookie_jar: Any | None = None,
) -> str:
    """Download bytes with per-request DNS pinning and manual safe redirects."""
    safe_headers: dict[str, str] = {}
    for name, value in (headers or {}).items():
        normalized = name.lower()
        if normalized in {"host", "proxy", "proxy-authorization", "cookie"}:
            raise UnsafeSourceURL("unsafe download header is not allowed", code="SOURCE_HEADER_UNSAFE", url=url)
        if normalized in {"referer", "origin"}:
            validate_source_url(value, allowed_domains=allowed_domains)
        safe_headers[name] = value
    final_url, connection, response = _open_safe_response(
        url, allowed_domains=allowed_domains, timeout=timeout, headers=safe_headers, cookie_jar=cookie_jar
    )
    try:
        if response.status < 200 or response.status >= 300:
            raise SourceDownloadHTTPError(response.status)
        destination = os.fspath(target)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        written = 0
        with open(destination, "wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise RuntimeError("source download exceeds configured size limit")
                stream.write(chunk)
    finally:
        response.close()
        connection.close()
    return final_url


_HLS_URI = re.compile(r'URI=(?:"([^"]+)"|([^,\s]+))')
_HLS_PLAYLIST_TAGS = {"#EXT-X-STREAM-INF", "#EXT-X-I-FRAME-STREAM-INF", "#EXT-X-MEDIA"}
_HLS_MEDIA_SUFFIXES = frozenset({".aac", ".ac3", ".ec3", ".m4a", ".m4s", ".mp3", ".mp4", ".ts", ".webm"})
_HLS_ATTRIBUTE = re.compile(r'([A-Z0-9-]+)=(?:"([^"]*)"|([^,]*))')
_HLS_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


def _hls_attributes(line: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2) if match.group(2) is not None else match.group(3)
        for match in _HLS_ATTRIBUTE.finditer(line)
    }


def _hls_selected_master_lines(lines: list[str]) -> set[int] | None:
    """Return master-line indexes retained for one video plus required audio.

    Master playlists are an alternative graph, not a request to download every
    rendition.  Retain the first declared full video variant (publisher order
    is deterministic) and its DEFAULT audio rendition when one is referenced.
    """
    variants: list[tuple[int, int, dict[str, str]]] = []
    audio: list[tuple[int, dict[str, str]]] = []
    for index, line in enumerate(lines):
        tag = line.strip().split(":", 1)[0]
        if (
            tag == "#EXT-X-STREAM-INF"
            and index + 1 < len(lines)
            and lines[index + 1].strip()
            and not lines[index + 1].lstrip().startswith("#")
        ):
            variants.append((index, index + 1, _hls_attributes(line)))
        elif tag == "#EXT-X-MEDIA":
            attributes = _hls_attributes(line)
            if attributes.get("TYPE", "").upper() == "AUDIO" and "URI" in attributes:
                audio.append((index, attributes))
    if not variants:
        return None
    stream_index, uri_index, stream_attributes = variants[0]
    retained = {stream_index, uri_index}
    audio_group = stream_attributes.get("AUDIO")
    if audio_group:
        candidates = [(index, attributes) for index, attributes in audio if attributes.get("GROUP-ID") == audio_group]
        if candidates:
            selected = next((item for item in candidates if item[1].get("DEFAULT", "").upper() == "YES"), candidates[0])
            retained.add(selected[0])
    return retained


def _hls_local_asset_suffix(remote_url: str, *, asset_kind: str, has_map: bool) -> str:
    """Keep only a format-relevant, allowlisted suffix on opaque local names."""
    if asset_kind == "key":
        return ".key"
    suffix = os.path.splitext(urlparse(remote_url).path)[1].lower()
    if suffix in _HLS_MEDIA_SUFFIXES:
        return suffix
    if asset_kind == "map":
        return ".m4s" if suffix == ".m4s" else ".mp4"
    return ".m4s" if has_map else ".ts"


def download_hls_playlist(
    source_url: str,
    target_dir: str | os.PathLike[str],
    *,
    allowed_domains: set[str] | frozenset[str] = DEFAULT_ALLOWED_DOMAINS,
    max_depth: int = 8,
    max_playlists: int = 32,
    max_assets: int = 5_000,
    max_total_bytes: int = 2 * 1024 * 1024 * 1024,
    max_planned_duration_seconds: float = 6 * 60 * 60,
    headers: dict[str, str] | None = None,
    cookie_jar: Any | None = None,
    manifest_validator: Any | None = None,
) -> str:
    """Materialize a safe HLS graph locally and return its local playlist path.

    ffmpeg is intentionally given only local paths. Every manifest, segment,
    encryption key, map, and redirect is fetched through ``safe_download_url``
    or its bounded manifest equivalent.
    """
    if max_depth < 0 or max_playlists < 1 or max_assets < 1 or max_total_bytes < 1 or max_planned_duration_seconds <= 0:
        raise ValueError("HLS resource limits must be positive")
    root = os.path.abspath(os.fspath(target_dir))
    safe_headers: dict[str, str] = {}
    for name, value in (headers or {}).items():
        normalized = name.lower()
        if normalized in {"host", "proxy", "proxy-authorization", "cookie", "authorization"}:
            raise UnsafeSourceURL("unsafe download header is not allowed", code="SOURCE_HEADER_UNSAFE", url=source_url)
        if normalized in {"referer", "origin"}:
            validate_source_url(value, allowed_domains=allowed_domains)
        safe_headers[name] = value
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, ".safe-hls")
    os.makedirs(cache, exist_ok=True)
    playlists: dict[str, str] = {}
    assets: dict[tuple[str, str, bool], str] = {}
    downloaded_bytes = 0
    planned_duration_by_playlist: dict[str, float] = {}
    validator = manifest_validator or require_supported_hls

    def local_name(remote: str, suffix: str) -> str:
        digest = hashlib.sha256(remote.encode("utf-8")).hexdigest()[:24]
        return os.path.join(cache, digest + suffix)

    def fetch_manifest(remote_url: str, *, depth: int) -> str:
        nonlocal downloaded_bytes
        if depth > max_depth:
            raise HlsResourceLimitError("HLS_NESTING_LIMIT_EXCEEDED")
        if remote_url in playlists:
            return playlists[remote_url]
        if len(set(playlists.values())) >= max_playlists:
            raise HlsResourceLimitError("HLS_PLAYLIST_LIMIT_EXCEEDED")
        final_url, connection, response = _open_safe_response(
            remote_url, allowed_domains=allowed_domains, timeout=30.0, headers=safe_headers, cookie_jar=cookie_jar
        )
        try:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"HLS manifest returned HTTP {response.status}")
            remaining_bytes = max_total_bytes - downloaded_bytes
            if remaining_bytes <= 0:
                raise HlsResourceLimitError("HLS_TOTAL_BYTES_LIMIT_EXCEEDED")
            body = response.read(min(_HLS_MAX_MANIFEST_BYTES, remaining_bytes) + 1)
            if len(body) > _HLS_MAX_MANIFEST_BYTES:
                raise HlsResourceLimitError("HLS_MANIFEST_BYTES_LIMIT_EXCEEDED")
            if len(body) > remaining_bytes:
                raise HlsResourceLimitError("HLS_TOTAL_BYTES_LIMIT_EXCEEDED")
            downloaded_bytes += len(body)
        finally:
            response.close()
            connection.close()
        try:
            text = body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RuntimeError("HLS manifest is not UTF-8") from exc
        if not text.lstrip().startswith("#EXTM3U"):
            raise RuntimeError("source is not a supported HLS playlist")
        validator(text)
        output = local_name(final_url, ".m3u8")
        playlists[remote_url] = output
        playlists[final_url] = output
        planned_duration_by_playlist[output] = 0.0
        lines = text.splitlines()
        retained_master_lines = _hls_selected_master_lines(lines)
        rewritten: list[str] = []
        pending_variant = False
        has_map = any(item.strip().startswith("#EXT-X-MAP") for item in lines)
        for index, line in enumerate(lines):
            stripped = line.strip()
            tag_name = stripped.split(":", 1)[0] if stripped.startswith("#") else ""
            if retained_master_lines is not None and (tag_name in _HLS_PLAYLIST_TAGS or pending_variant):
                if index not in retained_master_lines:
                    pending_variant = False
                    continue
            if (
                retained_master_lines is not None
                and not stripped.startswith("#")
                and index > 0
                and lines[index - 1].strip().startswith("#EXT-X-STREAM-INF")
                and index not in retained_master_lines
            ):
                continue
            if stripped.startswith("#"):
                if tag_name == "#EXTINF":
                    try:
                        duration = float(stripped.split(":", 1)[1].split(",", 1)[0])
                    except (IndexError, ValueError) as exc:
                        raise HlsResourceLimitError("HLS_DURATION_INVALID") from exc
                    if not math.isfinite(duration) or duration < 0:
                        raise HlsResourceLimitError("HLS_DURATION_INVALID")
                    planned_duration_by_playlist[output] += duration
                    if planned_duration_by_playlist[output] > max_planned_duration_seconds:
                        raise HlsResourceLimitError("HLS_DURATION_LIMIT_EXCEEDED")
                if "URI=" in line:
                    is_playlist_tag = tag_name in _HLS_PLAYLIST_TAGS

                    def replace_uri(match: re.Match[str]) -> str:
                        remote = urljoin(final_url, match.group(1) or match.group(2))
                        if is_playlist_tag:
                            local = fetch_manifest(remote, depth=depth + 1)
                        else:
                            asset_kind = (
                                "key" if tag_name == "#EXT-X-KEY" else "map" if tag_name == "#EXT-X-MAP" else "media"
                            )
                            local = fetch_asset(remote, asset_kind=asset_kind, has_map=has_map)
                        relative = os.path.relpath(local, os.path.dirname(output)).replace(os.sep, "/")
                        quote = '"' if match.group(1) is not None else ""
                        return "URI=" + quote + relative + quote

                    line = _HLS_URI.sub(replace_uri, line)
                pending_variant = tag_name == "#EXT-X-STREAM-INF"
                rewritten.append(line)
                continue
            if not stripped:
                rewritten.append(line)
                continue
            remote = urljoin(final_url, stripped)
            if pending_variant:
                local = fetch_manifest(remote, depth=depth + 1)
                pending_variant = False
            else:
                local = fetch_asset(remote, asset_kind="media", has_map=has_map)
            rewritten.append(os.path.relpath(local, os.path.dirname(output)).replace(os.sep, "/"))
        def has_remote_uri(line: str) -> bool:
            lowered = line.lower()
            if any(protocol in lowered for protocol in ("http:", "https:", "rtmp:", "data:")):
                return True
            match = _HLS_URI.search(line)
            value = (match.group(1) or match.group(2)) if match else line.lstrip()
            return value.lower().startswith(("http:", "https:", "rtmp:", "data:"))

        if any(has_remote_uri(line) for line in rewritten):
            raise RuntimeError("HLS playlist contains an unreplaced remote URI")
        with open(output, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(rewritten) + "\n")
        return output

    def fetch_asset(remote_url: str, *, asset_kind: str, has_map: bool) -> str:
        nonlocal downloaded_bytes
        key = (remote_url, asset_kind, has_map)
        if key in assets:
            return assets[key]
        if len(assets) >= max_assets:
            raise HlsResourceLimitError("HLS_ASSET_LIMIT_EXCEEDED")
        remaining_bytes = max_total_bytes - downloaded_bytes
        if remaining_bytes <= 0:
            raise HlsResourceLimitError("HLS_TOTAL_BYTES_LIMIT_EXCEEDED")
        output = local_name(remote_url, _hls_local_asset_suffix(remote_url, asset_kind=asset_kind, has_map=has_map))
        try:
            safe_download_url(
                remote_url,
                output,
                allowed_domains=allowed_domains,
                headers=safe_headers,
                cookie_jar=cookie_jar,
                max_bytes=remaining_bytes,
            )
        except RuntimeError as exc:
            if str(exc) == "source download exceeds configured size limit":
                raise HlsResourceLimitError("HLS_TOTAL_BYTES_LIMIT_EXCEEDED") from exc
            raise
        downloaded_bytes += os.path.getsize(output)
        assets[key] = output
        return output

    try:
        return fetch_manifest(source_url, depth=0)
    except Exception:
        # This directory contains only opaque local HLS files made for the
        # current attempt.  Never leave a partial graph to be consumed later.
        shutil.rmtree(cache, ignore_errors=True)
        raise


__all__ = [
    "DEFAULT_ALLOWED_DOMAINS",
    "HlsResourceLimitError",
    "SourceDownloadHTTPError",
    "UnsafeSourceURL",
    "preflight_source_url",
    "download_hls_playlist",
    "expand_safe_redirect",
    "safe_download_url",
    "validate_redirect",
    "validate_source_url",
]

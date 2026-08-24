"""SSRF-safe URL validation for source adapters."""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import re
import socket
import ssl
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

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
            self._host, self.port, self._validated_addresses, timeout=self.timeout, tls=False
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, *, addresses: list[str], timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())
        self._validated_addresses = addresses

    def connect(self) -> None:
        self.sock = _connect_pinned(
            self._host, self.port, self._validated_addresses, timeout=self.timeout, tls=True
        )


def _open_safe_response(
    url: str,
    *,
    allowed_domains: set[str] | frozenset[str],
    timeout: float,
    headers: dict[str, str] | None = None,
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


def safe_download_url(
    url: str,
    target: str | os.PathLike[str],
    *,
    allowed_domains: set[str] | frozenset[str] = DEFAULT_ALLOWED_DOMAINS,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
    headers: dict[str, str] | None = None,
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
        url, allowed_domains=allowed_domains, timeout=timeout, headers=safe_headers
    )
    try:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"source download returned HTTP {response.status}")
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


def download_hls_playlist(
    source_url: str,
    target_dir: str | os.PathLike[str],
    *,
    allowed_domains: set[str] | frozenset[str] = DEFAULT_ALLOWED_DOMAINS,
    max_depth: int = 8,
) -> str:
    """Materialize a safe HLS graph locally and return its local playlist path.

    ffmpeg is intentionally given only local paths. Every manifest, segment,
    encryption key, map, and redirect is fetched through ``safe_download_url``
    or its bounded manifest equivalent.
    """
    root = os.path.abspath(os.fspath(target_dir))
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, ".safe-hls")
    os.makedirs(cache, exist_ok=True)
    playlists: dict[str, str] = {}
    assets: dict[str, str] = {}

    def local_name(remote: str, suffix: str) -> str:
        digest = hashlib.sha256(remote.encode("utf-8")).hexdigest()[:24]
        return os.path.join(cache, digest + suffix)

    def fetch_manifest(remote_url: str, *, depth: int) -> str:
        if depth > max_depth:
            raise RuntimeError("HLS playlist nesting exceeds configured limit")
        if remote_url in playlists:
            return playlists[remote_url]
        final_url, connection, response = _open_safe_response(
            remote_url, allowed_domains=allowed_domains, timeout=30.0
        )
        try:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"HLS manifest returned HTTP {response.status}")
            body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise RuntimeError("HLS manifest exceeds configured size limit")
        finally:
            response.close()
            connection.close()
        try:
            text = body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RuntimeError("HLS manifest is not UTF-8") from exc
        if not text.lstrip().startswith("#EXTM3U"):
            raise RuntimeError("source is not a supported HLS playlist")
        output = local_name(final_url, ".m3u8")
        playlists[remote_url] = output
        lines = text.splitlines()
        rewritten: list[str] = []
        pending_variant = False
        for line in lines:
            stripped = line.strip()
            tag_name = stripped.split(":", 1)[0] if stripped.startswith("#") else ""
            if stripped.startswith("#"):
                if "URI=" in line:
                    is_playlist_tag = tag_name in _HLS_PLAYLIST_TAGS

                    def replace_uri(match: re.Match[str]) -> str:
                        remote = urljoin(final_url, match.group(1) or match.group(2))
                        if is_playlist_tag:
                            local = fetch_manifest(remote, depth=depth + 1)
                        else:
                            local = fetch_asset(remote)
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
                local = fetch_asset(remote)
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

    def fetch_asset(remote_url: str) -> str:
        if remote_url in assets:
            return assets[remote_url]
        output = local_name(remote_url, ".bin")
        safe_download_url(remote_url, output, allowed_domains=allowed_domains)
        assets[remote_url] = output
        return output

    return fetch_manifest(source_url, depth=0)


__all__ = [
    "DEFAULT_ALLOWED_DOMAINS",
    "UnsafeSourceURL",
    "preflight_source_url",
    "download_hls_playlist",
    "safe_download_url",
    "validate_redirect",
    "validate_source_url",
]

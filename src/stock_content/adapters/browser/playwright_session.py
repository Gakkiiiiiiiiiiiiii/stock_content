"""An injectable, allowlisted Playwright session for authorized Xiaoe pages."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr

from stock_content.adapters.sources.security import UnsafeSourceURL, validate_source_url

_SAFE_HEADERS = frozenset({"accept", "origin", "referer", "user-agent"})
_SUBTITLE_SUFFIXES = (".vtt", ".srt", ".ass", ".ttml")
_WINDOWS_LOCAL_SYSTEM_SID = "S-1-5-18"
_WINDOWS_READ_DATA = 0x1
_WINDOWS_FULL_CONTROL = 0x1F01FF


class BrowserSessionError(RuntimeError):
    """Safe-to-log browser failure without locator, cookie, or page details."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class StorageStateCookieJar:
    """Short-lived cookie selector.  Its values never cross a worker boundary."""

    def __init__(self, storage_state: Path) -> None:
        try:
            payload = json.loads(storage_state.read_text(encoding="utf-8"))
            cookies = payload.get("cookies")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc
        if not isinstance(cookies, list):
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        self._cookies = tuple(item for item in cookies if isinstance(item, dict))
        if not any(
            isinstance(item.get("name"), str) and isinstance(item.get("value"), str) and item.get("domain")
            for item in self._cookies
        ):
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(cookies=<redacted>)"

    def header_for(self, url: str) -> str | None:
        parsed = urlsplit(url)
        host, path = (parsed.hostname or "").lower(), parsed.path or "/"
        if parsed.scheme not in {"http", "https"} or not host:
            return None
        values: list[str] = []
        now = datetime.now().timestamp()
        for cookie in self._cookies:
            name, value = cookie.get("name"), cookie.get("value")
            domain, cookie_path = str(cookie.get("domain") or "").lower().lstrip("."), str(cookie.get("path") or "/")
            expires = cookie.get("expires", -1)
            if not isinstance(name, str) or not isinstance(value, str) or not domain:
                continue
            if host != domain and not host.endswith("." + domain):
                continue
            if not path.startswith(cookie_path) or (cookie.get("secure") and parsed.scheme != "https"):
                continue
            if isinstance(expires, (int, float)) and expires > 0 and expires <= now:
                continue
            values.append(f"{name}={value}")
        return "; ".join(values) if values else None

    def netscape_cookiefile(self, *, allowed_domains: frozenset[str]) -> str:
        """Return only valid, in-scope cookies in yt-dlp's Netscape format.

        This is intentionally an in-memory conversion surface.  Callers that
        need a file must use :func:`temporary_netscape_cookiefile`, which
        creates and removes a private worker-local file around one operation.
        """
        allowed = {domain.lower().lstrip(".") for domain in allowed_domains}
        now = datetime.now().timestamp()
        lines = ["# Netscape HTTP Cookie File"]
        for cookie in self._cookies:
            name, value = cookie.get("name"), cookie.get("value")
            domain_value, path = cookie.get("domain"), cookie.get("path")
            secure, http_only = cookie.get("secure"), cookie.get("httpOnly", False)
            if not isinstance(domain_value, str):
                continue
            has_subdomains = domain_value.startswith(".")
            domain = domain_value.lower().lstrip(".")
            if not _cookie_domain_allowed(domain, allowed):
                continue
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or not isinstance(path, str)
                or not path.startswith("/")
                or not isinstance(secure, bool)
                or not isinstance(http_only, bool)
                or any(character in "\t\r\n" for character in (name + value + domain + path))
                or any(character.isspace() for character in domain)
            ):
                raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
            expires = cookie.get("expires", -1)
            if isinstance(expires, bool) or not isinstance(expires, (int, float)) or not isfinite(expires):
                raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
            if expires > 0 and expires <= now:
                continue
            # Netscape format represents Playwright's session-cookie sentinel
            # (-1) as zero.  Do not invent an expiry for an authenticated
            # session, and preserve host-only vs domain-cookie semantics.
            expiry = "0" if expires <= 0 else str(int(expires))
            output_domain = ("#HttpOnly_" if http_only else "") + domain
            lines.append(
                "\t".join((output_domain, "TRUE" if has_subdomains else "FALSE", path,
                           "TRUE" if secure else "FALSE", expiry, name, value))
            )
        if len(lines) == 1:
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        return "\n".join(lines) + "\n"


def _cookie_domain_allowed(domain: str, allowed_domains: set[str]) -> bool:
    return bool(domain) and any(domain == allowed or domain.endswith("." + allowed) for allowed in allowed_domains)


def _run_windows_acl(command: list[str]) -> str:
    """Run one parameterized ACL command without surfacing its output."""
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True)
    except OSError as exc:
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc
    if completed.returncode != 0:
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
    return completed.stdout


def _windows_current_user_sid() -> str:
    """Get the current account SID for an explicit, non-inherited DACL."""
    output = _run_windows_acl(["whoami", "/user", "/fo", "csv", "/nh"])
    # The SID is the only S-1-* value emitted by whoami's one-row CSV output.
    for value in output.replace('"', "").replace(",", " ").split():
        if value.upper().startswith("S-1-"):
            return value
    raise BrowserSessionError("SOURCE_SESSION_EXPIRED")


def _windows_acl_snapshot(path: Path) -> dict[str, object]:
    """Return a locale-independent, SID-only view of the file's DACL.

    ``icacls`` is suitable for changing the DACL but its display names and
    inherited markers are localized.  Use .NET's security objects to emit
    stable SIDs and booleans for verification, without returning command
    output to a caller or log.
    """
    # ``powershell.exe -Command`` appends subsequent argv values to the source
    # text rather than binding them as ``$args``.  Embed a single-quoted,
    # escaped literal path in the non-shell command so a space or apostrophe
    # cannot alter the script's syntax.
    literal_path = os.fspath(path).replace("'", "''")
    script = r"""
$ErrorActionPreference = 'Stop'
$acl = [System.IO.File]::GetAccessControl('{literal_path}')
function Convert-ToSid($identity) {
    return $identity.Translate([System.Security.Principal.SecurityIdentifier]).Value
}
[pscustomobject]@{
    owner = $acl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value
    protected = [bool]$acl.AreAccessRulesProtected
    access = @($acl.Access | ForEach-Object {
        [pscustomobject]@{
            sid = Convert-ToSid $_.IdentityReference
            inherited = [bool]$_.IsInherited
            type = [string]$_.AccessControlType
            rights = [int64]$_.FileSystemRights
        }
    })
} | ConvertTo-Json -Compress -Depth 3
""".replace("{literal_path}", literal_path)
    try:
        snapshot = json.loads(
            _run_windows_acl(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
            )
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc
    if not isinstance(snapshot, dict):
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
    return snapshot


def _validate_windows_private_acl(path: Path) -> None:
    """Require a protected, explicit DACL for only operator and LocalSystem."""
    user_sid = _windows_current_user_sid().upper()
    snapshot = _windows_acl_snapshot(path)
    owner = snapshot.get("owner")
    protected = snapshot.get("protected")
    access = snapshot.get("access")
    if not isinstance(owner, str) or owner.upper() != user_sid or protected is not True or not isinstance(access, list):
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
    allowed = {user_sid, _WINDOWS_LOCAL_SYSTEM_SID}
    user_rights = 0
    system_rights = 0
    for entry in access:
        if not isinstance(entry, dict):
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        sid, inherited, access_type, rights = (
            entry.get("sid"),
            entry.get("inherited"),
            entry.get("type"),
            entry.get("rights"),
        )
        if (
            not isinstance(sid, str)
            or sid.upper() not in allowed
            or inherited is not False
            or access_type != "Allow"
            or isinstance(rights, bool)
            or not isinstance(rights, int)
        ):
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        if sid.upper() == user_sid:
            user_rights |= rights
        else:
            system_rights |= rights
    if not user_rights & _WINDOWS_READ_DATA or system_rights & _WINDOWS_FULL_CONTROL != _WINDOWS_FULL_CONTROL:
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED")


def _restrict_to_current_user(path: Path, *, user_access: str = "R") -> None:
    """Make a file private before any session or cookie bytes are written."""
    if os.name != "nt":
        path.chmod(0o400)
        return
    user_sid = _windows_current_user_sid()
    _run_windows_acl(
        [
            "icacls",
            os.fspath(path),
            "/inheritance:r",
            "/grant:r",
            f"*{user_sid}:({user_access})",
            f"*{_WINDOWS_LOCAL_SYSTEM_SID}:(F)",  # LocalSystem needs service-side secret access.
        ]
    )
    _validate_windows_private_acl(path)


def write_private_file(path: Path, contents: bytes) -> None:
    """Create a new private file and write bytes only after its ACL is private.

    On POSIX this is strict ``0400``.  On Windows inheritance is removed and
    only the current user's SID (read) plus LocalSystem (full control) are
    granted.  This intentionally fails closed when either system ACL command
    is unavailable or reports an error.
    """
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        if os.name != "nt":
            # Keep the exclusive descriptor open while the secret is written.
            # Creating with 0600 means no other account can read it at any
            # point; changing it to 0400 *before* closing prevents the
            # non-root POSIX failure caused by reopening an already-read-only
            # path for writing.
            stream = os.fdopen(descriptor, "wb")
            descriptor = None  # ``stream`` owns and closes this descriptor.
            with stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o400)
                os.fsync(stream.fileno())
            return

        os.close(descriptor)
        descriptor = None
        # The file remains inaccessible to every other user while it is
        # populated; downgrade the creator from full control to read-only
        # before returning it to a caller.
        _restrict_to_current_user(path, user_access="F")
        with path.open("wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        _restrict_to_current_user(path)
    except (BrowserSessionError, OSError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        if isinstance(exc, BrowserSessionError):
            raise
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc


def _private_temporary_file(directory: Path, *, prefix: str, suffix: str) -> Path:
    """Reserve a unique private destination without putting secret bytes in it."""
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(descriptor)
    path = Path(name)
    try:
        _restrict_to_current_user(path, user_access="F")
    except BrowserSessionError:
        path.unlink(missing_ok=True)
        raise
    return path


@contextmanager
def temporary_netscape_cookiefile(storage_state: Path, *, allowed_domains: frozenset[str]):
    """Expose a storage state to yt-dlp for one call, then remove it.

    State bytes and converted cookies never become a materialization, task,
    checkpoint, artifact, error, or log value.
    """
    validate_storage_state(storage_state)
    jar = StorageStateCookieJar(storage_state)
    temporary = tempfile.TemporaryDirectory(prefix="stock-content-cookies-")
    cookiefile = Path(temporary.name) / "cookies.txt"
    try:
        write_private_file(cookiefile, jar.netscape_cookiefile(allowed_domains=allowed_domains).encode("utf-8"))
        yield cookiefile
    finally:
        # TemporaryDirectory cleans the directory even when yt-dlp resolution
        # fails.  Do not expose a deletion error over the stable source code.
        temporary.cleanup()


def validate_storage_state(path: Path) -> None:
    """Validate an operator-provided Playwright state without revealing it."""
    if not path.is_file() or path.is_symlink():
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
    try:
        if path.stat().st_size <= 2 or path.stat().st_size > 8 * 1024 * 1024:
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        if os.name == "nt":
            _validate_windows_private_acl(path)
        elif stat.S_IMODE(path.stat().st_mode) != 0o400:
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        StorageStateCookieJar(path)
    except OSError as exc:
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc


def save_visible_storage_state(
    *,
    page_url: str,
    destination: Path,
    allowed_domains: frozenset[str],
    timeout_seconds: float = 300.0,
    confirmation: Callable[[float], bool] | None = None,
    require_operator_confirmation: bool = True,
) -> None:
    """Open an isolated visible Chromium session, then atomically save its state.

    This deliberately does not access a local Chrome profile.  The caller is
    responsible for telling the user to complete login in the visible window.
    A caller must explicitly confirm that login is complete before any cookie
    is examined, so anonymous pre-login cookies can never trigger a save.
    """
    validate_source_url(page_url, allowed_domains=allowed_domains)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserSessionError("SOURCE_BROWSER_UNAVAILABLE") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    playwright = browser = context = None
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(page_url, wait_until="domcontentloaded", timeout=max(1, int(timeout_seconds * 1000)))
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        allowed = {domain.lower().lstrip(".") for domain in allowed_domains}
        if require_operator_confirmation:
            remaining = deadline - time.monotonic()
            if confirmation is None or remaining <= 0 or not confirmation(remaining):
                raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        if time.monotonic() >= deadline:
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        # Read the returned state in memory, rather than asking the browser
        # process to overwrite a pre-created ACL-protected file.  This avoids
        # a Windows driver write failure and still never exposes its bytes.
        state = context.storage_state()
        cookies = state.get("cookies") if isinstance(state, dict) else None
        if not isinstance(cookies, list) or not any(
            isinstance(cookie.get("name"), str)
            and isinstance(cookie.get("value"), str)
            and _cookie_domain_allowed(str(cookie.get("domain") or "").lower().lstrip("."), allowed)
            for cookie in cookies
        ):
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED")
        try:
            state_bytes = json.dumps(state, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc
        temporary_directory = tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent)
        temporary = Path(temporary_directory.name) / "storage-state.json"
        write_private_file(temporary, state_bytes)
        validate_storage_state(temporary)
        os.replace(temporary, destination)
        temporary = None
    except BrowserSessionError:
        raise
    except Exception as exc:
        raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc
    finally:
        try:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        if context is not None:
            context.close()
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()
        if temporary_directory is not None:
            temporary_directory.cleanup()


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
    duration_seconds: float | None = None


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

        validate_storage_state(storage_state)
        captured: list[CapturedResponse] = []
        subtitles: list[CapturedResponse] = []
        unsafe_route = False
        temporary = tempfile.TemporaryDirectory(prefix="xiaoe-state-")
        copied_state = Path(temporary.name) / "storage-state.json"
        browser = context = playwright = None
        try:
            try:
                write_private_file(copied_state, storage_state.read_bytes())
            except OSError as exc:
                raise BrowserSessionError("SOURCE_SESSION_EXPIRED") from exc
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(storage_state=str(copied_state))
            page = context.new_page()

            def route_request(route) -> None:
                nonlocal unsafe_route
                try:
                    validate_source_url(route.request.url, allowed_domains=self._allowed_domains)
                except UnsafeSourceURL:
                    if _unsafe_route_is_critical(route.request, page):
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
            duration = page.evaluate("""() => {
                const value = document.querySelector('video')?.duration;
                return Number.isFinite(value) && value > 0 ? value : null;
            }""")
            return PageCapture(
                course_id, lesson_id, title, None, None, media, tuple(subtitles), duration_seconds=duration
            )
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


def _unsafe_route_is_critical(request, page) -> bool:
    """Only blocked navigation/media can change the authorization outcome."""
    resource_type = str(getattr(request, "resource_type", "")).lower()
    path = urlsplit(str(getattr(request, "url", ""))).path.lower()
    if resource_type == "media" or path.endswith((".m3u8", ".mpd")):
        return True
    is_navigation = getattr(request, "is_navigation_request", None)
    is_main_frame = getattr(request, "frame", None) == getattr(page, "main_frame", None)
    return bool(is_navigation() if callable(is_navigation) else is_navigation) and is_main_frame


__all__ = [
    "BrowserSessionError", "CapturedResponse", "PageCapture", "PlaywrightSession", "StorageStateCookieJar",
    "save_visible_storage_state", "temporary_netscape_cookiefile", "validate_storage_state",
]

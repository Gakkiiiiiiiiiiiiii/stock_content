from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest

import stock_content.adapters.browser.playwright_session as session
from stock_content.adapters.browser.playwright_session import (
    BrowserSessionError,
    PlaywrightSession,
    StorageStateCookieJar,
    save_visible_storage_state,
    temporary_netscape_cookiefile,
    validate_storage_state,
    write_private_file,
)
from stock_content.adapters.sources.xiaoe_page import XiaoeResolutionError, _page_url_from_template
from stock_content.cli import capture_storage_state


def _state(path: Path) -> Path:
    write_private_file(
        path,
        (
            b'{"cookies":[{"name":"session","value":"canary-secret","domain":"example.test",'
            b'"path":"/course","secure":true,"expires":4102444800}]}'
        ),
    )
    return path


def test_cookie_jar_scopes_cookie_and_redacts_values(tmp_path: Path) -> None:
    jar = StorageStateCookieJar(_state(tmp_path / "state.json"))
    assert jar.header_for("https://cdn.example.test/course/video") == "session=canary-secret"
    assert jar.header_for("https://cdn.example.test/other") is None
    assert jar.header_for("http://cdn.example.test/course/video") is None
    assert "canary-secret" not in repr(jar)


def test_storage_state_rejects_non_json_or_symlink(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(BrowserSessionError) as error:
        validate_storage_state(invalid)
    assert error.value.code == "SOURCE_SESSION_EXPIRED"


def test_temporary_cookiefile_is_private_and_removed(tmp_path: Path) -> None:
    cookiefile = temporary_netscape_cookiefile(
        _state(tmp_path / "state.json"), allowed_domains=frozenset({"example.test"})
    )
    with cookiefile as jar:
        assert jar.exists()
        assert "canary-secret" in jar.read_text(encoding="utf-8")
        if os.name != "nt":
            assert jar.stat().st_mode & 0o777 == 0o400
    assert not jar.exists()


def test_posix_private_write_keeps_exclusive_descriptor_until_final_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for non-root workers: do not reopen a 0400 path to write it."""
    target = tmp_path / "state.json"
    events: list[tuple[str, int | None]] = []
    running_on_posix = os.name != "nt"
    original_open = session.os.open
    original_fdopen = session.os.fdopen
    original_fchmod = getattr(session.os, "fchmod", None)

    def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
        events.append(("open", mode))
        return original_open(path, flags, mode)

    def tracked_fdopen(descriptor, mode="r", *args, **kwargs):
        events.append(("fdopen", descriptor))
        return original_fdopen(descriptor, mode, *args, **kwargs)

    def tracked_fchmod(descriptor, mode):
        events.append(("fchmod", mode))
        if running_on_posix and original_fchmod is not None:
            original_fchmod(descriptor, mode)

    monkeypatch.setattr(session.os, "name", "posix")
    monkeypatch.setattr(session.os, "open", tracked_open)
    monkeypatch.setattr(session.os, "fdopen", tracked_fdopen)
    monkeypatch.setattr(session.os, "fchmod", tracked_fchmod, raising=False)

    write_private_file(target, b"canary-secret")

    assert target.read_bytes() == b"canary-secret"
    assert events[0] == ("open", 0o600)
    assert events[1][0] == "fdopen"
    assert events[-1] == ("fchmod", 0o400)
    if running_on_posix:
        assert target.stat().st_mode & 0o777 == 0o400


def test_posix_private_write_removes_partial_file_after_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "state.json"
    monkeypatch.setattr(session.os, "name", "posix")
    monkeypatch.setattr(session.os, "fchmod", lambda _descriptor, _mode: None, raising=False)
    monkeypatch.setattr(session.os, "fsync", lambda _descriptor: (_ for _ in ()).throw(OSError("disk failure")))

    with pytest.raises(BrowserSessionError, match="SOURCE_SESSION_EXPIRED"):
        write_private_file(target, b"canary-secret")

    assert not target.exists()


def _windows_acl_snapshot(*, access: list[dict[str, object]], protected: bool = True) -> str:
    return json.dumps({"owner": "S-1-5-21-1000", "protected": protected, "access": access})


def _private_windows_access() -> list[dict[str, object]]:
    return [
        {"sid": "S-1-5-21-1000", "inherited": False, "type": "Allow", "rights": 0x1},
        {"sid": "S-1-5-18", "inherited": False, "type": "Allow", "rights": 0x1F01FF},
    ]


def test_windows_acl_helper_requires_only_explicit_operator_and_system(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "whoami":
            return subprocess.CompletedProcess(command, 0, '"HOST\\operator","S-1-5-21-1000"\n', "")
        if command[0] == "powershell":
            return subprocess.CompletedProcess(command, 0, _windows_acl_snapshot(access=_private_windows_access()), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(session.os, "name", "nt")
    monkeypatch.setattr(session.subprocess, "run", run)
    target = tmp_path / "state.json"
    write_private_file(
        target,
        b'{"cookies":[{"name":"session","value":"canary","domain":"example.test"}]}',
    )
    assert target.is_file()
    grants = [command for command in commands if command[0] == "icacls"]
    assert grants[-1][2:] == ["/inheritance:r", "/grant:r", "*S-1-5-21-1000:(R)", "*S-1-5-18:(F)"]
    validate_storage_state(target)


@pytest.mark.parametrize(
    "access",
    [
        [
            {"sid": "S-1-5-21-1000", "inherited": True, "type": "Allow", "rights": 0x1},
            {"sid": "S-1-5-18", "inherited": False, "type": "Allow", "rights": 0x1F01FF},
        ],
        [
            {"sid": "S-1-5-21-1000", "inherited": False, "type": "Allow", "rights": 0x1},
            {"sid": "S-1-5-18", "inherited": False, "type": "Allow", "rights": 0x1F01FF},
            {"sid": "S-1-1-0", "inherited": False, "type": "Allow", "rights": 0x1},
        ],
    ],
    ids=["inherited-ace", "unexpected-principal"],
)
def test_windows_acl_validation_rejects_inherited_or_unexpected_ace(monkeypatch, tmp_path: Path, access) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"cookies":[]}', encoding="utf-8")

    def run(command, **_kwargs):
        if command[0] == "whoami":
            return subprocess.CompletedProcess(command, 0, '"HOST\\operator","S-1-5-21-1000"\n', "")
        if command[0] == "powershell":
            return subprocess.CompletedProcess(command, 0, _windows_acl_snapshot(access=access), "")
        raise AssertionError(command)

    monkeypatch.setattr(session.os, "name", "nt")
    monkeypatch.setattr(session.subprocess, "run", run)
    with pytest.raises(BrowserSessionError, match="SOURCE_SESSION_EXPIRED"):
        validate_storage_state(target)


def _install_fake_playwright(monkeypatch, *, cookies):
    class Page:
        def goto(self, *_args, **_kwargs):
            return None

        def wait_for_timeout(self, _milliseconds):
            return None

    class Context:
        def __init__(self):
            self.closed = False

        def new_page(self):
            return Page()

        def cookies(self):
            return cookies

        def storage_state(self, *, path=None):
            value = {
                "cookies": [
                    {
                        "name": "session",
                        "value": "canary-secret",
                        "domain": "example.test",
                        "path": "/",
                        "secure": True,
                        "expires": 4102444800,
                    }
                ]
            }
            if path is not None:
                Path(path).write_text(__import__("json").dumps(value), encoding="utf-8")
            return value

        def close(self):
            self.closed = True

    class Browser:
        def new_context(self):
            return Context()

        def close(self):
            return None

    class Playwright:
        chromium = SimpleNamespace(launch=lambda **_kwargs: Browser())

        def start(self):
            return self

        def stop(self):
            return None

    package = ModuleType("playwright")
    sync_api = ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: Playwright()
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)


def test_visible_save_atomically_replaces_existing_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "validate_source_url", lambda *_args, **_kwargs: None)
    _install_fake_playwright(
        monkeypatch,
        cookies=[{"name": "session", "value": "canary-secret", "domain": "example.test"}],
    )
    destination = tmp_path / "state.json"
    _state(destination)
    save_visible_storage_state(
        page_url="https://example.test/login",
        destination=destination,
        allowed_domains=frozenset({"example.test"}),
        timeout_seconds=1,
        confirmation=lambda _remaining: True,
    )
    validate_storage_state(destination)
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_anonymous_cookie_does_not_save_before_operator_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "validate_source_url", lambda *_args, **_kwargs: None)
    _install_fake_playwright(monkeypatch, cookies=[{"name": "anony_token", "value": "x", "domain": "example.test"}])
    destination = tmp_path / "state.json"
    with pytest.raises(BrowserSessionError, match="SOURCE_SESSION_EXPIRED"):
        save_visible_storage_state(
            page_url="https://example.test/login",
            destination=destination,
            allowed_domains=frozenset({"example.test"}),
            timeout_seconds=1,
            confirmation=lambda _remaining: False,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_capture_cli_accepts_only_explicit_bounded_inputs(monkeypatch, tmp_path: Path, capsys) -> None:
    call = {}
    monkeypatch.setattr(capture_storage_state, "save_visible_storage_state", lambda **kwargs: call.update(kwargs))
    assert capture_storage_state.main(
        [
            "--page-url", "https://www.example.test/login", "--allowed-domain", "example.test",
            "--destination", str(tmp_path / "private.json"), "--timeout-seconds", "60",
        ]
    ) == 0
    assert call["allowed_domains"] == frozenset({"example.test"}) and capsys.readouterr().out == "STORAGE_STATE_SAVED\n"
    assert call["require_operator_confirmation"] is True
    with pytest.raises(SystemExit):
        capture_storage_state.main(
            ["--page-url", "https://example.test", "--allowed-domain", "*.example.test", "--destination", "x"]
        )


def test_cli_confirmation_accepts_enter_and_times_out(capsys) -> None:
    assert capture_storage_state._wait_for_operator_confirmation(0.1, readline=lambda: "\n")
    assert capsys.readouterr().err == "READY_FOR_LOGIN\n"
    blocked = Event()
    assert not capture_storage_state._wait_for_operator_confirmation(0.01, readline=lambda: blocked.wait())
    assert capsys.readouterr().err == "READY_FOR_LOGIN\n"


def _install_capture_playwright(monkeypatch, route_specs) -> None:
    class Request:
        def __init__(self, url, resource_type, navigation, frame):
            self.url, self.resource_type, self._navigation, self.frame = url, resource_type, navigation, frame

        def is_navigation_request(self):
            return self._navigation

    class Route:
        def __init__(self, request):
            self.request, self.aborted, self.continued = request, False, False

        def abort(self):
            self.aborted = True

        def continue_(self):
            self.continued = True

    class Response:
        url = "https://cdn.example.test/live.m3u8"
        headers = {"content-type": "application/vnd.apple.mpegurl"}
        request = SimpleNamespace(headers={})

    class Page:
        def __init__(self, context):
            self.context, self.main_frame, self.handlers = context, object(), {}

        def on(self, event, callback):
            self.handlers[event] = callback

        def goto(self, *_args, **_kwargs):
            for url, resource_type, navigation, main in route_specs:
                route = Route(Request(url, resource_type, navigation, self.main_frame if main else object()))
                self.context.route_callback(route)
                if url.endswith("live.m3u8") and route.continued:
                    self.handlers["response"](Response())

        def title(self):
            return "Authorized"

        def evaluate(self, _script):
            return None

    class Context:
        def route(self, _pattern, callback):
            self.route_callback = callback

        def new_page(self):
            return Page(self)

        def close(self):
            return None

    class Browser:
        def new_context(self, **_kwargs):
            return Context()

        def close(self):
            return None

    class Playwright:
        chromium = SimpleNamespace(launch=lambda **_kwargs: Browser())

        def start(self):
            return self

        def stop(self):
            return None

    package, sync_api = ModuleType("playwright"), ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: Playwright()
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)


def test_third_party_analytics_is_aborted_without_blocking_allowed_media(monkeypatch, tmp_path: Path) -> None:
    def validate(url, **_kwargs):
        if "example.test" not in url:
            raise session.UnsafeSourceURL("blocked")

    monkeypatch.setattr(session, "validate_source_url", validate)
    _install_capture_playwright(
        monkeypatch,
        [
            ("https://analytics.invalid/pixel.js", "script", False, False),
            ("https://cdn.example.test/live.m3u8", "media", False, False),
        ],
    )
    capture = PlaywrightSession(
        allowed_domains=frozenset({"example.test"}), page_url_for=lambda _identity: "https://www.example.test/course"
    ).capture("course/lesson", _state(tmp_path / "state.json"))
    assert capture.media is not None and capture.media.kind == "hls"


@pytest.mark.parametrize(
    "spec",
    [
        ("https://attacker.invalid/next", "document", True, True),
        ("https://attacker.invalid/stream.m3u8", "media", False, False),
    ],
)
def test_unsafe_main_navigation_or_media_fails_closed(monkeypatch, tmp_path: Path, spec) -> None:
    def validate(url, **_kwargs):
        if "example.test" not in url:
            raise session.UnsafeSourceURL("blocked")

    monkeypatch.setattr(session, "validate_source_url", validate)
    _install_capture_playwright(monkeypatch, [spec])
    browser = PlaywrightSession(
        allowed_domains=frozenset({"example.test"}), page_url_for=lambda _identity: "https://www.example.test/course"
    )
    with pytest.raises(BrowserSessionError, match="SOURCE_DOMAIN_NOT_ALLOWLISTED"):
        browser.capture("course/lesson", _state(tmp_path / "state.json"))


def test_xiaoe_template_binds_only_safe_course_and_lesson_ids() -> None:
    url = _page_url_from_template(
        "https://tenant.h5.xiaoeknow.com/p/course/video/{lesson_id}?product_id={course_id}", "p_123/v_456"
    )
    assert url == "https://tenant.h5.xiaoeknow.com/p/course/video/v_456?product_id=p_123"
    with pytest.raises(XiaoeResolutionError):
        _page_url_from_template("https://tenant.example/{source_ref}", "p_123/../../private")

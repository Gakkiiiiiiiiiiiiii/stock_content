from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from urllib.request import Request

import pytest

from stock_content.adapters.sources import bilibili as bilibili_module
from stock_content.adapters.sources import security
from stock_content.adapters.sources import xiaoe as xiaoe_module
from stock_content.adapters.sources.bilibili import BilibiliSourceAdapter
from stock_content.adapters.sources.security import UnsafeSourceURL
from stock_content.adapters.sources.xiaoe import XiaoeHlsSourceAdapter


def _dns(address_by_host: dict[str, str]):
    def getaddrinfo(host: str, port: int, **_: object):
        address = address_by_host[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    return getaddrinfo


def test_initial_malicious_url_is_rejected_before_external_downloader(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def run(_: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        raise AssertionError("downloader must not receive an unsafe URL")

    monkeypatch.setattr(BilibiliSourceAdapter, "_run", staticmethod(run))
    with pytest.raises(UnsafeSourceURL) as error:
        BilibiliSourceAdapter().resolve("https://attacker.example/video/BV1")

    assert error.value.code == "SOURCE_DOMAIN_NOT_ALLOWLISTED"
    assert not called


def test_xiaoe_initial_malicious_url_is_rejected_before_ffmpeg() -> None:
    with pytest.raises(UnsafeSourceURL) as error:
        XiaoeHlsSourceAdapter().resolve("https://attacker.example/live.m3u8")

    assert error.value.code == "SOURCE_DOMAIN_NOT_ALLOWLISTED"


def test_allowlisted_domain_resolving_to_private_address_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        _dns({"www.bilibili.com": "192.168.10.10"}),
    )

    with pytest.raises(UnsafeSourceURL) as error:
        security.validate_source_url("https://www.bilibili.com/video/BV1")

    assert error.value.code == "SOURCE_PRIVATE_ADDRESS"


@pytest.mark.parametrize("target", ["https://attacker.example/steal", "https://cdn.bilibili.com/steal"])
def test_redirect_target_is_validated_before_following(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    addresses = {
        "www.bilibili.com": "93.184.216.34",
        "cdn.bilibili.com": "10.0.0.8",
    }
    monkeypatch.setattr(security.socket, "getaddrinfo", _dns(addresses))
    handler = security._ValidatingRedirectHandler(
        allowed_domains=security.DEFAULT_ALLOWED_DOMAINS, resolve_host=True
    )

    with pytest.raises(UnsafeSourceURL) as error:
        handler.redirect_request(
            Request("https://www.bilibili.com/video/BV1"),
            None,
            302,
            "Found",
            {},
            target,
        )

    assert error.value.code == "SOURCE_REDIRECT_UNSAFE"


def test_allowlisted_redirect_chain_is_checked_without_public_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        _dns({"www.bilibili.com": "93.184.216.34", "b23.tv": "93.184.216.35"}),
    )
    handler = security._ValidatingRedirectHandler(
        allowed_domains=security.DEFAULT_ALLOWED_DOMAINS, resolve_host=True
    )
    first = Request("https://www.bilibili.com/video/BV1")
    second = handler.redirect_request(first, None, 302, "Found", {}, "https://b23.tv/BV1")
    assert second.full_url == "https://b23.tv/BV1"
    third = handler.redirect_request(second, None, 302, "Found", {}, "/final")
    assert third.full_url == "https://b23.tv/final"


def test_preflight_uses_checked_final_url_and_never_reads_public_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        _dns({"www.bilibili.com": "93.184.216.34"}),
    )

    class Response:
        def geturl(self) -> str:
            return "https://www.bilibili.com/video/BV1"

        def close(self) -> None:
            return None

    class FakeOpener:
        def open(self, request: Request, *, timeout: float):
            assert request.full_url == "https://www.bilibili.com/video/BV1"
            assert timeout == 2
            return Response()

    checked = security.preflight_source_url(
        "https://www.bilibili.com/video/BV1", opener=FakeOpener(), timeout=2
    )
    assert checked == "https://www.bilibili.com/video/BV1"


def test_adapters_preflight_before_their_external_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    urls: list[str] = []
    monkeypatch.setattr(bilibili_module, "preflight_source_url", lambda url: urls.append(url) or url)
    monkeypatch.setattr(xiaoe_module, "preflight_source_url", lambda url: urls.append(url) or url)

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fixture")
    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if "--dump-single-json" in arguments:
            return subprocess.CompletedProcess(
                [], 0, stdout='{"id":"BV1","url":"https://cdn.bilivideo.com/video.mp4"}', stderr=""
            )
        return subprocess.CompletedProcess([], 0, stdout=f"{media}\n", stderr="")

    monkeypatch.setattr(BilibiliSourceAdapter, "_run", staticmethod(run))
    downloaded: list[str] = []

    def safe_download(url: str, target: Path, **_: object) -> str:
        downloaded.append(url)
        target.write_bytes(media.read_bytes())
        return url

    monkeypatch.setattr(bilibili_module, "safe_download_url", safe_download)
    monkeypatch.setattr(bilibili_module, "validate_source_url", lambda url, **_: url)
    local_playlist = tmp_path / "safe.m3u8"
    local_playlist.write_text("#EXTM3U\n", encoding="utf-8")
    monkeypatch.setattr(xiaoe_module, "download_hls_playlist", lambda *_args, **_kwargs: str(local_playlist))
    monkeypatch.setattr(
        xiaoe_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    )

    BilibiliSourceAdapter().resolve("BV1")
    BilibiliSourceAdapter().download("BV1", tmp_path)
    XiaoeHlsSourceAdapter().resolve("https://m.xiaoe-tech.com/live.m3u8")
    XiaoeHlsSourceAdapter().download("https://m.xiaoe-tech.com/live.m3u8", tmp_path)

    assert urls == [
        "https://www.bilibili.com/video/BV1",
        "https://www.bilibili.com/video/BV1",
        "https://m.xiaoe-tech.com/live.m3u8",
        "https://m.xiaoe-tech.com/live.m3u8",
    ]
    assert downloaded == ["https://cdn.bilivideo.com/video.mp4"]


def test_bilibili_discovered_private_media_url_is_rejected_before_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(bilibili_module, "preflight_source_url", lambda url: url)

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        return subprocess.CompletedProcess(
            [], 0, stdout='{"id":"BV1","url":"https://attacker.example/video.mp4"}', stderr=""
        )

    monkeypatch.setattr(BilibiliSourceAdapter, "_run", staticmethod(run))
    with pytest.raises(UnsafeSourceURL) as error:
        BilibiliSourceAdapter().download("BV1", tmp_path)
    assert error.value.code == "SOURCE_DOMAIN_NOT_ALLOWLISTED"
    assert all("attacker.example" not in argument for command in commands for argument in command)


def test_bilibili_separate_streams_are_downloaded_with_headers_then_merged_locally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bilibili_module, "preflight_source_url", lambda url: url)
    metadata = {
        "requested_formats": [
            {
                "url": "https://cdn.bilivideo.com/video.m4s",
                "vcodec": "avc1",
                "acodec": "none",
                "http_headers": {"Referer": "https://www.bilibili.com/video/BV1", "Cookie": "ignored"},
            },
            {
                "url": "https://cdn.bilivideo.com/audio.m4s",
                "vcodec": "none",
                "acodec": "mp4a",
                "http_headers": {"User-Agent": "fixture-agent"},
            },
        ]
    }
    monkeypatch.setattr(
        BilibiliSourceAdapter,
        "_run",
        staticmethod(lambda _: subprocess.CompletedProcess([], 0, stdout=json.dumps(metadata), stderr="")),
    )
    downloads: list[tuple[str, dict[str, str]]] = []

    def safe_download(url: str, target: Path, *, headers: dict[str, str], **_: object) -> str:
        downloads.append((url, headers))
        target.write_bytes(b"stream")
        return url

    monkeypatch.setattr(bilibili_module, "safe_download_url", safe_download)
    monkeypatch.setattr(bilibili_module, "validate_source_url", lambda url, **_: url)
    ffmpeg_args: list[str] = []

    def merge(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        ffmpeg_args.extend(args)
        Path(args[-1]).write_bytes(b"merged")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(bilibili_module.subprocess, "run", merge)
    result = BilibiliSourceAdapter().download("https://www.bilibili.com/video/BV1", tmp_path)
    assert result.read_bytes() == b"merged"
    assert downloads == [
        ("https://cdn.bilivideo.com/video.m4s", {"Referer": "https://www.bilibili.com/video/BV1"}),
        ("https://cdn.bilivideo.com/audio.m4s", {"User-Agent": "fixture-agent"}),
    ]
    assert not any("http://" in arg or "https://" in arg for arg in ffmpeg_args)


class _FakeResponse:
    def __init__(self, status: int, body: bytes = b"", location: str | None = None) -> None:
        self.status = status
        self._body = body
        self._location = location

    def getheader(self, name: str) -> str | None:
        return self._location if name.lower() == "location" else None

    def read(self, _: int = -1) -> bytes:
        body, self._body = self._body, b""
        return body

    def close(self) -> None:
        return None


def test_safe_get_validates_redirect_before_connecting_to_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeConnection:
        def __init__(self, host: str, port: int, **_: object) -> None:
            self.host = host

        def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
            calls.append((self.host, path))

        def getresponse(self) -> _FakeResponse:
            return _FakeResponse(302, location="https://cdn.bilivideo.com/private")

        def close(self) -> None:
            return None

    monkeypatch.setattr(security, "_PinnedHTTPSConnection", FakeConnection)
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        _dns({"www.bilibili.com": "93.184.216.34", "cdn.bilivideo.com": "10.0.0.8"}),
    )
    with pytest.raises(UnsafeSourceURL) as error:
        security.safe_download_url(
            "https://www.bilibili.com/video/BV1", tmp_path / "video.bin"
        )
    assert error.value.code == "SOURCE_PRIVATE_ADDRESS"
    assert calls == [("www.bilibili.com", "/video/BV1")]


def test_safe_get_allowlisted_redirect_writes_bytes_without_external_downloader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    responses = {
        ("www.bilibili.com", "/video/BV1"): _FakeResponse(
            302, location="https://cdn.bilivideo.com/video.mp4"
        ),
        ("cdn.bilivideo.com", "/video.mp4"): _FakeResponse(200, body=b"media"),
    }

    class FakeConnection:
        def __init__(self, host: str, port: int, **_: object) -> None:
            self.host = host
            self.path = ""

        def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
            self.path = path

        def getresponse(self) -> _FakeResponse:
            return responses[(self.host, self.path)]

        def close(self) -> None:
            return None

    monkeypatch.setattr(security, "_PinnedHTTPSConnection", FakeConnection)
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        _dns({"www.bilibili.com": "93.184.216.34", "cdn.bilivideo.com": "93.184.216.35"}),
    )
    output = tmp_path / "video.bin"
    final = security.safe_download_url("https://www.bilibili.com/video/BV1", output)
    assert final == "https://cdn.bilivideo.com/video.mp4"
    assert output.read_bytes() == b"media"


def test_hls_materializer_fetches_manifest_segments_and_key_locally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bodies = {
        "/live.m3u8": (
            b"#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"/key.bin\"\n"
            b"#EXTINF:1,\n/segment.ts\n"
        ),
        "/key.bin": b"key",
        "/segment.ts": b"segment",
    }

    class FakeConnection:
        def __init__(self, host: str, port: int, **_: object) -> None:
            self.host = host
            self.path = ""

        def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
            self.path = path

        def getresponse(self) -> _FakeResponse:
            return _FakeResponse(200, body=bodies[self.path])

        def close(self) -> None:
            return None

    monkeypatch.setattr(security, "_PinnedHTTPSConnection", FakeConnection)
    monkeypatch.setattr(
        security.socket, "getaddrinfo", _dns({"m.xiaoe-tech.com": "93.184.216.34"})
    )
    playlist = Path(
        security.download_hls_playlist("https://m.xiaoe-tech.com/live.m3u8", tmp_path)
    )
    text = playlist.read_text(encoding="utf-8")
    assert "https://" not in text
    assert "key.bin" not in text
    assert "segment.ts" not in text
    assert "#EXTM3U" in text
    assert list((tmp_path / ".safe-hls").glob("*.bin"))


def test_hls_segment_private_redirect_fails_closed_before_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeConnection:
        def __init__(self, host: str, port: int, **_: object) -> None:
            self.host = host
            self.path = ""

        def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
            self.path = path

        def getresponse(self) -> _FakeResponse:
            if self.path == "/live.m3u8":
                return _FakeResponse(200, body=b"#EXTM3U\n#EXTINF:1,\n/segment.ts\n")
            return _FakeResponse(302, location="https://m.xiaoe-tech.com/private.ts")

        def close(self) -> None:
            return None

    monkeypatch.setattr(security, "_PinnedHTTPSConnection", FakeConnection)
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        _dns({"m.xiaoe-tech.com": "93.184.216.34"}),
    )
    with pytest.raises(UnsafeSourceURL) as error:
        security.download_hls_playlist("https://m.xiaoe-tech.com/live.m3u8", tmp_path)
    assert error.value.code == "SOURCE_REDIRECT_UNSAFE"

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

import stock_content.adapters.sources.xiaoe_materializer as materializer_module
from stock_content.adapters.browser.playwright_session import CapturedResponse, PageCapture
from stock_content.adapters.credentials.file_secret_provider import FileSecretProvider
from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.models import SourceArtifactMetadataRow
from stock_content.adapters.postgres.repositories.artifact_repository import SqlArtifactRepository
from stock_content.adapters.sources.bilibili_materializer import BilibiliMaterializer, MaterializedMedia
from stock_content.adapters.sources.dash_materializer import DashLocalizer, DashMaterializationError
from stock_content.adapters.sources.security import HlsResourceLimitError, SourceDownloadHTTPError, UnsafeSourceURL
from stock_content.adapters.sources.xiaoe import XiaoeHlsSourceAdapter
from stock_content.adapters.sources.xiaoe_materializer import XiaoeMaterializationError, XiaoeMaterializer
from stock_content.adapters.sources.xiaoe_page import XiaoeHlsResolver, XiaoePageResolver, XiaoeResolutionError
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import DownloadStage, ResolveSourceStage
from stock_content.domain.source_materialization import MediaStream, ResolvedSource, SourceMaterialization


@pytest.fixture(autouse=True)
def _allow_test_cdn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTENT_XIAOE_ALLOWED_DOMAINS", "example.test")
    monkeypatch.setattr(
        BilibiliMaterializer,
        "_ffprobe",
        staticmethod(
            lambda _path: {
                "format": {"duration": "1.0", "bit_rate": "1000"},
                "streams": [
                    {"codec_type": "video", "codec_name": "h264"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
            }
        ),
    )


class _Browser:
    def __init__(self, capture: PageCapture) -> None:
        self.capture_value = capture
        self.calls: list[tuple[str, Path]] = []

    def capture(self, source_identity: str, storage_state: Path) -> PageCapture:
        self.calls.append((source_identity, storage_state))
        return self.capture_value


def _provider(tmp_path: Path) -> tuple[FileSecretProvider, str]:
    state = tmp_path / "state.json"
    state.write_text('{"cookies":["secret-cookie-canary"]}', encoding="utf-8")
    if os.name != "nt":
        state.chmod(0o400)
    reference = "authorized-xiaoe-session"
    return FileSecretProvider({reference: state}), hashlib.sha256(reference.encode()).hexdigest()


def _capture(url: str = "https://cdn.example.test/live.m3u8?sig=secret-url-canary") -> PageCapture:
    return PageCapture(
        "course-1",
        "lesson-2",
        "Authorized lesson",
        "Author",
        None,
        CapturedResponse(SecretStr(url), "hls", {"Referer": SecretStr("https://m.xiaoe-tech.com/course-1")}),
    )


def test_page_resolution_uses_hash_selected_state_and_keeps_all_locators_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import stock_content.adapters.sources.xiaoe_page as page_module

    monkeypatch.setattr(page_module, "validate_source_url", lambda value, **_: value)
    provider, reference_hash = _provider(tmp_path)
    browser = _Browser(_capture())
    resolver = XiaoePageResolver(
        credential_provider=provider, browser=browser, allowed_domains=frozenset({"example.test", "xiaoe-tech.com"})
    )
    materialization = resolver.resolve("course-1/lesson-2", credential_ref_hash=reference_hash)
    assert browser.calls == [("course-1/lesson-2", (tmp_path / "state.json").resolve())]
    assert materialization.public.canonical_source_ref == "course-1/lesson-2"
    assert materialization.requires_reresolve and materialization.credential_ref_hash == reference_hash
    projection = materialization.model_dump_json()
    assert "secret-url-canary" not in projection
    assert "secret-cookie-canary" not in projection
    assert "secret-url-canary" not in repr(materialization)


def test_page_resolution_projects_the_safe_canonical_course_page(monkeypatch, tmp_path: Path) -> None:
    """The queued course/lesson identity is not itself Bundle provenance."""
    import stock_content.adapters.sources.xiaoe_page as page_module

    monkeypatch.setattr(page_module, "validate_source_url", lambda value, **_: value)
    provider, reference_hash = _provider(tmp_path)
    resolver = XiaoePageResolver(
        credential_provider=provider,
        browser=_Browser(_capture()),
        allowed_domains=frozenset({"example.test", "xiaoe-tech.com"}),
        public_url_for=lambda _: (
            "https://tenant.h5.xiaoeknow.com/p/course/video/lesson-2"
            "?product_id=course-1&tracking=must-not-persist"
        ),
    )
    materialization = resolver.resolve("course-1/lesson-2", credential_ref_hash=reference_hash)

    assert materialization.public.canonical_url == (
        "https://tenant.h5.xiaoeknow.com/p/course/video/lesson-2?product_id=course-1"
    )
    assert "tracking" not in materialization.model_dump_json()


def test_resolve_download_and_sql_provenance_keep_xiaoe_public_page_only(monkeypatch, tmp_path: Path) -> None:
    """Exercise the real resolver → SourceArtifact → SQL metadata boundary."""
    import stock_content.adapters.sources.xiaoe_page as page_module

    monkeypatch.setattr(page_module, "validate_source_url", lambda value, **_: value)
    provider, reference_hash = _provider(tmp_path)
    resolver = XiaoePageResolver(
        credential_provider=provider,
        browser=_Browser(_capture()),
        allowed_domains=frozenset({"example.test", "xiaoe-tech.com"}),
        public_url_for=lambda _: (
            "https://tenant.h5.xiaoeknow.com/p/course/video/lesson-2"
            "?product_id=course-1&signature=must-not-persist"
        ),
    )

    class Adapter:
        def resolve_materialization(self, source_ref: str, **_: object) -> SourceMaterialization:
            return resolver.resolve(source_ref, credential_ref_hash=reference_hash)

        def materialize(self, _: SourceMaterialization, directory: Path, **__: object) -> MaterializedMedia:
            media = directory / "source.mp4"
            media.write_bytes(b"safe-media")
            return MaterializedMedia(media, hashlib.sha256(b"safe-media").hexdigest(), 1.0, {})

    context = PipelineContext(
        "xiaoe-safe-page",
        source={"type": "xiaoe", "ref": "course-1/lesson-2"},
        options={
            "credential_ref_hash": reference_hash,
            "source_artifact_metadata_required": True,
            "source_policy_version": "source-policy.v1",
            "retention_class": "standard",
            "access_classification": "RESTRICTED",
            "raw_storage_dir": str(tmp_path / "raw"),
        },
    )
    adapter = Adapter()
    ResolveSourceStage({"xiaoe": adapter}).execute(context)
    DownloadStage({"xiaoe": adapter}, work_root=tmp_path).execute(context)
    source = context.artifacts.source
    assert source is not None
    expected_url = "https://tenant.h5.xiaoeknow.com/p/course/video/lesson-2?product_id=course-1"
    assert source.source_metadata["canonical_url"] == expected_url
    assert "signature" not in repr(source)

    database = Database(f"sqlite:///{tmp_path / 'provenance.db'}")
    database.create_schema()
    SqlArtifactRepository(database.session_factory).put(source)
    with database.session_factory() as session:
        row = session.get(SourceArtifactMetadataRow, source.artifact_id)
    assert row is not None and row.canonical_url == expected_url
    assert "signature" not in repr(row)


def test_signed_hls_secret_reference_is_resolved_at_worker_and_can_reresolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import stock_content.adapters.sources.xiaoe_page as page_module

    monkeypatch.setattr(page_module, "validate_source_url", lambda value, **_: value)
    locator = tmp_path / "authorized-hls.txt"
    locator.write_text("https://cdn.example.test/live.m3u8?signature=first-canary", encoding="utf-8")
    if os.name != "nt":
        locator.chmod(0o400)
    reference = "authorized-xiaoe-hls"
    reference_hash = hashlib.sha256(reference.encode()).hexdigest()
    adapter = XiaoeHlsSourceAdapter(credential_provider=FileSecretProvider({reference: locator}))
    public_ref = "https://cdn.example.test/live.m3u8"
    first = adapter.resolve_materialization(public_ref, credential_ref_hash=reference_hash)
    assert first.requires_reresolve and first.credential_ref_hash == reference_hash
    assert "first-canary" not in first.model_dump_json()
    if os.name != "nt":
        locator.chmod(0o600)
    locator.write_text("https://cdn.example.test/live.m3u8?signature=second-canary", encoding="utf-8")
    if os.name != "nt":
        locator.chmod(0o400)
    second = adapter.resolve_materialization(public_ref, credential_ref_hash=reference_hash)
    assert "second-canary" not in second.model_dump_json()
    assert second.requires_reresolve


def test_page_resolution_rejects_unapproved_capture_and_missing_media(tmp_path: Path) -> None:
    provider, reference_hash = _provider(tmp_path)
    resolver = XiaoePageResolver(
        credential_provider=provider,
        browser=_Browser(_capture("https://attacker.example/live.m3u8")),
        allowed_domains=frozenset({"xiaoe-tech.com"}),
    )
    with pytest.raises(XiaoeResolutionError) as error:
        resolver.resolve("course-1/lesson-2", credential_ref_hash=reference_hash)
    assert error.value.code == "SOURCE_DOMAIN_NOT_ALLOWLISTED"
    missing = XiaoePageResolver(
        credential_provider=provider,
        browser=_Browser(PageCapture("course-1", "lesson-2", "title", None, None, None)),
        allowed_domains=frozenset({"xiaoe-tech.com"}),
    )
    with pytest.raises(XiaoeResolutionError) as error:
        missing.resolve("course-1/lesson-2", credential_ref_hash=reference_hash)
    assert error.value.code == "SOURCE_MEDIA_NOT_FOUND"


def _hls_materialization(*, url: str = "https://cdn.example.test/live.m3u8?signature=canary", expires_at=None):
    return SourceMaterialization(
        public=ResolvedSource(
            source_type="xiaoe_hls",
            canonical_source_ref="https://cdn.example.test/live.m3u8",
            source_identity_hash="source",
            platform_id="lesson",
            title="lesson",
        ),
        streams=[MediaStream(stream_id="hls", kind="hls", url=SecretStr(url), expires_at=expires_at)],
        subtitles=[],
    )


def _playlist_downloader(url: str, target: Path, *, headers, manifest_validator, **_: object) -> str:
    assert "signature=canary" in url or "fresh" in url
    assert headers == {}
    manifest_validator("#EXTM3U\n#EXT-X-KEY:METHOD=NONE\n#EXTINF:1,\nlocal.ts\n")
    playlist = target / "local.m3u8"
    playlist.write_text("#EXTM3U\n", encoding="utf-8")
    return str(playlist)


def test_direct_hls_uses_same_secret_materialization_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import stock_content.adapters.sources.xiaoe_page as page_module

    monkeypatch.setattr(page_module, "validate_source_url", lambda value, **_: value)
    materialization = XiaoeHlsResolver(allowed_domains=frozenset({"example.test"})).resolve(
        "https://cdn.example.test/live.m3u8?signature=canary"
    )

    def ffmpeg(arguments: list[str]) -> None:
        assert not any("signature=canary" in value for value in arguments)
        assert arguments[arguments.index("-allowed_extensions") + 1] == "ALL"
        assert arguments[arguments.index("-protocol_whitelist") + 1] == "file,crypto,data"
        assert "-safe" not in arguments
        assert all(not value.startswith(("http:", "https:")) for value in arguments)
        Path(arguments[-1]).write_bytes(b"media")

    materializer = XiaoeMaterializer(playlist_downloader=_playlist_downloader, ffmpeg=ffmpeg)
    result = materializer.materialize(materialization, tmp_path)
    assert result.path.read_bytes() == b"media"
    assert "signature=canary" not in repr(result)
    assert "signature=canary" not in materialization.model_dump_json()


def test_materialization_revalidates_storage_state_immediately_before_cookie_jar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage_state = tmp_path / "operator-state.json"
    storage_state.write_text("replaced-after-resolution", encoding="utf-8")
    calls: list[str] = []
    expected_cookie_jar = object()

    def validate(path: Path) -> None:
        assert path == storage_state
        calls.append("validate")

    def make_jar(path: Path) -> object:
        assert path == storage_state
        calls.append("cookie-jar")
        return expected_cookie_jar

    def playlist(_url: str, target: Path, *, cookie_jar: object | None, **_kwargs: object) -> str:
        assert cookie_jar is expected_cookie_jar
        calls.append("download")
        path = target / "local.m3u8"
        path.write_text("#EXTM3U\n", encoding="utf-8")
        return str(path)

    def ffmpeg(arguments: list[str]) -> None:
        Path(arguments[-1]).write_bytes(b"media")

    monkeypatch.setattr(materializer_module, "validate_storage_state", validate)
    monkeypatch.setattr(materializer_module, "StorageStateCookieJar", make_jar)
    XiaoeMaterializer(playlist_downloader=playlist, ffmpeg=ffmpeg).materialize(
        _hls_materialization(), tmp_path, storage_state=storage_state
    )
    assert calls == ["validate", "cookie-jar", "download"]


def test_media_probe_rejects_single_track_and_never_publishes(tmp_path: Path) -> None:
    def ffmpeg(arguments: list[str]) -> None:
        Path(arguments[-1]).write_bytes(b"not-a-real-video")

    def probe(_path: Path) -> dict:
        return {
            "format": {"duration": "10", "bit_rate": "1000"},
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
        }

    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(playlist_downloader=_playlist_downloader, ffmpeg=ffmpeg, probe=probe).materialize(
            _hls_materialization(), tmp_path
        )
    assert error.value.code == "MEDIA_PROBE_INVALID"
    assert not (tmp_path / "source.mp4").exists()


@pytest.mark.parametrize("method", ["SAMPLE-AES", "SAMPLE-AES-CTR", "UNKNOWN"])
def test_drm_policy_rejects_before_ffmpeg(method: str, tmp_path: Path) -> None:
    called = False

    def playlist(_url: str, _target: Path, *, manifest_validator, **_: object) -> str:
        manifest_validator(f"#EXTM3U\n#EXT-X-KEY:METHOD={method},URI=key\n")
        raise AssertionError("unsupported DRM must not fetch a key")

    def ffmpeg(_: list[str]) -> None:
        nonlocal called
        called = True

    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(playlist_downloader=playlist, ffmpeg=ffmpeg).materialize(_hls_materialization(), tmp_path)
    assert error.value.code == "SOURCE_DRM_UNSUPPORTED"
    assert not called


def test_hls_resource_limit_code_is_preserved_at_materialization_boundary(tmp_path: Path) -> None:
    def playlist(_url: str, target: Path, **_kwargs: object) -> str:
        cache = target / ".safe-hls"
        cache.mkdir()
        (cache / "partial.ts").write_bytes(b"partial")
        (target / "source.part.mp4").write_bytes(b"partial")
        raise HlsResourceLimitError("HLS_TOTAL_BYTES_LIMIT_EXCEEDED")

    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(playlist_downloader=playlist).materialize(_hls_materialization(), tmp_path)

    assert error.value.code == "HLS_TOTAL_BYTES_LIMIT_EXCEEDED"
    assert not (tmp_path / ".safe-hls").exists()
    assert not (tmp_path / "source.part.mp4").exists()


def test_expired_signed_url_reresolves_once_and_missing_media_fails_closed(tmp_path: Path) -> None:
    expired = _hls_materialization(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    fresh = _hls_materialization(url="https://cdn.example.test/live.m3u8?fresh=1")
    calls = 0

    def reresolve() -> SourceMaterialization:
        nonlocal calls
        calls += 1
        return fresh

    def ffmpeg(arguments: list[str]) -> None:
        Path(arguments[-1]).write_bytes(b"media")

    result = XiaoeMaterializer(playlist_downloader=_playlist_downloader, ffmpeg=ffmpeg).materialize(
        expired, tmp_path, reresolve=reresolve
    )
    assert result.path.is_file() and calls == 1
    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(playlist_downloader=lambda *_args, **_kwargs: str(tmp_path / "missing.m3u8")).materialize(
            _hls_materialization(), tmp_path
        )
    assert error.value.code == "SOURCE_MEDIA_NOT_FOUND"


def test_direct_hls_rejects_unapproved_network_target() -> None:
    with pytest.raises(XiaoeResolutionError) as error:
        XiaoeHlsResolver(allowed_domains=frozenset({"xiaoe-tech.com"})).resolve("https://attacker.example/live.m3u8")
    assert error.value.code == "SOURCE_DOMAIN_NOT_ALLOWLISTED"


def _dash_materialization(url: str = "https://cdn.example.test/manifest.mpd?token=dash-secret-canary"):
    return SourceMaterialization(
        public=ResolvedSource(
            source_type="xiaoe",
            canonical_source_ref="course-1/lesson-2",
            source_identity_hash="source",
            platform_id="course-1",
            title="lesson",
        ),
        streams=[MediaStream(stream_id="dash", kind="dash", url=SecretStr(url))],
        subtitles=[],
        credential_ref_hash="authorized-session",
        requires_reresolve=True,
    )


def _dash_downloader(contents: dict[str, bytes]):
    def download(url: str, target: Path, **_: object) -> str:
        if url not in contents:
            raise RuntimeError("404")
        target.write_bytes(contents[url])
        return url

    return download


def _dash_http_downloader(contents: dict[str, bytes], failures: dict[str, int]):
    """Offline safe-fetch seam: statuses carry no response body or locator."""

    def download(url: str, target: Path, **_: object) -> str:
        if url in failures:
            raise SourceDownloadHTTPError(failures[url])
        if url not in contents:
            raise SourceDownloadHTTPError(404)
        target.write_bytes(contents[url])
        return url

    return download


def test_authorized_dash_is_rewritten_to_a_local_only_graph(tmp_path: Path) -> None:
    mpd_url = "https://cdn.example.test/manifest.mpd?token=dash-secret-canary"
    mpd = (
        b'<MPD type="static" mediaPresentationDuration="PT2S"><Period>'
        b'<AdaptationSet><Representation id="video"><SegmentList>'
        b'<Initialization sourceURL="init.mp4"/><SegmentURL media="one.m4s"/>'
        b'<SegmentURL media="two.m4s"/></SegmentList></Representation></AdaptationSet>'
        b'<AdaptationSet><Representation id="audio"><SegmentList>'
        b'<Initialization sourceURL="audio-init.mp4"/><SegmentURL media="audio.m4s"/>'
        b"</SegmentList></Representation></AdaptationSet></Period></MPD>"
    )
    contents = {
        mpd_url: mpd,
        "https://cdn.example.test/init.mp4": b"init",
        "https://cdn.example.test/one.m4s": b"one",
        "https://cdn.example.test/two.m4s": b"two",
        "https://cdn.example.test/audio-init.mp4": b"audio-init",
        "https://cdn.example.test/audio.m4s": b"audio",
    }
    seen: list[str] = []
    local_graphs: list[str] = []

    def ffmpeg(arguments: list[str]) -> None:
        seen.extend(arguments)
        local_graphs.append(Path(arguments[arguments.index("-i") + 1]).read_text(encoding="utf-8"))
        Path(arguments[-1]).write_bytes(b"media")

    materializer = XiaoeMaterializer(dash_localizer=DashLocalizer(downloader=_dash_downloader(contents)), ffmpeg=ffmpeg)
    result = materializer.materialize(_dash_materialization(mpd_url), tmp_path)
    local_mpd = tmp_path / "source.local.mpd"
    assert result.path.read_bytes() == b"media"
    assert str(local_mpd) in seen and all("dash-secret-canary" not in item for item in seen)
    assert "http" not in local_graphs[0].lower()
    assert not local_mpd.exists() and not (tmp_path / ".safe-dash").exists()
    assert "dash-secret-canary" not in repr(result)


def test_authorized_page_selected_dash_reaches_the_local_materializer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import stock_content.adapters.sources.xiaoe_page as page_module

    mpd_url = "https://cdn.example.test/manifest.mpd?dash-page-secret"
    monkeypatch.setattr(page_module, "validate_source_url", lambda value, **_: value)
    provider, reference_hash = _provider(tmp_path)
    capture = PageCapture(
        "course-1",
        "lesson-2",
        "Authorized lesson",
        None,
        None,
        CapturedResponse(SecretStr(mpd_url), "dash", {}),
    )
    resolved = XiaoePageResolver(
        credential_provider=provider,
        browser=_Browser(capture),
        allowed_domains=frozenset({"example.test"}),
    ).resolve("course-1/lesson-2", credential_ref_hash=reference_hash)
    mpd = (
        b'<MPD type="static" mediaPresentationDuration="PT1S"><Period><AdaptationSet>'
        b'<Representation><SegmentList><Initialization sourceURL="init"/><SegmentURL media="one"/>'
        b"</SegmentList></Representation></AdaptationSet></Period></MPD>"
    )
    contents = {mpd_url: mpd, "https://cdn.example.test/init": b"i", "https://cdn.example.test/one": b"s"}

    def ffmpeg(arguments: list[str]) -> None:
        assert all("dash-page-secret" not in item for item in arguments)
        Path(arguments[-1]).write_bytes(b"media")

    result = XiaoeMaterializer(
        dash_localizer=DashLocalizer(downloader=_dash_downloader(contents)), ffmpeg=ffmpeg
    ).materialize(resolved, tmp_path)
    assert result.path.is_file()


@pytest.mark.parametrize(
    ("mpd", "expected"),
    [
        (b'<MPD type="dynamic" mediaPresentationDuration="PT1S"/>', "SOURCE_DASH_UNSUPPORTED"),
        (
            b'<MPD type="static" mediaPresentationDuration="PT1S"><Period><ContentProtection/></Period></MPD>',
            "SOURCE_DRM_UNSUPPORTED",
        ),
        (
            (
                b'<MPD type="static" mediaPresentationDuration="PT1S"><Period><AdaptationSet>'
                b'<Representation><SegmentList><Initialization sourceURL="x"/></SegmentList>'
                b"</Representation></AdaptationSet></Period></MPD>"
            ),
            "SOURCE_DASH_LIMIT_EXCEEDED",
        ),
    ],
)
def test_dash_rejects_unsupported_or_protected_manifests(tmp_path: Path, mpd: bytes, expected: str) -> None:
    url = "https://cdn.example.test/manifest.mpd"
    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(dash_localizer=DashLocalizer(downloader=_dash_downloader({url: mpd}))).materialize(
            _dash_materialization(url), tmp_path
        )
    assert error.value.code == expected


@pytest.mark.parametrize("code", ["SOURCE_DOMAIN_NOT_ALLOWLISTED", "SOURCE_PRIVATE_ADDRESS", "SOURCE_REDIRECT_UNSAFE"])
def test_dash_propagates_network_policy_failures_without_locator(tmp_path: Path, code: str) -> None:
    url = "https://cdn.example.test/manifest.mpd?secret-canary"

    def unsafe(_url: str, _target: Path, **_: object) -> str:
        raise UnsafeSourceURL("blocked", code=code, url="https://private.invalid/secret-canary")

    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(dash_localizer=DashLocalizer(downloader=unsafe)).materialize(
            _dash_materialization(url), tmp_path
        )
    assert error.value.code == code
    assert "secret-canary" not in str(error.value)


def test_dash_rejects_cross_host_baseurl_before_asset_fetch(tmp_path: Path) -> None:
    url = "https://cdn.example.test/manifest.mpd"
    mpd = (
        b'<MPD type="static" mediaPresentationDuration="PT1S"><BaseURL>https://other.example.test/</BaseURL>'
        b'<Period><AdaptationSet><Representation><SegmentList><Initialization sourceURL="init"/>'
        b'<SegmentURL media="one"/></SegmentList></Representation></AdaptationSet></Period></MPD>'
    )
    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(dash_localizer=DashLocalizer(downloader=_dash_downloader({url: mpd}))).materialize(
            _dash_materialization(url), tmp_path
        )
    assert error.value.code == "SOURCE_DOMAIN_NOT_ALLOWLISTED"


def test_dash_expiry_reresolves_once(tmp_path: Path) -> None:
    expired = _dash_materialization()
    expired.streams[0].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    fresh = _dash_materialization("https://cdn.example.test/fresh.mpd")
    mpd = (
        b'<MPD type="static" mediaPresentationDuration="PT1S"><Period><AdaptationSet>'
        b'<Representation><SegmentList><Initialization sourceURL="init"/><SegmentURL media="one"/>'
        b"</SegmentList></Representation></AdaptationSet></Period></MPD>"
    )
    contents = {
        "https://cdn.example.test/fresh.mpd": mpd,
        "https://cdn.example.test/init": b"i",
        "https://cdn.example.test/one": b"s",
    }
    calls = 0

    def reresolve() -> SourceMaterialization:
        nonlocal calls
        calls += 1
        return fresh

    def ffmpeg(arguments: list[str]) -> None:
        Path(arguments[-1]).write_bytes(b"media")

    XiaoeMaterializer(dash_localizer=DashLocalizer(downloader=_dash_downloader(contents)), ffmpeg=ffmpeg).materialize(
        expired, tmp_path, reresolve=reresolve
    )
    assert calls == 1


def _static_dash_mpd(*, asset: str = "one") -> bytes:
    return (
        b'<MPD type="static" mediaPresentationDuration="PT1S"><Period><AdaptationSet>'
        b'<Representation><SegmentList><Initialization sourceURL="init"/><SegmentURL media="'
        + asset.encode("ascii")
        + b'"/></SegmentList></Representation></AdaptationSet></Period></MPD>'
    )


@pytest.mark.parametrize("expires_at", [False, True])
def test_dash_http_403_manifest_reresolves_once_from_a_clean_graph(tmp_path: Path, expires_at: bool) -> None:
    expired_url = "https://cdn.example.test/expired.mpd?signature=expired-canary"
    fresh_url = "https://cdn.example.test/fresh.mpd?signature=fresh-canary"
    expired = _dash_materialization(expired_url)
    if expires_at:
        expired.streams[0].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    fresh = _dash_materialization(fresh_url)
    contents = {
        fresh_url: _static_dash_mpd(),
        "https://cdn.example.test/init": b"i",
        "https://cdn.example.test/one": b"s",
    }
    calls = 0
    seen: list[str] = []

    def reresolve() -> SourceMaterialization:
        nonlocal calls
        calls += 1
        return fresh

    def ffmpeg(arguments: list[str]) -> None:
        seen.extend(arguments)
        Path(arguments[-1]).write_bytes(b"fresh-media")

    result = XiaoeMaterializer(
        dash_localizer=DashLocalizer(downloader=_dash_http_downloader(contents, {expired_url: 403})), ffmpeg=ffmpeg
    ).materialize(expired, tmp_path, reresolve=reresolve)

    assert result.path.read_bytes() == b"fresh-media"
    assert calls == 1
    assert not (tmp_path / ".safe-dash").exists()
    assert all("expired-canary" not in value and "fresh-canary" not in value for value in seen)


def test_dash_http_403_segment_reresolves_once_and_rebuilds_graph(tmp_path: Path) -> None:
    expired_url = "https://cdn.example.test/expired.mpd?signature=expired-canary"
    fresh_url = "https://cdn.example.test/fresh.mpd?signature=fresh-canary"
    expired = _dash_materialization(expired_url)
    fresh = _dash_materialization(fresh_url)
    contents = {
        expired_url: _static_dash_mpd(),
        fresh_url: _static_dash_mpd(asset="fresh-one"),
        "https://cdn.example.test/init": b"i",
        "https://cdn.example.test/one": b"s",
        "https://cdn.example.test/fresh-one": b"fresh-s",
    }
    calls = 0

    def reresolve() -> SourceMaterialization:
        nonlocal calls
        calls += 1
        return fresh

    def ffmpeg(arguments: list[str]) -> None:
        Path(arguments[-1]).write_bytes(b"fresh-media")

    result = XiaoeMaterializer(
        dash_localizer=DashLocalizer(
            downloader=_dash_http_downloader(contents, {"https://cdn.example.test/one": 403})
        ),
        ffmpeg=ffmpeg,
    ).materialize(expired, tmp_path, reresolve=reresolve)

    assert result.path.read_bytes() == b"fresh-media"
    assert calls == 1
    assert not (tmp_path / ".safe-dash").exists()


def test_dash_second_http_403_is_terminal_after_one_refresh(tmp_path: Path) -> None:
    expired_url = "https://cdn.example.test/expired.mpd?signature=first-canary"
    fresh_url = "https://cdn.example.test/fresh.mpd?signature=second-canary"
    calls = 0

    def reresolve() -> SourceMaterialization:
        nonlocal calls
        calls += 1
        return _dash_materialization(fresh_url)

    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(
            dash_localizer=DashLocalizer(
                downloader=_dash_http_downloader({}, {expired_url: 403, fresh_url: 403})
            )
        ).materialize(_dash_materialization(expired_url), tmp_path, reresolve=reresolve)
    assert error.value.code == "SOURCE_SESSION_EXPIRED"
    assert calls == 1
    assert "first-canary" not in str(error.value) and "second-canary" not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        UnsafeSourceURL("blocked", code="SOURCE_PRIVATE_ADDRESS", url="https://private.invalid/canary"),
        UnsafeSourceURL("blocked", code="SOURCE_DOMAIN_NOT_ALLOWLISTED", url="https://other.invalid/canary"),
        DashMaterializationError("SOURCE_DRM_UNSUPPORTED"),
        DashMaterializationError("SOURCE_DASH_UNSUPPORTED"),
        SourceDownloadHTTPError(404),
        SourceDownloadHTTPError(500),
    ],
)
def test_dash_non_expiry_failures_never_refresh(tmp_path: Path, failure: Exception) -> None:
    url = "https://cdn.example.test/manifest.mpd?negative-canary"
    calls = 0

    def downloader(_url: str, _target: Path, **_: object) -> str:
        raise failure

    def reresolve() -> SourceMaterialization:
        nonlocal calls
        calls += 1
        return _dash_materialization("https://cdn.example.test/fresh.mpd")

    with pytest.raises(XiaoeMaterializationError) as error:
        XiaoeMaterializer(dash_localizer=DashLocalizer(downloader=downloader)).materialize(
            _dash_materialization(url), tmp_path, reresolve=reresolve
        )
    assert calls == 0
    assert "negative-canary" not in str(error.value)

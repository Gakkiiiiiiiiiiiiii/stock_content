from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from stock_content.adapters.credentials.file_secret_provider import FileSecretProvider, SecretUnavailable
from stock_content.adapters.sources.bilibili import BilibiliSourceAdapter
from stock_content.adapters.sources.bilibili_materializer import (
    BilibiliMaterializationError,
    BilibiliMaterializer,
    MaterializedMedia,
)
from stock_content.adapters.sources.bilibili_resolver import (
    BilibiliResolver,
    canonical_bilibili_url,
    select_chinese_subtitle,
)
from stock_content.adapters.sources.security import UnsafeSourceURL
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import (
    ASRStage,
    DownloadStage,
    EvidenceGroundingStage,
    ResolveSourceStage,
    TranscriptCandidateStage,
    TranscriptSelectionStage,
)
from stock_content.domain.claim_draft import ClaimOccurrenceDraft
from stock_content.domain.semantic_segment import materialize_semantic_segments
from stock_content.domain.source_materialization import (
    MediaStream,
    ResolvedSource,
    SourceMaterialization,
    SubtitleTrack,
)


def _materialization(*, url: str = "https://cdn.bilivideo.com/secret?token=do-not-persist", expires_at=None):
    return SourceMaterialization(
        public=ResolvedSource(
            source_type="bilibili", canonical_source_ref="https://www.bilibili.com/video/BV1fixture",
            source_identity_hash="identity", platform_id="BV1fixture", title="fixture",
        ),
        streams=[MediaStream(stream_id="muxed", kind="muxed", url=SecretStr(url), expires_at=expires_at)],
        subtitles=[],
    )


@pytest.mark.parametrize(
    ("value", "expected", "part"),
    [
        ("BV1fixture", "https://www.bilibili.com/video/BV1fixture", None),
        ("av170001", "https://www.bilibili.com/video/av170001", None),
        ("https://www.bilibili.com/video/BV1fixture?p=2", "https://www.bilibili.com/video/BV1fixture", 2),
        ("https://b23.tv/fixture", "https://www.bilibili.com/video/BV1fixture", 3),
    ],
)
def test_canonical_inputs_cover_bv_av_url_b23_and_part(value: str, expected: str, part: int | None) -> None:
    actual, selected_part = canonical_bilibili_url(
        value,
        redirect_expander=lambda _: "https://www.bilibili.com/video/BV1fixture?p=3",
    )
    assert (actual, selected_part) == (expected, part)


def test_b23_private_redirect_cannot_become_a_canonical_source() -> None:
    with pytest.raises(UnsafeSourceURL) as error:
        canonical_bilibili_url("https://b23.tv/fixture", redirect_expander=lambda _: "https://127.0.0.1/private")
    assert error.value.code == "SOURCE_DOMAIN_NOT_ALLOWLISTED"


def test_resolver_produces_ephemeral_materialization_and_uses_bilibili_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stock_content.adapters.sources.bilibili_resolver as module

    monkeypatch.setattr(module, "validate_source_url", lambda value, **_: value)
    arguments: list[str] = []

    def extract(values: list[str]) -> dict:
        arguments.extend(values)
        return {
            "id": "BV1fixture", "title": "fixture", "uploader": "owner", "timestamp": 1,
            "url": "https://cdn.bilivideo.com/stream?token=do-not-persist",
            "http_headers": {"Referer": "https://www.bilibili.com/video/BV1fixture", "Cookie": "do-not-persist"},
            "subtitles": {
                "zh-CN": [
                    {"url": "https://cdn.bilivideo.com/manual.vtt?token=do-not-persist", "ext": "vtt"}
                ]
            },
        }

    materialization = BilibiliResolver(extractor=extract).resolve("BV1fixture")
    assert "--use-extractors" in arguments and arguments[arguments.index("--use-extractors") + 1] == "Bilibili"
    assert "--ignore-config" in arguments
    assert materialization.public.canonical_source_ref == "https://www.bilibili.com/video/BV1fixture"
    assert materialization.streams[0].headers == {"Referer": SecretStr("https://www.bilibili.com/video/BV1fixture")}
    persisted_projection = materialization.model_dump_json()
    assert "do-not-persist" not in persisted_projection
    assert "do-not-persist" not in repr(materialization)


def test_subtitle_priority_is_manual_then_automatic_then_asr() -> None:
    tracks = [
        SubtitleTrack(track_id="automatic", language="zh-CN", source="automatic", format="vtt", url=SecretStr("x")),
        SubtitleTrack(track_id="manual", language="zh-Hant", source="official", format="vtt", url=SecretStr("x")),
        SubtitleTrack(track_id="manual-cn", language="zh-CN", source="official", format="vtt", url=SecretStr("x")),
    ]
    selected, reason = select_chinese_subtitle(tracks)
    assert (selected.track_id if selected else None, reason) == ("manual-cn", "manual_zh_hans_or_zh_cn")
    selected, reason = select_chinese_subtitle(tracks[:2])
    assert (selected.track_id if selected else None, reason) == ("manual", "manual_other_chinese")
    selected, reason = select_chinese_subtitle(tracks[:1])
    assert (selected.track_id if selected else None, reason) == ("automatic", "automatic_zh_hans_or_zh_cn")
    assert select_chinese_subtitle([]) == (None, "asr_required")


def _probe(_: Path) -> dict:
    return {
        "format": {"duration": "12.0", "bit_rate": "64000"},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}, {"codec_type": "video", "codec_name": "h264"}],
    }


def test_expired_stream_is_reresolved_exactly_once_and_secrets_stay_out_of_result(tmp_path: Path) -> None:
    first = _materialization(url="https://cdn.bilivideo.com/old?token=old")
    second = _materialization(url="https://cdn.bilivideo.com/new?token=new")
    calls: list[str] = []
    refreshes = 0

    def download(url: str, target: Path, **_: object) -> str:
        calls.append(url)
        if "old" in url:
            raise RuntimeError("HTTP 403")
        target.write_bytes(b"media")
        return url

    def reresolve() -> SourceMaterialization:
        nonlocal refreshes
        refreshes += 1
        return second

    result = BilibiliMaterializer(downloader=download, probe=_probe).materialize(first, tmp_path, reresolve=reresolve)
    assert refreshes == 1 and len(calls) == 2
    assert "old" not in repr(result) and "new" not in repr(result)
    assert result.subtitle_metadata["type"] == "asr"


def test_materializer_fails_closed_for_integrity_and_tracks(tmp_path: Path) -> None:
    def download(_: str, target: Path, **__: object) -> str:
        target.write_bytes(b"media")
        return "https://cdn.bilivideo.com/final"

    expected = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(BilibiliMaterializationError) as error:
        BilibiliMaterializer(downloader=download, probe=_probe).materialize(
            _materialization(), tmp_path, expected_sha256=expected
        )
    assert error.value.code == "MEDIA_SHA256_MISMATCH"
    with pytest.raises(BilibiliMaterializationError) as error:
        BilibiliMaterializer(
            downloader=download,
            probe=lambda _: {"format": {"duration": "1", "bit_rate": "1"}, "streams": []},
        ).materialize(_materialization(), tmp_path)
    assert error.value.code == "MEDIA_TRACKS_INVALID"


def test_file_secret_provider_returns_only_allowlisted_readonly_path(tmp_path: Path) -> None:
    secret = tmp_path / "cookie.txt"
    secret.write_text("cookie-canary", encoding="utf-8")
    if os.name != "nt":
        secret.chmod(0o400)
    provider = FileSecretProvider({"public-authorized-cookie": secret})
    assert provider.resolve("public-authorized-cookie") == secret.resolve()
    assert "cookie-canary" not in repr(provider)
    with pytest.raises(SecretUnavailable) as error:
        provider.resolve("unknown")
    assert str(error.value) == "SOURCE_SESSION_EXPIRED"


def test_authorized_cookiefile_is_resolved_in_worker_only_and_never_projected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import stock_content.adapters.sources.bilibili_resolver as resolver_module

    monkeypatch.setattr(resolver_module, "validate_source_url", lambda value, **_: value)
    cookiefile = tmp_path / "cookies.txt"
    cookiefile.write_text("cookie-canary", encoding="utf-8")
    if os.name != "nt":
        cookiefile.chmod(0o400)
    reference = "authorized-bilibili-cookie"
    seen: list[str] = []

    adapter = BilibiliSourceAdapter(credential_provider=FileSecretProvider({reference: cookiefile}))
    monkeypatch.setattr(
        adapter,
        "_run",
        lambda arguments: (
            seen.extend(arguments)
            or SimpleNamespace(stdout=json.dumps({
                "id": "BV1fixture", "title": "fixture", "url": "https://cdn.bilivideo.com/media?token=canary",
            }))
        ),
    )
    materialization = adapter.resolve_materialization(
        "BV1fixture", credential_ref_hash=hashlib.sha256(reference.encode()).hexdigest()
    )
    assert seen[seen.index("--cookies") + 1] == str(cookiefile.resolve())
    assert "cookie-canary" not in repr(materialization)
    assert "cookie-canary" not in materialization.model_dump_json()


def test_legacy_adapter_registers_the_new_ephemeral_seam() -> None:
    assert callable(BilibiliSourceAdapter().resolve_materialization)
    assert callable(BilibiliSourceAdapter().materialize)


def test_pipeline_keeps_materialization_in_runtime_and_persists_only_public_subtitle_hashes(tmp_path: Path) -> None:
    materialization = _materialization()

    class Adapter:
        def resolve_materialization(self, _: str, *, part: int | None = None) -> SourceMaterialization:
            assert part == 2
            return materialization

        def materialize(self, _: SourceMaterialization, directory: Path, **__: object) -> MaterializedMedia:
            media = directory / "media.bin"
            media.write_bytes(b"media")
            return MaterializedMedia(
                media,
                hashlib.sha256(b"media").hexdigest(),
                1.0,
                {
                    "language": "zh-CN",
                    "type": "official",
                    "selection_reason": "manual_zh_hans_or_zh_cn",
                    "raw_sha256": "raw-hash",
                    "normalized_sha256": "normalized-hash",
                },
            )

    context = PipelineContext("bili-runtime", source={"type": "bilibili", "ref": "BV1fixture"}, options={"part": 2})
    adapter = Adapter()
    ResolveSourceStage({"bilibili": adapter}).execute(context)
    assert context.runtime.source_materialization is materialization
    assert "do-not-persist" not in repr(context.state.metadata)
    DownloadStage({"bilibili": adapter}, work_root=tmp_path).execute(context)
    assert context.artifacts.source is not None
    assert context.artifacts.source.source_metadata["subtitle"]["normalized_sha256"] == "normalized-hash"
    assert "do-not-persist" not in repr(context.artifacts.source.source_metadata)


@pytest.mark.parametrize(
    ("origin", "expected_source", "expected_asr_calls"),
    [("official", "OFFICIAL_SUBTITLE", 0), ("automatic", "AUTO_SUBTITLE", 0)],
)
def test_materialized_bilibili_subtitles_reach_candidate_stage_without_request_injection(
    tmp_path: Path, origin: str, expected_source: str, expected_asr_calls: int
) -> None:
    materialization = _materialization()
    materialization.subtitles = [
        SubtitleTrack(
            track_id=f"{origin}-track", language="zh-CN", source=origin, format="vtt",
            url=SecretStr(f"https://cdn.bilivideo.com/{origin}.vtt?token=do-not-persist"),
        )
    ]
    calls: list[str] = []

    def download(url: str, target: Path, **_: object) -> str:
        calls.append(url)
        target.write_bytes(
            b"WEBVTT\n\n00:00:00.000 --> 00:00:12.000\n\xe8\x90\xa5\xe6\x94\xb6100\xe4\xba\xbf\xe5\x85\x83 2025Q1\n"
            if url.endswith(".vtt?token=do-not-persist")
            else b"media"
        )
        return url

    class Adapter:
        def resolve_materialization(self, _: str, **__: object) -> SourceMaterialization:
            return materialization

        def materialize(self, value: SourceMaterialization, directory: Path, **__: object) -> MaterializedMedia:
            return BilibiliMaterializer(downloader=download, probe=_probe).materialize(value, directory)

    context = PipelineContext(
        "subtitle-runtime",
        source={"type": "bilibili", "ref": "https://b23.tv/fixture"},
        options={"part": 2},
    )
    adapter = Adapter()
    ResolveSourceStage({"bilibili": adapter}).execute(context)
    DownloadStage({"bilibili": adapter}, work_root=tmp_path).execute(context)
    assert "subtitle_candidates" not in context.options
    assert "do-not-persist" not in repr(context.state.metadata)
    TranscriptCandidateStage().execute(context)

    class CountASR:
        calls = 0

        def transcribe(self, *_: object) -> list[object]:
            self.calls += 1
            raise AssertionError("qualified materialized subtitle must skip ASR")

    recognizer = CountASR()
    ASRStage(recognizer).execute(context)
    TranscriptSelectionStage().execute(context)
    assert recognizer.calls == expected_asr_calls
    assert context.artifacts.transcript is not None
    segment = context.artifacts.transcript.segments[0]
    assert (segment.start_ms, segment.end_ms, segment.source) == (0, 12_000, expected_source)
    assert segment.source_artifact_id.startswith("subtitle-")
    assert calls, "the materializer, not a request option, fetched the subtitle"
    semantic_segment = materialize_semantic_segments(context.artifacts.transcript, [])[0]
    context.state.semantic_segments = [semantic_segment]
    context.state.claim_drafts = [ClaimOccurrenceDraft(
        semantic_segment_id=semantic_segment.semantic_segment_id,
        knowledge_kind="CLAIM",
        claim_type="FINANCIAL_METRIC",
        subject_key="600000",
        predicate_key="营收",
        conclusion="营收100亿元",
        evidence_segment_indices=[0],
        extraction_confidence=1.0,
    )]
    EvidenceGroundingStage().execute(context)
    evidence = context.artifacts.evidence.evidences[0]
    assert (evidence.start_ms, evidence.end_ms, evidence.evidence_text, evidence.source_type) == (
        0, 12_000, "营收100亿元 2025Q1", expected_source,
    )
    assert "do-not-persist" not in repr(context.artifacts)


def test_materialized_absent_subtitle_requires_asr_and_bad_cues_fail_closed(tmp_path: Path) -> None:
    context = PipelineContext(
        "no-subtitle", source={"type": "bilibili", "ref": "BV1fixture"},
        options={"duration_ms": 12_000, "segments": [{"start_seconds": 0, "end_seconds": 12, "text": "营收100亿元"}]},
    )
    # Candidate-stage behaviour is independent of a locator and needs only a
    # media artifact; keep this explicit so request options cannot impersonate
    # a materialized subtitle.
    from stock_content.domain.artifacts import MediaArtifact

    context.artifacts.media = MediaArtifact(artifact_id="media", artifact_type="media", duration_ms=12_000)
    TranscriptCandidateStage().execute(context)
    assert context.options["_asr_required"] is True
    ASRStage(object()).execute(context)
    assert len(context.state.transcript_candidates) == 1
    assert context.state.transcript_candidates[0].source.value == "ASR"

    bad = _materialization()
    bad.subtitles = [
        SubtitleTrack(track_id="bad", language="zh", source="official", format="vtt", url=SecretStr("https://cdn.bilivideo.com/bad.vtt"))
    ]

    def download(_: str, target: Path, **__: object) -> str:
        if target.suffix == ".vtt":
            target.write_bytes(b"00:00:08.000 --> 00:00:13.000\ninvalid\n")
        else:
            target.write_bytes(b"media")
        return "ok"

    with pytest.raises(BilibiliMaterializationError, match="SUBTITLE_CUES_INVALID"):
        BilibiliMaterializer(downloader=download, probe=_probe).materialize(bad, tmp_path)

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from stock_content.adapters.credentials.file_secret_provider import FileSecretProvider, SecretUnavailable
from stock_content.adapters.sources.security import download_hls_playlist, preflight_source_url
from stock_content.adapters.sources.xiaoe_materializer import XiaoeMaterializer
from stock_content.adapters.sources.xiaoe_page import (
    XiaoeHlsResolver,
    XiaoePageResolver,
    XiaoeResolutionError,
    page_resolver_from_environment,
)
from stock_content.domain.source_materialization import SourceMaterialization


class XiaoeHlsSourceAdapter:
    """Compatibility adapter plus the runtime-only direct-HLS seam."""

    def __init__(
        self,
        *,
        resolver: XiaoeHlsResolver | None = None,
        materializer: XiaoeMaterializer | None = None,
        credential_provider: FileSecretProvider | None = None,
    ) -> None:
        self._resolver = resolver or XiaoeHlsResolver()
        self._materializer = materializer or XiaoeMaterializer()
        self._credential_provider = credential_provider

    @classmethod
    def from_environment(cls) -> "XiaoeHlsSourceAdapter":
        reference = os.getenv("CONTENT_XIAOE_HLS_CREDENTIAL_REF", "").strip()
        locator_file = os.getenv("CONTENT_XIAOE_HLS_LOCATOR_FILE", "").strip()
        provider = FileSecretProvider({reference: locator_file}) if reference and locator_file else None
        return cls(credential_provider=provider)

    def resolve(self, source_ref: str) -> dict[str, Any]:
        if not source_ref.startswith(("http://", "https://")):
            raise ValueError("invalid HLS URL")
        preflight_source_url(source_ref)
        return {"source_ref": source_ref, "title": "Xiaoe course video", "author": None}

    def download(self, source_ref: str, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / "source.mp4"
        checked_url = preflight_source_url(source_ref)
        local_playlist = download_hls_playlist(checked_url, target_dir)
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-protocol_whitelist",
            "file,crypto,data",
            "-safe",
            "0",
            "-i",
            local_playlist,
            "-c",
            "copy",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg HLS download failed: {completed.stderr[-1500:]}")
        return output

    def resolve_materialization(
        self, source_ref: str, *, credential_ref_hash: str | None = None, **_: object
    ) -> SourceMaterialization:
        # The signed URL is materialized only at worker execution from an
        # allowlisted read-only secret file.  It never enters a task, log, or
        # checkpoint, and a refresh reruns this resolution exactly once.
        if credential_ref_hash:
            if self._credential_provider is None:
                raise XiaoeResolutionError("SOURCE_SESSION_EXPIRED")
            try:
                locator = self._credential_provider.resolve_hash(credential_ref_hash).read_text(
                    encoding="utf-8"
                ).strip()
            except (SecretUnavailable, OSError, UnicodeError) as exc:
                raise XiaoeResolutionError("SOURCE_SESSION_EXPIRED") from exc
            materialization = self._resolver.resolve(locator)
            if materialization.public.canonical_source_ref != source_ref:
                raise XiaoeResolutionError("SOURCE_LOCATOR_MISMATCH")
            return materialization.model_copy(
                update={"credential_ref_hash": credential_ref_hash, "requires_reresolve": True}
            )
        return self._resolver.resolve(source_ref)

    def materialize(self, materialization: SourceMaterialization, target_dir: Path, **kwargs: object):
        return self._materializer.materialize(materialization, target_dir, **kwargs)


class XiaoePageSourceAdapter:
    """Page mode is unavailable unless an operator configures an injected port."""

    def __init__(
        self, resolver: XiaoePageResolver | None = None, *, materializer: XiaoeMaterializer | None = None
    ) -> None:
        self._resolver = resolver
        self._materializer = materializer or XiaoeMaterializer()

    @classmethod
    def from_environment(cls) -> "XiaoePageSourceAdapter":
        return cls(page_resolver_from_environment())

    def resolve_materialization(
        self, source_ref: str, *, credential_ref_hash: str | None = None, **_: object
    ) -> SourceMaterialization:
        if self._resolver is None:
            raise XiaoeResolutionError("SOURCE_PAGE_RESOLVER_DISABLED")
        return self._resolver.resolve(source_ref, credential_ref_hash=credential_ref_hash)

    def materialize(self, materialization: SourceMaterialization, target_dir: Path, **kwargs: object):
        if self._resolver is None:
            raise XiaoeResolutionError("SOURCE_PAGE_RESOLVER_DISABLED")
        # The state file remains in this adapter and is used only by the
        # immediate materialization call; it is not attached to durable task
        # data, artifacts, checkpoints, or materialization metadata.
        return self._materializer.materialize(
            materialization,
            target_dir,
            storage_state=self._resolver.storage_state_for(materialization.credential_ref_hash),
            **kwargs,
        )

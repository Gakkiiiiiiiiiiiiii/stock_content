"""Fail-closed DRM classification for authorized source materialization."""
from __future__ import annotations

from enum import StrEnum


class DrmKind(StrEnum):
    NONE = "NONE"
    AES_128 = "AES_128"
    SAMPLE_AES = "SAMPLE_AES"
    WIDEVINE = "WIDEVINE"
    FAIRPLAY = "FAIRPLAY"
    UNKNOWN_ENCRYPTED = "UNKNOWN_ENCRYPTED"


class DrmPolicyError(RuntimeError):
    """Stable failure; it intentionally carries no manifest detail."""

    code = "SOURCE_DRM_UNSUPPORTED"

    def __init__(self) -> None:
        super().__init__(self.code)


def classify_hls_drm(playlist: str) -> DrmKind:
    """Classify only the encryption declarations needed for the allow policy."""
    encrypted = False
    for line in playlist.splitlines():
        upper = line.strip().upper()
        if not upper.startswith("#EXT-X-KEY"):
            continue
        encrypted = True
        if "METHOD=NONE" in upper:
            encrypted = False
        elif "METHOD=AES-128" in upper:
            return DrmKind.AES_128
        elif "METHOD=SAMPLE-AES" in upper:
            return DrmKind.SAMPLE_AES
        else:
            return DrmKind.UNKNOWN_ENCRYPTED
    return DrmKind.UNKNOWN_ENCRYPTED if encrypted else DrmKind.NONE


def require_supported_hls(playlist: str) -> DrmKind:
    """Allow clear media and AES-128 only; never acquire a DRM licence."""
    kind = classify_hls_drm(playlist)
    if kind not in {DrmKind.NONE, DrmKind.AES_128}:
        raise DrmPolicyError()
    return kind


def require_supported_drm(kind: DrmKind | str | None) -> DrmKind:
    """Reject DRM hints captured from a page before any media fetch occurs."""
    try:
        value = DrmKind(str(kind or DrmKind.NONE).upper())
    except ValueError as exc:
        raise DrmPolicyError() from exc
    if value not in {DrmKind.NONE, DrmKind.AES_128}:
        raise DrmPolicyError()
    return value


__all__ = ["DrmKind", "DrmPolicyError", "classify_hls_drm", "require_supported_drm", "require_supported_hls"]

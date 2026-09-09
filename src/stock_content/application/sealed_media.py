"""Fail-closed validation for replaying immutable private raw media."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class SealedMediaValidationError(ValueError):
    """The sealed replay handle cannot safely be used."""


@dataclass(frozen=True)
class SealedMedia:
    path: Path
    content_hash: str
    content_length: int


def validate_sealed_media(
    raw_storage_uri: str,
    *,
    private_root: str,
    expected_hash: str,
    expected_length: int | None,
) -> SealedMedia:
    """Resolve and hash a source-sealed file without accepting external locators."""
    if not private_root or not expected_hash:
        raise SealedMediaValidationError("sealed media root or hash is missing")
    if "://" in raw_storage_uri and not raw_storage_uri.startswith("file://"):
        raise SealedMediaValidationError("sealed media locator is not a local file")
    local_uri = raw_storage_uri.removeprefix("file://")
    if len(local_uri) >= 3 and local_uri[0] == "/" and local_uri[2] == ":":
        local_uri = local_uri[1:]
    try:
        root = Path(private_root).resolve(strict=True)
        path = Path(local_uri).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SealedMediaValidationError("sealed media is outside the private root") from exc
    if not root.is_dir() or not path.is_file():
        raise SealedMediaValidationError("sealed media is unavailable")
    digest = hashlib.sha256()
    length = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                length += len(chunk)
    except OSError as exc:
        raise SealedMediaValidationError("sealed media is unreadable") from exc
    if digest.hexdigest() != expected_hash:
        raise SealedMediaValidationError("sealed media hash does not match")
    if expected_length is not None and length != expected_length:
        raise SealedMediaValidationError("sealed media length does not match")
    return SealedMedia(path=path, content_hash=digest.hexdigest(), content_length=length)


__all__ = ["SealedMedia", "SealedMediaValidationError", "validate_sealed_media"]

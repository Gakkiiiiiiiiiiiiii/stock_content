"""Fail-closed, local-only materialization for a small static DASH subset.

The MPD locator is a runtime secret. It is consequently used only to fetch a
bounded graph and is never copied into the rewritten MPD, task data, or error
messages. ``safe_download_url`` is the only network seam: it DNS-pins each
request and validates redirects before bytes are read.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import xml.etree.ElementTree as element_tree
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from stock_content.adapters.sources.security import SourceDownloadHTTPError, UnsafeSourceURL, safe_download_url
from stock_content.domain.drm_policy import DrmPolicyError


class DashMaterializationError(RuntimeError):
    """A stable, locator-free DASH failure."""

    def __init__(self, code: str, *, retryable_locator_expiry: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable_locator_expiry = retryable_locator_expiry


_MAX_MPD_BYTES = 8 * 1024 * 1024
_MAX_REPRESENTATIONS = 16
_MAX_SEGMENTS = 1_200
_MAX_ASSET_BYTES = 128 * 1024 * 1024
_DURATION = re.compile(r"^PT(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _duration(value: str | None) -> float:
    if not value:
        raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
    match = _DURATION.fullmatch(value)
    if not match:
        raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
    return float(match.group("h") or 0) * 3600 + float(match.group("m") or 0) * 60 + float(match.group("s") or 0)


def _children(node: element_tree.Element, name: str) -> list[element_tree.Element]:
    return [child for child in node if _local_name(child.tag) == name]


def _first(node: element_tree.Element, name: str) -> element_tree.Element | None:
    return next(iter(_children(node, name)), None)


def _contains_protection(root: element_tree.Element) -> bool:
    protected = {"contentprotection", "pssh", "widevine", "playready", "fairplay", "encryption"}
    for item in root.iter():
        name = _local_name(item.tag).lower()
        values = " ".join(str(value) for value in item.attrib.values()).lower()
        if name in protected or any(marker in name or marker in values for marker in protected):
            return True
    return False


def _base_url(parent_url: str, node: element_tree.Element) -> str:
    bases = _children(node, "BaseURL")
    if len(bases) > 1:
        raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
    if not bases:
        return parent_url
    value = (bases[0].text or "").strip()
    if not value:
        raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
    return urljoin(parent_url, value)


def _same_media_host(source_url: str, candidate_url: str) -> bool:
    """DASH graphs do not get to pivot to another otherwise-allowlisted CDN."""
    return urlsplit(source_url).hostname == urlsplit(candidate_url).hostname


def _template_segments(template: element_tree.Element, *, duration_seconds: float, representation_id: str) -> list[str]:
    media = template.get("media")
    timescale = int(template.get("timescale", "1"))
    start = int(template.get("startNumber", "1"))
    if not media or timescale <= 0:
        raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
    timeline = _first(template, "SegmentTimeline")
    values: list[tuple[int, int]] = []
    if timeline is not None:
        current: int | None = None
        for item in _children(timeline, "S"):
            segment_duration = int(item.get("d", "0"))
            repeat = int(item.get("r", "0"))
            if segment_duration <= 0 or repeat < 0:
                raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
            current = int(item.get("t")) if item.get("t") is not None else current
            if current is None:
                current = 0
            for _ in range(repeat + 1):
                values.append((current, segment_duration))
                current += segment_duration
                if len(values) > _MAX_SEGMENTS:
                    raise DashMaterializationError("SOURCE_DASH_LIMIT_EXCEEDED")
    else:
        segment_duration = int(template.get("duration", "0"))
        if segment_duration <= 0:
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
        count = int((duration_seconds * timescale + segment_duration - 1) // segment_duration)
        if count < 1 or count > _MAX_SEGMENTS:
            raise DashMaterializationError("SOURCE_DASH_LIMIT_EXCEEDED")
        values = [(index * segment_duration, segment_duration) for index in range(count)]
    result: list[str] = []
    for index, (time, _) in enumerate(values):
        value = media.replace("$RepresentationID$", representation_id)
        value = re.sub(
            r"\$Number(?:%0(\d+)d)?\$",
            lambda match: f"{start + index:0{match.group(1)}d}" if match.group(1) else str(start + index),
            value,
        )
        value = value.replace("$Time$", str(time))
        if "$" in value:
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
        result.append(value)
    return result


class DashLocalizer:
    """Download a finite clear static MPD graph and write a local MPD only."""

    def __init__(self, *, downloader: Callable[..., str] = safe_download_url) -> None:
        self._downloader = downloader

    @staticmethod
    def cleanup(target_dir: Path) -> None:
        """Remove only the private graph that this materializer owns."""
        root = target_dir.resolve()
        cache = target_dir / ".safe-dash"
        local_mpd = target_dir / "source.local.mpd"
        try:
            if cache.exists() and not cache.is_symlink() and cache.resolve().is_relative_to(root):
                shutil.rmtree(cache)
            if local_mpd.is_file() and not local_mpd.is_symlink() and local_mpd.resolve().is_relative_to(root):
                local_mpd.unlink()
        except OSError:
            # Cleanup is best-effort and must not expose a locator or replace a
            # materialization result with an unrelated filesystem error.
            return

    def materialize(
        self, source_url: str, target_dir: Path, *, allowed_domains: frozenset[str], headers: dict[str, str]
    ) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        cache = target_dir / ".safe-dash"
        cache.mkdir(exist_ok=True)
        raw_mpd = cache / "manifest.xml"
        try:
            final_mpd = self._downloader(
                source_url, raw_mpd, allowed_domains=allowed_domains, headers=headers, max_bytes=_MAX_MPD_BYTES
            )
            body = raw_mpd.read_bytes()
        except DashMaterializationError:
            raise
        except UnsafeSourceURL as exc:
            raise DashMaterializationError(exc.code) from exc
        except SourceDownloadHTTPError as exc:
            if exc.status in {401, 403}:
                raise DashMaterializationError("SOURCE_SESSION_EXPIRED", retryable_locator_expiry=True) from exc
            raise DashMaterializationError("SOURCE_MEDIA_NOT_FOUND") from exc
        except Exception as exc:
            raise DashMaterializationError("SOURCE_MEDIA_NOT_FOUND") from exc
        if len(body) > _MAX_MPD_BYTES or b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
        try:
            root = element_tree.fromstring(body)
        except element_tree.ParseError as exc:
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED") from exc
        if (
            _local_name(root.tag) != "MPD"
            or root.get("type", "static").lower() != "static"
            or _contains_protection(root)
        ):
            if _contains_protection(root):
                raise DrmPolicyError()
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
        presentation_duration = _duration(root.get("mediaPresentationDuration"))
        period = _first(root, "Period")
        if period is None or len(_children(root, "Period")) != 1:
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
        root_base = _base_url(final_mpd, root)
        period_base = _base_url(root_base, period)
        if not _same_media_host(final_mpd, root_base) or not _same_media_host(final_mpd, period_base):
            raise DashMaterializationError("SOURCE_DOMAIN_NOT_ALLOWLISTED")
        representations = 0
        downloaded = 0

        def fetch(remote: str) -> str:
            nonlocal downloaded
            locator = urljoin(period_base, remote)
            if not _same_media_host(final_mpd, locator):
                raise DashMaterializationError("SOURCE_DOMAIN_NOT_ALLOWLISTED")
            filename = hashlib.sha256(locator.encode("utf-8")).hexdigest() + ".m4s"
            output = cache / filename
            if not output.exists():
                try:
                    final_asset = self._downloader(
                        locator, output, allowed_domains=allowed_domains, headers=headers, max_bytes=_MAX_ASSET_BYTES
                    )
                    if not _same_media_host(final_mpd, final_asset):
                        raise DashMaterializationError("SOURCE_REDIRECT_UNSAFE")
                except UnsafeSourceURL as exc:
                    raise DashMaterializationError(exc.code) from exc
                except SourceDownloadHTTPError as exc:
                    if exc.status in {401, 403}:
                        raise DashMaterializationError("SOURCE_SESSION_EXPIRED", retryable_locator_expiry=True) from exc
                    raise DashMaterializationError("SOURCE_MEDIA_NOT_FOUND") from exc
                except Exception as exc:
                    raise DashMaterializationError("SOURCE_MEDIA_NOT_FOUND") from exc
                downloaded += output.stat().st_size
                if downloaded > _MAX_ASSET_BYTES:
                    raise DashMaterializationError("SOURCE_DASH_LIMIT_EXCEEDED")
            return str(output.relative_to(target_dir)).replace("\\", "/")

        for adaptation in _children(period, "AdaptationSet"):
            adaptation_base = _base_url(period_base, adaptation)
            for representation in _children(adaptation, "Representation"):
                representations += 1
                if representations > _MAX_REPRESENTATIONS:
                    raise DashMaterializationError("SOURCE_DASH_LIMIT_EXCEEDED")
                representation_base = _base_url(adaptation_base, representation)
                template = _first(representation, "SegmentTemplate")
                if template is None:
                    template = _first(adaptation, "SegmentTemplate")
                if template is None:
                    template = _first(period, "SegmentTemplate")
                segment_list = _first(representation, "SegmentList")
                if segment_list is None:
                    segment_list = _first(adaptation, "SegmentList")
                if (template is None) == (segment_list is None):
                    raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
                for parent in (period, adaptation, representation):
                    for base in _children(parent, "BaseURL"):
                        parent.remove(base)
                    for inherited in _children(parent, "SegmentTemplate"):
                        parent.remove(inherited)
                local_list = element_tree.Element("SegmentList")
                if template is not None:
                    init = template.get("initialization")
                    if not init:
                        raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
                    initialization = fetch(urljoin(representation_base, init))
                    segments = [
                        fetch(urljoin(representation_base, item))
                        for item in _template_segments(
                            template,
                            duration_seconds=presentation_duration,
                            representation_id=representation.get("id", "0"),
                        )
                    ]
                else:
                    init_node = _first(segment_list, "Initialization")
                    if init_node is None or not init_node.get("sourceURL"):
                        raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
                    segment_nodes = [node for node in _children(segment_list, "SegmentURL") if node.get("media")]
                    if not segment_nodes or len(segment_nodes) > _MAX_SEGMENTS:
                        raise DashMaterializationError("SOURCE_DASH_LIMIT_EXCEEDED")
                    initialization = fetch(urljoin(representation_base, init_node.get("sourceURL", "")))
                    segments = [fetch(urljoin(representation_base, node.get("media", ""))) for node in segment_nodes]
                element_tree.SubElement(local_list, "Initialization", {"sourceURL": initialization})
                for segment in segments:
                    element_tree.SubElement(local_list, "SegmentURL", {"media": segment})
                for prior in _children(representation, "SegmentList"):
                    representation.remove(prior)
                representation.append(local_list)
        if not representations:
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
        output = target_dir / "source.local.mpd"
        element_tree.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
        text = output.read_text(encoding="utf-8")
        if any(value in text.lower() for value in ("http:", "https:", "<baseurl")):
            raise DashMaterializationError("SOURCE_DASH_UNSUPPORTED")
        return output


__all__ = ["DashLocalizer", "DashMaterializationError"]

"""Canonical ingestion request normalization; deliberately no network resolution."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from stock_content.domain.source_materialization import ContentIngestionCommand, CredentialReference


class IngestionValidationError(ValueError):
    code = "INVALID_INGESTION_REQUEST"


_FORBIDDEN_OPTION_PARTS = ("cookie", "header", "storage_state", "storage state", "signed_url", "signed url")
_BILIBILI_BV = re.compile(r"\b(BV[0-9A-Za-z]+)\b", re.IGNORECASE)
_BILIBILI_AV = re.compile(r"\b(av[1-9][0-9]*)\b", re.IGNORECASE)
_XIAOE_PAGE_PATH = re.compile(r"^/p/course/video/([A-Za-z0-9_-]{1,160})/?$")
_STABLE_XIAOE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*(?:/[A-Za-z0-9][A-Za-z0-9_.:-]*)?$")
_ALLOWED_OPTIONS = {"language"}
_LEGACY_ALLOWED_OPTIONS = _ALLOWED_OPTIONS | {
    # Compatibility fixture and deterministic replay controls from the
    # pre-canonical ingestion API.  This list is intentionally finite; it is
    # not an escape hatch for credentials, headers, or browser state.
    "metadata", "transcript", "offline_fixture", "asr_model", "asr_model_version",
    "duration_ms",
    "quant_market_snapshot_ids", "code_sha", "available_from", "trace_id",
}


def credential_allowlist_from_environment() -> tuple[frozenset[str], frozenset[str]]:
    """The API accepts only explicitly configured worker credential references."""
    references = frozenset(
        item.strip() for item in os.getenv("CONTENT_INGESTION_CREDENTIAL_REFS", "").split(",") if item.strip()
    )
    providers = frozenset(
        item.strip()
        for item in os.getenv("CONTENT_INGESTION_CREDENTIAL_PROVIDERS", "file-secret").split(",")
        if item.strip()
    )
    # File-backed media credentials are intentionally named by the operator,
    # never supplied by a caller.  Include the two source-specific names here
    # so the API and video worker cannot drift into different allowlists.
    references = references | frozenset(
        value.strip()
        for value in (
            os.getenv("CONTENT_BILIBILI_CREDENTIAL_REF", ""),
            os.getenv("CONTENT_XIAOE_HLS_CREDENTIAL_REF", ""),
            # The page resolver consumes the Playwright storage state only in
            # the video worker.  The API still needs to recognise its opaque,
            # operator-configured reference when it validates a command.
            os.getenv("CONTENT_XIAOE_CREDENTIAL_REF", ""),
        )
        if value.strip()
    )
    providers = providers | frozenset(
        value.strip()
        for value in (os.getenv("CONTENT_XIAOE_CREDENTIAL_PROVIDER", ""),)
        if value.strip()
    )
    return references, providers


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_hash_for(command: ContentIngestionCommand) -> str:
    """Hash only public, canonical request fields; secrets are intentionally excluded."""
    return _hash(_canonical_json({
        "source_type": command.source_type,
        "canonical_source_ref": command.canonical_source_ref,
        "part": command.part,
        "transcript_policy": command.transcript_policy,
        "options": command.options,
    }))


def source_identity_hash_for(source_type: str, canonical_source_ref: str) -> str:
    return _hash(f"{source_type}:{canonical_source_ref}")


def _safe_public_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise IngestionValidationError("source URL must be an absolute https URL")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def canonical_bilibili_ref(*, url: str | None, bv_id: str | None) -> str:
    if bool(url) == bool(bv_id):
        raise IngestionValidationError("exactly one of url or bv_id is required")
    candidate = bv_id or url or ""
    # Do not follow b23 redirects at HTTP ingress.  Redirect expansion is a
    # network operation and belongs to the SSRF-protected video worker.
    match = _BILIBILI_BV.fullmatch(candidate.strip())
    if match:
        return "BV" + match.group(1)[2:]
    av_match = _BILIBILI_AV.fullmatch(candidate.strip())
    if av_match:
        return av_match.group(1).lower()
    parsed = urlsplit(candidate)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not parsed.hostname
    ):
        raise IngestionValidationError("Bilibili source must be a public BV, AV, or approved URL")
    host = parsed.hostname.lower().rstrip(".")
    if host == "b23.tv":
        if not parsed.path or parsed.path == "/":
            raise IngestionValidationError("Bilibili short URL is invalid")
        return urlunsplit(("https", "b23.tv", parsed.path, "", ""))
    if host not in {"bilibili.com", "www.bilibili.com"}:
        raise IngestionValidationError("Bilibili source host is not approved")
    if not re.fullmatch(r"/video/(?:BV[0-9A-Za-z]+|av[1-9][0-9]*)/?", parsed.path, re.IGNORECASE):
        raise IngestionValidationError("Bilibili source URL is invalid")
    # Keep only the selected-page parameter.  The lower resolver validates
    # its cardinality/range and binds it to the queued command's part.
    query = parsed.query if re.fullmatch(r"(?:p=[1-9][0-9]*)?", parsed.query) else ""
    if parsed.query and not query:
        raise IngestionValidationError("Bilibili source URL has unsupported query parameters")
    return urlunsplit(("https", "www.bilibili.com", parsed.path, query, ""))


def canonical_xiaoe_hls_ref(m3u8_url: str | None) -> tuple[str, str | None]:
    if not m3u8_url:
        raise IngestionValidationError("m3u8_url is required")
    public_url = _safe_public_url(m3u8_url)
    return public_url, _hash(m3u8_url) if public_url != m3u8_url else None


def canonical_xiaoe_page_ref(value: str) -> str:
    """Reduce a public Xiaoe lesson page to its stable product/lesson identity.

    The page URL is not queued because it can contain tracking material and
    the worker must use its operator-configured URL template.  Keeping only
    these two identifiers also makes HTTP retries idempotent across equivalent
    page links while preserving the source resolver's course/lesson contract.
    """
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not host
        or not (host == "xiaoeknow.com" or host.endswith(".xiaoeknow.com")
                or host == "xiaoe-tech.com" or host.endswith(".xiaoe-tech.com"))
    ):
        raise IngestionValidationError("Xiaoe source URL is not an approved course page")
    match = _XIAOE_PAGE_PATH.fullmatch(parsed.path)
    if match is None or parsed.fragment:
        raise IngestionValidationError("Xiaoe source URL is invalid")
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if len(pairs) != 1 or pairs[0][0] != "product_id" or not _STABLE_XIAOE_REF.fullmatch(pairs[0][1]):
        raise IngestionValidationError("Xiaoe source URL must contain exactly one product_id")
    product_id = pairs[0][1]
    lesson_id = match.group(1)
    return f"{product_id}/{lesson_id}"


def _is_public_xiaoe_hls_ref(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and parsed.path.lower().endswith(".m3u8")
    )


def normalize_command(
    *,
    source_type: str,
    source_ref: str,
    part: int | None = None,
    transcript_policy: str = "subtitle_first",
    options: dict | None = None,
    idempotency_key: str | None = None,
    credential_ref: CredentialReference | None = None,
    locator_secret: str | None = None,
    trace_id: str | None = None,
    decision_id: str | None = None,
    allow_legacy_options: bool = False,
    allowed_credential_refs: frozenset[str] | None = None,
    allowed_credential_providers: frozenset[str] | None = None,
) -> ContentIngestionCommand:
    if source_type not in {"bilibili", "xiaoe", "xiaoe_hls"}:
        raise IngestionValidationError("source_type must be bilibili, xiaoe, or xiaoe_hls")
    if transcript_policy != "subtitle_first":
        raise IngestionValidationError("transcript_policy must be subtitle_first")
    if part is not None and part < 1:
        raise IngestionValidationError("part must be greater than zero")
    normalized_options = dict(options or {})
    bad_options = [str(key) for key in normalized_options if any(
        part in str(key).lower() for part in _FORBIDDEN_OPTION_PARTS
    )]
    if bad_options:
        raise IngestionValidationError("secret transport fields are not accepted")
    unexpected = set(normalized_options) - (_LEGACY_ALLOWED_OPTIONS if allow_legacy_options else _ALLOWED_OPTIONS)
    if unexpected:
        raise IngestionValidationError("unsupported ingestion options")
    if source_type == "xiaoe":
        if source_ref.startswith(("https://", "http://")):
            source_ref = canonical_xiaoe_page_ref(source_ref)
        is_direct_hls = _is_public_xiaoe_hls_ref(source_ref)
        if not is_direct_hls and (not _STABLE_XIAOE_REF.fullmatch(source_ref) or "://" in source_ref):
            raise IngestionValidationError("xiaoe source_ref must be a stable course/lesson identity or public HLS URL")
        if credential_ref is None:
            raise IngestionValidationError("xiaoe requires an allowlisted credential_ref")
        if allowed_credential_refs is not None and credential_ref.credential_ref not in allowed_credential_refs:
            raise IngestionValidationError("credential_ref is not allowlisted")
        if allowed_credential_providers is not None and credential_ref.provider not in allowed_credential_providers:
            raise IngestionValidationError("credential provider is not allowlisted")
        if is_direct_hls:
            source_type = "xiaoe_hls"
    elif credential_ref is not None and source_type == "bilibili":
        if allowed_credential_refs is not None and credential_ref.credential_ref not in allowed_credential_refs:
            raise IngestionValidationError("credential_ref is not allowlisted")
        if allowed_credential_providers is not None and credential_ref.provider not in allowed_credential_providers:
            raise IngestionValidationError("credential provider is not allowlisted")
    if source_type == "bilibili":
        source_ref = canonical_bilibili_ref(url=None, bv_id=source_ref)
        parsed_bilibili = urlsplit(source_ref)
        if parsed_bilibili.query:
            selected = parsed_bilibili.query.removeprefix("p=")
            # A page embedded in the canonical URL is the source identity;
            # preserve it when the request used the API default part.
            if selected.isdigit() and part in {None, 1}:
                part = int(selected)
    return ContentIngestionCommand(
        source_type=source_type,
        canonical_source_ref=source_ref,
        part=part,
        transcript_policy=transcript_policy,
        options=normalized_options,
        idempotency_key=idempotency_key or None,
        credential_ref_hash=_hash(credential_ref.credential_ref) if credential_ref else None,
        locator_secret_hash=_hash(locator_secret) if locator_secret else None,
        trace_id=trace_id,
        decision_id=decision_id,
    )


def command_with_legacy_policy(command: ContentIngestionCommand) -> ContentIngestionCommand:
    """Retain policy metadata without allowing arbitrary legacy options through."""
    from stock_content.domain.source_policy import policy_for_source

    policy = policy_for_source(command.source_type)
    return replace(command, options={
        **command.options,
        "source_policy_version": policy.policy_version,
        "retention_class": policy.retention_class,
        "access_classification": policy.access_classification.value,
        "source_artifact_metadata_required": True,
        "enforce_source_policy": True,
    })


__all__ = [
    "IngestionValidationError", "canonical_bilibili_ref", "canonical_xiaoe_hls_ref",
    "canonical_xiaoe_page_ref",
    "command_with_legacy_policy",
    "credential_allowlist_from_environment",
    "normalize_command",
    "request_hash_for",
    "source_identity_hash_for",
]

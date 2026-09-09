"""Public, durable source-page URL projections.

Signed stream locators belong to the runtime materialization only.  This
module deliberately accepts only a small public projection for metadata and
Bundle provenance, including Xiaoe's product identity query which is part of
the course-page address rather than an access credential.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_XIAOE_PAGE_PATH = re.compile(r"^/p/course/video/([A-Za-z0-9_-]{1,160})/?$")
_STABLE_XIAOE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def canonical_public_source_url(source_type: str, value: object) -> str | None:
    """Return a secret-free public page URL or ``None`` when it is invalid.

    Xiaoe course pages require the public ``product_id`` query to distinguish
    a product/lesson pair.  All other query parameters are discarded.  For
    other source types query strings are never durable provenance.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    host = parsed.hostname.lower().rstrip(".")
    if source_type == "xiaoe":
        if not (
            host == "xiaoeknow.com"
            or host.endswith(".xiaoeknow.com")
            or host == "xiaoe-tech.com"
            or host.endswith(".xiaoe-tech.com")
        ):
            return None
        if _XIAOE_PAGE_PATH.fullmatch(parsed.path) is None:
            return None
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        product_ids = [item for key, item in pairs if key == "product_id"]
        if len(product_ids) != 1 or not _STABLE_XIAOE_IDENTIFIER.fullmatch(product_ids[0]):
            return None
        # Reconstruct only the documented public identity parameter.  The
        # resolver may see tracking or signed material URLs, neither of which
        # can cross this boundary.
        return urlunsplit(("https", parsed.netloc, parsed.path, urlencode((("product_id", product_ids[0]),)), ""))
    return urlunsplit(("https", parsed.netloc, parsed.path, "", ""))


__all__ = ["canonical_public_source_url"]

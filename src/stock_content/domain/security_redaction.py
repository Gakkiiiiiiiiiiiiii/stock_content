"""Secret-safe serialization helpers for durable and diagnostic surfaces."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_SENSITIVE_KEY = re.compile(
    r"(?:authorization|cookie|storage[ _-]?state|secret|token|password|api[ _-]?key|signed)", re.I
)
_PUBLIC_OPERATIONAL_KEYS = frozenset({"fencing_token"})
_ASSIGNMENT = re.compile(
    r"(?i)\b(token|sig(?:nature)?|x-amz-(?:signature|credential)|authorization|cookie|key)=[^\s&]+"
)
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")


def redact_text(value: str) -> str:
    """Remove URL query/fragment and common credential forms without echoing values."""
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    value = _BEARER.sub("Bearer <redacted>", value)
    return _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", value)


def redact_for_serialization(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-friendly projection that never retains credential values."""
    if key and key.lower() not in _PUBLIC_OPERATIONAL_KEYS and _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(item_key): redact_for_serialization(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_for_serialization(item) for item in value]
    if is_dataclass(value):
        return redact_for_serialization(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return redact_for_serialization(model_dump())
    return value


def contains_sensitive_value(value: Any) -> bool:
    """True if a durable payload contains a signed locator or credential value."""
    if isinstance(value, Mapping):
        return any(
            (str(key).lower() not in _PUBLIC_OPERATIONAL_KEYS and _SENSITIVE_KEY.search(str(key)))
            or contains_sensitive_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(contains_sensitive_value(item) for item in value)
    if isinstance(value, str):
        parsed = urlsplit(value)
        return bool(parsed.query or parsed.fragment or _BEARER.search(value) or _ASSIGNMENT.search(value))
    return False


__all__ = ["contains_sensitive_value", "redact_for_serialization", "redact_text"]

"""Deterministic content-knowledge-bundle.v1 identity and canonical JSON."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Mapping

CONTRACT = "content-knowledge-bundle.v1"
SCHEMA_VERSION = "1.0.0"
CANONICALIZATION_VERSION = "content-bundle-c14n-v1"
V2_CONTRACT = "content-knowledge-bundle.v2"
V2_SCHEMA_VERSION = "2.0.0"
V2_CANONICALIZATION_VERSION = "content-bundle-c14n-v2"
PUBLIC_STRICT = "PUBLIC_STRICT"
_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT")
_SET_ARRAYS = frozenset({"reason_codes", "warnings", "evidence_refs", "evidence_ids"})


class CanonicalizationError(ValueError):
    pass


def _timestamp(value: str) -> str:
    if not _TIMESTAMP.match(value):
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return value
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z").replace(".000000Z", "Z")


def _number(value: int | float | Decimal) -> int | Decimal:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        raise CanonicalizationError("non-finite numbers are not valid canonical JSON")
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise CanonicalizationError("non-finite numbers are not valid canonical JSON")
    if decimal == 0:
        return 0
    decimal = decimal.normalize()
    return int(decimal) if decimal == decimal.to_integral() else decimal


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def normalize(value: Any, *, parent_key: str | None = None) -> Any:
    """Normalize only JSON values, deliberately rejecting implicit repr/defaults."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _timestamp(unicodedata.normalize("NFC", value))
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return _number(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalizationError("timestamps must be timezone-aware")
        return _timestamp(value.isoformat())
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalizationError("canonical JSON object keys must be strings")
        return {unicodedata.normalize("NFC", key): normalize(item, parent_key=key) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        result = [normalize(item) for item in value]
        if parent_key in _SET_ARRAYS:
            return sorted(result, key=lambda item: canonical_json(item))
        if parent_key == "items":
            return sorted(result, key=_item_key)
        if parent_key == "evidence":
            return sorted(
                result,
                key=lambda item: (
                    (str(item.get("evidence_id", "")), item.get("start_ms", -1), item.get("end_ms", -1))
                    if isinstance(item, dict)
                    else canonical_json(item)
                ),
            )
        return result
    raise CanonicalizationError(f"unsupported non-JSON value: {type(value).__name__}")


def _item_key(item: Any) -> tuple[Any, ...]:
    if not isinstance(item, dict):
        return (canonical_json(item),)
    subject = item.get("subject") or {}
    temporal = item.get("temporal") or {}
    return (
        str(subject.get("type", "")),
        str(subject.get("key", "")),
        str(item.get("predicate", "")),
        str(temporal.get("target_start", "")),
        str(temporal.get("target_end", "")),
        str(item.get("occurrence_id", "")),
        str(item.get("knowledge_id", item.get("claim_id", ""))),
    )


def canonical_json(value: Any) -> str:
    normalized = normalize(value)

    # Decimal needs its canonical spelling, which stdlib json otherwise cannot emit.
    def encode(item: Any) -> str:
        if isinstance(item, Decimal):
            return _decimal_text(item)
        if isinstance(item, list):
            return "[" + ",".join(encode(part) for part in item) + "]"
        if isinstance(item, dict):
            return (
                "{"
                + ",".join(
                    json.dumps(key, ensure_ascii=False, separators=(",", ":")) + ":" + encode(item[key])
                    for key in sorted(item)
                )
                + "}"
            )
        return json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    return encode(normalized)


def sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KnowledgeBundleRequest:
    content_snapshot_id: str
    query: str
    symbol: str
    business_as_of: datetime
    knowledge_as_of: datetime
    availability_as_of: datetime
    minimum_support_status: str
    max_items: int
    policy: str = PUBLIC_STRICT
    policy_version: str = "content-bundle-policy.v1"
    # The default deliberately remains v1.  In particular, this field is not
    # included in ``canonical_request`` for v1, so old bundle ids and hashes
    # replay byte-for-byte.
    contract_version: str = CONTRACT
    # v2 makes topic scope explicit.  It is derived from ``symbol`` so a
    # caller cannot use an all-subject request to silently claim a symbol
    # scoped Bundle (or vice versa).
    subject_scope: str | None = None

    def __post_init__(self) -> None:
        if not self.content_snapshot_id or self.content_snapshot_id.lower() in {"latest", "current", "default"}:
            raise ValueError("content_snapshot_id must be a concrete snapshot id")
        if not self.query or not self.symbol or self.policy != PUBLIC_STRICT or not self.policy_version:
            raise ValueError("bundle request has invalid required binding")
        if self.contract_version not in {CONTRACT, V2_CONTRACT}:
            raise ValueError("unsupported bundle contract version")
        derived_subject_scope = "ALL_SUBJECTS" if self.symbol.strip().upper() == "UNSPECIFIED" else "SUBJECT_ONLY"
        if self.contract_version == CONTRACT and self.subject_scope is not None:
            raise ValueError("subject_scope requires content-knowledge-bundle.v2")
        if self.contract_version == V2_CONTRACT and self.subject_scope not in {None, derived_subject_scope}:
            raise ValueError("SUBJECT_SCOPE_SYMBOL_MISMATCH")
        if not 1 <= self.max_items <= 100:
            raise ValueError("max_items must be between 1 and 100")
        for field in ("business_as_of", "knowledge_as_of", "availability_as_of"):
            value = getattr(self, field)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KnowledgeBundleRequest":
        allowed = {
            "content_snapshot_id",
            "query",
            "symbol",
            "business_as_of",
            "knowledge_as_of",
            "availability_as_of",
            "minimum_support_status",
            "max_items",
            "policy",
            "policy_version",
            "contract_version",
            "subject_scope",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(f"unknown bundle request fields: {', '.join(sorted(unexpected))}")
        required = allowed - {"policy", "policy_version", "contract_version", "subject_scope"}
        missing = sorted(key for key in required if value.get(key) is None)
        if missing:
            raise ValueError(f"bundle request missing {', '.join(missing)}")

        def parse(raw: Any) -> datetime:
            if isinstance(raw, datetime):
                return raw
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))

        return cls(
            **{
                **dict(value),
                "business_as_of": parse(value["business_as_of"]),
                "knowledge_as_of": parse(value["knowledge_as_of"]),
                "availability_as_of": parse(value["availability_as_of"]),
            }
        )

    def canonical_request(self) -> dict[str, Any]:
        request = {
            "content_snapshot_id": self.content_snapshot_id,
            "query": self.query,
            "symbol": self.symbol,
            "business_as_of": self.business_as_of,
            "knowledge_as_of": self.knowledge_as_of,
            "availability_as_of": self.availability_as_of,
            "minimum_support_status": self.minimum_support_status,
            "max_items": self.max_items,
            "policy": self.policy,
            "policy_version": self.policy_version,
        }
        if self.contract_version != CONTRACT:
            request["contract_version"] = self.contract_version
            # A multi-topic source must say so explicitly rather than using a
            # fabricated common subject.  ``UNSPECIFIED`` is a scope marker,
            # never a knowledge-item subject.
            request["subject_scope"] = self.subject_scope or (
                "ALL_SUBJECTS" if self.symbol.strip().upper() == "UNSPECIFIED" else "SUBJECT_ONLY"
            )
        return request

    @property
    def request_hash(self) -> str:
        return sha256(self.canonical_request())

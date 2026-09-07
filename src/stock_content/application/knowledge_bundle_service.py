from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from stock_content.domain.knowledge_bundle import (
    CANONICALIZATION_VERSION,
    CONTRACT,
    SCHEMA_VERSION,
    KnowledgeBundleRequest,
    canonical_json,
    sha256,
)
from stock_content.domain.knowledge_enums import support_rank
from stock_content.ports.knowledge_bundle_repository import KnowledgeBundleAuthority, KnowledgeBundleRepository


@dataclass(frozen=True)
class BundleProducerMetadata:
    service: str
    service_version: str
    git_commit: str
    pipeline_version: str
    contract_checksum: str

    def __post_init__(self) -> None:
        if not all(
            str(getattr(self, name)).strip() and str(getattr(self, name)).strip().lower() != "unknown"
            for name in self.__dataclass_fields__
        ):
            raise ValueError("bundle producer metadata must be injected and complete")


class KnowledgeBundleService:
    def __init__(
        self,
        authority: KnowledgeBundleAuthority,
        repository: KnowledgeBundleRepository,
        producer: BundleProducerMetadata,
    ) -> None:
        self._authority, self._repository, self._producer = authority, repository, producer

    def create(self, request: KnowledgeBundleRequest | Mapping[str, Any]) -> dict[str, Any]:
        request = (
            request if isinstance(request, KnowledgeBundleRequest) else KnowledgeBundleRequest.from_mapping(request)
        )
        source = self._authority.read_bundle_source(request)
        if source is None:
            raise ValueError("SNAPSHOT_NOT_FOUND")
        if not source.get("snapshot_available"):
            raise ValueError("SNAPSHOT_NOT_AVAILABLE")
        if source.get("source_available_from") and _is_after(
            source["source_available_from"], request.availability_as_of, "source_available_from"
        ):
            raise ValueError("SOURCE_NOT_AVAILABLE")
        raw_items = list(source.get("items") or [])
        items = [self._validate_item(item, request) for item in raw_items]
        items.sort(
            key=lambda item: canonical_json(
                [
                    item.get("subject", {}).get("type"),
                    item.get("subject", {}).get("key"),
                    item.get("predicate"),
                    item.get("temporal", {}).get("target_start"),
                    item.get("temporal", {}).get("target_end"),
                    item.get("occurrence_id"),
                    item.get("knowledge_id", item.get("claim_id")),
                ]
            )
        )
        items = items[: request.max_items]
        public_source = self._public_source(dict(source.get("source") or {}))
        # Public output is canonicalized before it crosses either the
        # repository or HTTP boundary.  Hashing alone is insufficient: a
        # consumer must not observe row-order dependent JSON for the same
        # immutable Bundle identity.
        payload = json.loads(canonical_json({
            "contract": CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "request": request.canonical_request(),
            "request_hash": request.request_hash,
            "content_snapshot_id": request.content_snapshot_id,
            "query": request.query,
            "source": public_source,
            "business_as_of": request.business_as_of,
            "knowledge_as_of": request.knowledge_as_of,
            "availability_as_of": request.availability_as_of,
            "items": items,
            "quality": {
                "knowledge_count": len(items),
                "grounded_ratio": 1 if items else 0,
                "numeric_grounded_ratio": 1 if items else 0,
                "transcript_coverage": source.get("transcript_coverage"),
                "warnings": list(source.get("warnings") or []),
            },
            "producer": self._producer.__dict__,
        }))
        digest = sha256(payload)
        bundle = {"bundle_id": "ckb_" + digest.removeprefix("sha256:"), "bundle_hash": digest, **payload}
        return self._repository.insert(bundle)

    def get(self, bundle_id: str) -> dict[str, Any] | None:
        return self._repository.get(bundle_id)

    @staticmethod
    def _public_source(source: dict[str, Any]) -> dict[str, Any]:
        url = source.get("canonical_url")
        if url:
            parts = urlsplit(str(url))
            url = (
                urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
                if parts.scheme in {"http", "https"}
                else None
            )
        public = {
            key: value
            for key, value in {
                "source_type": source.get("source_type"),
                "source_identity_hash": source.get("source_identity_hash"),
                "source_version_id": source.get("source_version_id"),
                "canonical_url": url,
                "title": source.get("title"),
                "author": source.get("author"),
                "published_at": source.get("published_at"),
                "source_content_hash": source.get("source_content_hash"),
            }.items()
            if value is not None and (not isinstance(value, str) or value.strip())
        }
        required = ("source_type", "source_identity_hash", "source_version_id", "canonical_url", "source_content_hash")
        if any(not str(public.get(key, "")).strip() for key in required):
            raise ValueError("INCOMPLETE_SQL_SOURCE_PROVENANCE")
        if public.get("published_at") is not None:
            _parse_aware_instant(public["published_at"], "source_published_at")
        return public

    @staticmethod
    def _validate_item(raw: Mapping[str, Any], request: KnowledgeBundleRequest) -> dict[str, Any]:
        item = dict(raw)
        required = (
            "knowledge_id",
            "claim_id",
            "occurrence_id",
            "statement",
            "subject",
            "predicate",
            "temporal",
            "evidence",
            "verification",
            "support_status",
            "lifecycle_status",
            "grounding_status",
        )
        if any(key not in item for key in required):
            raise ValueError("INCOMPLETE_SQL_AUTHORITY_ROW")
        if (
            item.get("claim_schema_version") != "claim.atomic.v1"
            or item.get("grounding_status") != "GROUNDED"
            or item.get("legacy_grounding_incomplete")
        ):
            raise ValueError("LEGACY_OR_UNGROUNDED_CLAIM")
        if item.get("known_from") and _is_after(item["known_from"], request.knowledge_as_of, "known_from"):
            raise ValueError("KNOWLEDGE_AS_OF_EXCEEDED")
        if item.get("available_from") and _is_after(
            item["available_from"], request.availability_as_of, "available_from"
        ):
            raise ValueError("AVAILABILITY_AS_OF_EXCEEDED")
        if item["lifecycle_status"] != "ACTIVE":
            raise ValueError("LIFECYCLE_NOT_ACTIVE")
        if support_rank(item["support_status"]) < support_rank(request.minimum_support_status):
            raise ValueError("MINIMUM_SUPPORT_STATUS_NOT_MET")
        evidence = list(item["evidence"])
        if not evidence or not any(entry.get("ownership") == "PRIMARY" for entry in evidence):
            raise ValueError("PRIMARY_EVIDENCE_REQUIRED")
        allowed = {
            "knowledge_id", "claim_id", "occurrence_id", "statement", "subject", "predicate", "object",
            "condition", "invalidation", "support_status", "lifecycle_status", "confidence", "temporal",
            "evidence", "verification", "contradiction_group_id", "grounding_status", "known_from",
            "available_from", "claim_schema_version", "legacy_grounding_incomplete",
        }
        if set(item) - allowed:
            raise ValueError("UNDECLARED_BUNDLE_ITEM_FIELD")
        for nested, fields in {
            "subject": ("type", "key"),
            "object": ("value", "unit"),
            "temporal": ("target_start", "target_end", "precision"),
            "verification": ("status", "reason_codes"),
        }.items():
            value = item.get(nested)
            if not isinstance(value, Mapping) or any(field not in value for field in fields):
                raise ValueError(f"INCOMPLETE_BUNDLE_{nested.upper()}")
        evidence_fields = (
            "evidence_id",
            "ownership",
            "artifact_id",
            "segment_id",
            "start_ms",
            "end_ms",
            "quote",
            "quote_hash",
            "modality",
        )
        if any(
            not isinstance(entry, Mapping) or any(field not in entry for field in evidence_fields)
            for entry in evidence
        ):
            raise ValueError("INCOMPLETE_BUNDLE_EVIDENCE")
        item_text_fields = ("knowledge_id", "claim_id", "occurrence_id", "statement", "predicate")
        subject_fields = ("type", "key")
        if (
            any(not isinstance(item[key], str) or not item[key].strip() for key in item_text_fields)
            or any(
                not isinstance(item["subject"][key], str) or not item["subject"][key].strip()
                for key in subject_fields
            )
            or not isinstance(item["verification"]["status"], str)
            or not item["verification"]["status"].strip()
            or not isinstance(item["verification"]["reason_codes"], list)
            or any(not isinstance(code, str) or not code.strip() for code in item["verification"]["reason_codes"])
        ):
            raise ValueError("INVALID_BUNDLE_SEMANTIC_FIELD")
        object_value = item["object"]["value"]
        if isinstance(object_value, (Mapping, list, tuple)):
            raise ValueError("INVALID_BUNDLE_OBJECT_VALUE")
        _parse_aware_instant(item["temporal"]["target_start"], "temporal_target_start")
        _parse_aware_instant(item["temporal"]["target_end"], "temporal_target_end")
        evidence_text_fields = ("evidence_id", "artifact_id", "segment_id", "quote", "quote_hash", "modality")
        evidence_coordinate_fields = ("start_ms", "end_ms")
        for entry in evidence:
            if (
                entry["ownership"] not in {"PRIMARY", "SECONDARY"}
                or any(not isinstance(entry[key], str) or not entry[key].strip() for key in evidence_text_fields)
                or any(
                    isinstance(entry[key], bool) or not isinstance(entry[key], int) or entry[key] < 0
                    for key in evidence_coordinate_fields
                )
                or entry["end_ms"] < entry["start_ms"]
            ):
                raise ValueError("INVALID_BUNDLE_EVIDENCE")
        forbidden = ("cookie", "header", "storage", "signed", "prompt", "raw_response", "pii")
        if any(token in canonical_json(item).lower() for token in forbidden):
            raise ValueError("UNSAFE_BUNDLE_CONTENT")
        for key in (
            "known_from",
            "available_from",
            "claim_schema_version",
            "legacy_grounding_incomplete",
        ):
            item.pop(key, None)
        return item


def _is_after(value: Any, as_of: datetime, field: str) -> bool:
    """Compare source instants, rejecting non-instants rather than guessing.

    SQL ``TIMESTAMPTZ`` comparisons are inclusive.  The service must preserve
    that semantic when an authority returns ISO text with a different but
    equivalent offset (for example ``Z`` versus ``+00:00``).
    """
    return _parse_aware_instant(value, field) > _parse_aware_instant(as_of, "as_of")


def _parse_aware_instant(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"INVALID_{field.upper()}") from error
    else:
        raise ValueError(f"INVALID_{field.upper()}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"INVALID_{field.upper()}")
    return parsed.astimezone(UTC)

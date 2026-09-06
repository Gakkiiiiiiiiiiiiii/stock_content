"""Deterministic source licensing and access policy."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceDecision(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class AccessClassification(StrEnum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    RESTRICTED = "RESTRICTED"
    SENSITIVE = "SENSITIVE"


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    source_type: str
    license_class: str
    allowed_uses: frozenset[str]
    retention_class: str
    access_classification: AccessClassification
    robots_or_terms_reference: str | None = None
    rate_limit_per_minute: int | None = None
    policy_version: str = "source-policy.v1"

    def decide(self, use: str) -> SourceDecision:
        return SourceDecision.ALLOW if use in self.allowed_uses else SourceDecision.DENY

    def validate_rate(self, requests_last_minute: int) -> bool:
        return self.rate_limit_per_minute is None or requests_last_minute < self.rate_limit_per_minute


KNOWN_SOURCE_POLICIES: dict[str, SourcePolicy] = {
    "bilibili": SourcePolicy(
        "bilibili", "public_terms", frozenset({"ingest", "transcribe", "derive"}),
        "standard", AccessClassification.PUBLIC, "https://www.bilibili.com/robots.txt", 30,
    ),
    "xiaoe_hls": SourcePolicy(
        "xiaoe_hls", "licensed_feed", frozenset({"ingest", "transcribe", "derive"}),
        "standard", AccessClassification.RESTRICTED, "licensed-feed:contract", 10,
    ),
}


def policy_for_source(source_type: str) -> SourcePolicy:
    try:
        return KNOWN_SOURCE_POLICIES[source_type]
    except KeyError as exc:
        raise ValueError(f"source type is not governed: {source_type}") from exc


def allow_source(policy: SourcePolicy, use: str, *, requests_last_minute: int = 0) -> bool:
    return policy.decide(use) is SourceDecision.ALLOW and policy.validate_rate(requests_last_minute)


__all__ = [
    "AccessClassification", "KNOWN_SOURCE_POLICIES", "SourceDecision", "SourcePolicy",
    "allow_source", "policy_for_source",
]

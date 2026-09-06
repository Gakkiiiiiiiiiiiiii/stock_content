"""Deterministic source-governance and PII-redaction evidence policies."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from stock_content.domain.source_policy import SourcePolicy

GOVERNANCE_EVIDENCE_VERSION = "source-governance-evidence.v1"
PII_REDACTION_POLICY_VERSION = "pii-redaction.v1"


class GovernanceEvidenceError(ValueError):
    """Required source-governance evidence is absent, malformed, or inconsistent."""


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    detected_types: tuple[str, ...]


_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CN_NATIONAL_ID", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("CN_MOBILE", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
)


def redact_pii(text: str) -> RedactionResult:
    """Replace supported direct identifiers without retaining their original values."""
    redacted = str(text)
    detected: list[str] = []
    for category, pattern in _PII_PATTERNS:
        redacted, count = pattern.subn(f"[REDACTED:{category}]", redacted)
        if count:
            detected.append(category)
    return RedactionResult(redacted, tuple(detected))


def governance_evidence_for(policy: SourcePolicy) -> dict[str, str | bool]:
    """Build the immutable policy evidence attached to every governed source."""
    values = {
        "source_type": policy.source_type,
        "source_policy_version": policy.policy_version,
        "license_class": policy.license_class,
        "robots_or_terms_reference": policy.robots_or_terms_reference,
        "retention_class": policy.retention_class,
        "access_classification": policy.access_classification.value,
    }
    missing = sorted(name for name, value in values.items() if not str(value or "").strip())
    if missing:
        raise GovernanceEvidenceError(f"source policy lacks auditable governance fields: {missing}")
    return {
        "evidence_version": GOVERNANCE_EVIDENCE_VERSION,
        **values,
        # This is deliberately explicit even when it equals access class: a
        # source's access boundary and its privacy review are independently
        # auditable policy decisions.
        "privacy_classification": policy.access_classification.value,
        "pii_redaction_policy_version": PII_REDACTION_POLICY_VERSION,
        "pii_redaction_required": True,
    }


def validate_governance_evidence(metadata: dict[str, Any], *, source_type: str | None = None) -> dict[str, Any]:
    """Fail closed unless immutable source metadata contains complete evidence."""
    evidence = dict(metadata.get("governance_evidence") or {})
    required = (
        "evidence_version",
        "source_type",
        "source_policy_version",
        "license_class",
        "robots_or_terms_reference",
        "retention_class",
        "access_classification",
        "privacy_classification",
        "pii_redaction_policy_version",
    )
    missing = sorted(name for name in required if not str(evidence.get(name) or "").strip())
    if missing:
        raise GovernanceEvidenceError(f"source governance evidence is missing: {missing}")
    if evidence["evidence_version"] != GOVERNANCE_EVIDENCE_VERSION:
        raise GovernanceEvidenceError("source governance evidence version is unsupported")
    if evidence["pii_redaction_policy_version"] != PII_REDACTION_POLICY_VERSION:
        raise GovernanceEvidenceError("source PII-redaction evidence version is unsupported")
    if evidence.get("pii_redaction_required") is not True:
        raise GovernanceEvidenceError("source governance evidence must require PII redaction")
    if source_type is not None and evidence["source_type"] != source_type:
        raise GovernanceEvidenceError("source governance evidence source_type does not match artifact")
    for field in ("source_policy_version", "retention_class", "access_classification"):
        value = metadata.get(field)
        if value is not None and value != evidence[field]:
            raise GovernanceEvidenceError(f"source governance evidence {field} does not match artifact metadata")
    return evidence


__all__ = [
    "GOVERNANCE_EVIDENCE_VERSION",
    "PII_REDACTION_POLICY_VERSION",
    "GovernanceEvidenceError",
    "RedactionResult",
    "governance_evidence_for",
    "redact_pii",
    "validate_governance_evidence",
]

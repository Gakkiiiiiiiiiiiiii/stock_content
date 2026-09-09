"""The reviewed, multi-topic content knowledge Bundle v2 projection.

v1 is a locked replay contract.  v2 is intentionally a separate wire shape:
it carries the semantics which a consumer needs to distinguish a source
opinion from a verified fact, and an audit trail from public-strict support.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .knowledge_bundle import canonical_json
from .knowledge_enums import support_rank

PRIMARY_DOMAINS = frozenset(
    {
        "PORTFOLIO_RISK_MANAGEMENT",
        "FINANCIAL_SECTOR_CAPITAL_POLICY",
        "AI_INFERENCE_COMPUTE",
        "INFORMATION_INFRASTRUCTURE_POLICY",
        "CENTRAL_BANK_GOLD_RESERVES",
        "UNKNOWN",
    }
)
ATTRIBUTED_NATURES = frozenset({"OPINION", "FORECAST", "CAUSAL_THESIS"})
SOURCE_GRADES = frozenset({"PRIMARY", "SECONDARY", "SOURCE_ASSERTION", "UNKNOWN"})
EXTERNAL_TRUTH_STATUSES = frozenset({"NOT_CHECKED", "NOT_FOUND", "EXTERNALLY_VERIFIED", "EXTERNAL_CONFLICT"})
REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
_COURSE_PREFIX = re.compile(r"^(?:(?:本)?课程|视频)(?:提出|设置|建议|强调|认为|指出)[：:，,、\s]*")
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:\s*[%％]|\s*(?:万亿|亿|万|日|月|年|bps|倍|EFLOPS))?")


def atomic_statement(value: Any) -> str:
    """Remove presentation framing, without removing source attribution data."""
    statement = _COURSE_PREFIX.sub("", str(value or "").strip())
    if not statement:
        raise ValueError("EMPTY_ATOMIC_STATEMENT")
    return statement


def is_numeric_claim(item: Mapping[str, Any]) -> bool:
    declared = item.get("numeric_claim")
    if isinstance(declared, bool):
        return declared
    object_value = ((item.get("object") or {}).get("value"))
    return isinstance(object_value, (int, float)) and not isinstance(object_value, bool) or bool(
        _NUMBER.search(str(item.get("statement") or ""))
    )


def public_strict_eligible(item: Mapping[str, Any], minimum_support_status: str) -> bool:
    if item.get("lifecycle_status") != "ACTIVE":
        return False
    review = item.get("occurrence_review") or {}
    if review.get("status") == REVIEW_REQUIRED:
        return False
    # An unresolved contradiction is audit material, not a qualified public
    # support.  The consumer enforces the same fail-closed boundary.
    if str(item.get("contradiction_group_id") or "").strip():
        return False
    if item.get("grounding_status") != "GROUNDED":
        return False
    if support_rank(item.get("support_status")) < support_rank(minimum_support_status):
        return False
    # A secondary fact may remain visible to audit, but it cannot be made a
    # public-strict support merely because the transcript mentions it.
    return any(
        evidence.get("ownership") == "PRIMARY" and evidence.get("modality") == "transcript"
        for evidence in item.get("evidence") or []
        if isinstance(evidence, Mapping)
    )


def externally_grounded(item: Mapping[str, Any]) -> bool:
    """Return whether an item is an externally grounded fact.

    Evidence ownership describes who owns the video/transcript artifact.  It
    is not a statement about whether a policy, reserve, or market number came
    from a primary external source.  Those are intentionally separate fields.
    """
    return (
        item.get("source_grade") == "PRIMARY"
        and item.get("external_truth_status") == "EXTERNALLY_VERIFIED"
    )


def validate_v2_item(raw: Mapping[str, Any], *, minimum_support_status: str) -> dict[str, Any]:
    """Validate and normalize one SQL-authoritative v2 occurrence projection."""
    item = dict(raw)
    required = {
        "knowledge_id", "claim_id", "occurrence_id", "statement", "subject", "predicate", "object",
        "primary_domain", "claim_nature", "attribution", "detail", "temporal", "evidence",
        "occurrence_review", "support_status", "lifecycle_status", "verification", "grounding_status",
    }
    if required - set(item):
        raise ValueError("INCOMPLETE_SQL_AUTHORITY_ROW")
    if item.get("claim_schema_version") != "claim.atomic.v1" or item.get("legacy_grounding_incomplete"):
        raise ValueError("LEGACY_OR_UNGROUNDED_CLAIM")
    if item.get("lifecycle_status") not in {"ACTIVE", "EXTRACTED"}:
        raise ValueError("LIFECYCLE_NOT_ACTIVE")
    item["statement"] = atomic_statement(item["statement"])
    if item["primary_domain"] not in PRIMARY_DOMAINS:
        raise ValueError("INVALID_PRIMARY_DOMAIN")
    if not isinstance(item["claim_nature"], str) or not item["claim_nature"].strip():
        raise ValueError("INVALID_CLAIM_NATURE")
    subject = item["subject"]
    if not isinstance(subject, Mapping) or not all(str(subject.get(key) or "").strip() for key in ("type", "key")):
        raise ValueError("INVALID_BUNDLE_SEMANTIC_FIELD")
    if str(subject["key"]).upper() == "UNSPECIFIED":
        raise ValueError("UNSPECIFIED_IS_SCOPE_NOT_SUBJECT")
    attribution = item["attribution"]
    if not isinstance(attribution, Mapping) or not isinstance(attribution.get("attributed"), bool):
        raise ValueError("INVALID_ATTRIBUTION")
    if item["claim_nature"] in ATTRIBUTED_NATURES and not attribution["attributed"]:
        raise ValueError("ATTRIBUTION_REQUIRED")
    if attribution["attributed"] and not str(attribution.get("source_label") or "").strip():
        raise ValueError("ATTRIBUTION_SOURCE_REQUIRED")
    if item.get("source_grade") not in SOURCE_GRADES:
        raise ValueError("INVALID_SOURCE_GRADE")
    if item.get("external_truth_status") not in EXTERNAL_TRUTH_STATUSES:
        raise ValueError("INVALID_EXTERNAL_TRUTH_STATUS")
    if item["source_grade"] in {"SECONDARY", "UNKNOWN"} and not attribution["attributed"]:
        raise ValueError("UNVERIFIED_EXTERNAL_FACT_MUST_BE_ATTRIBUTED")
    detail = item["detail"]
    if not isinstance(detail, Mapping):
        raise ValueError("INVALID_KNOWLEDGE_DETAIL")
    allowed_detail = {"explanation", "mechanism", "procedure", "formula", "example", "scope", "risks"}
    detail_values = [str(value or "").strip() for value in detail.values() if str(value or "").strip()]
    if set(detail) - allowed_detail or not detail_values:
        raise ValueError("INVALID_KNOWLEDGE_DETAIL")
    # A copied statement is not an explanation.  v2 deliberately fails
    # closed until the producer supplies at least one substantive detail.
    if all(atomic_statement(value) == item["statement"] for value in detail_values):
        raise ValueError("NONTRIVIAL_KNOWLEDGE_DETAIL_REQUIRED")
    temporal = item["temporal"]
    if not isinstance(temporal, Mapping) or temporal.get("kind") not in {
        "EVENT", "AS_OF", "FORECAST_TARGET", "RECURRING_RULE", "UNKNOWN"
    }:
        raise ValueError("INVALID_TEMPORAL_BINDING")
    if temporal["kind"] == "UNKNOWN":
        if temporal.get("start") or temporal.get("end") or temporal.get("as_of") or temporal.get("rule"):
            raise ValueError("UNKNOWN_TEMPORAL_MUST_NOT_INVENT_DATE")
        if temporal.get("explicitly_unknown") is not True:
            raise ValueError("UNKNOWN_TEMPORAL_MUST_BE_EXPLICIT")
    elif temporal.get("explicitly_unknown"):
        raise ValueError("KNOWN_TEMPORAL_CANNOT_BE_UNKNOWN")
    if temporal["kind"] == "FORECAST_TARGET" and not (
        temporal.get("start") or temporal.get("end") or temporal.get("label")
    ):
        raise ValueError("FORECAST_TARGET_REQUIRED")
    if temporal["kind"] == "AS_OF" and not (temporal.get("as_of") or temporal.get("label")):
        raise ValueError("AS_OF_REQUIRED")
    if temporal["kind"] == "RECURRING_RULE" and not str(temporal.get("rule") or "").strip():
        raise ValueError("RECURRING_RULE_REQUIRED")
    review = item["occurrence_review"]
    if not isinstance(review, Mapping) or review.get("status") not in {
        "NOT_REQUIRED", REVIEW_REQUIRED, "REVIEWED"
    } or not isinstance(review.get("reason_codes"), list):
        raise ValueError("INVALID_OCCURRENCE_REVIEW")
    if review["status"] == REVIEW_REQUIRED and not review["reason_codes"]:
        raise ValueError("REVIEW_REASON_REQUIRED")
    if item.get("lifecycle_status") == "EXTRACTED" and review["status"] != REVIEW_REQUIRED:
        # EXTRACTED is admitted only as review-blocked audit material for
        # conservative quality accounting; it can never become a public item.
        raise ValueError("LIFECYCLE_NOT_ACTIVE")
    evidence = item["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("PRIMARY_EVIDENCE_REQUIRED")
    for entry in evidence:
        _validate_evidence(entry)
    if not any(entry["ownership"] == "PRIMARY" for entry in evidence):
        raise ValueError("PRIMARY_EVIDENCE_REQUIRED")
    if not isinstance(item["verification"], Mapping) or not isinstance(item["verification"].get("reason_codes"), list):
        raise ValueError("INCOMPLETE_BUNDLE_VERIFICATION")
    return item


def _validate_evidence(entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError("INCOMPLETE_BUNDLE_EVIDENCE")
    required = {"evidence_id", "ownership", "modality", "artifact_id", "artifact_hash", "locator", "content"}
    if required - set(entry) or entry.get("ownership") not in {"PRIMARY", "SECONDARY"}:
        raise ValueError("INCOMPLETE_BUNDLE_EVIDENCE")
    if entry.get("modality") not in {"transcript", "frame", "ocr", "vision"}:
        raise ValueError("INVALID_BUNDLE_EVIDENCE")
    if not all(str(entry.get(key) or "").strip() for key in ("evidence_id", "artifact_id", "artifact_hash")):
        raise ValueError("INVALID_BUNDLE_EVIDENCE")
    if not str(entry["artifact_hash"]).startswith("sha256:"):
        raise ValueError("INVALID_BUNDLE_EVIDENCE")
    locator = entry["locator"]
    if (
        not isinstance(locator, Mapping)
        or not isinstance(locator.get("start_ms"), int)
        or not isinstance(locator.get("end_ms"), int)
    ):
        raise ValueError("INVALID_BUNDLE_EVIDENCE")
    if locator["start_ms"] < 0 or locator["end_ms"] < locator["start_ms"]:
        raise ValueError("INVALID_BUNDLE_EVIDENCE")
    if entry["modality"] in {"frame", "ocr", "vision"} and not str(locator.get("frame_id") or "").strip():
        raise ValueError("FRAME_ID_REQUIRED")
    if entry["modality"] in {"ocr", "vision"}:
        model = entry.get("model")
        if (
            not isinstance(model, Mapping)
            or not str(model.get("name") or "").strip()
            or not str(model.get("version") or "").strip()
        ):
            raise ValueError("EVIDENCE_MODEL_IDENTITY_REQUIRED")


def conservative_quality(
    candidates: list[dict[str, Any]],
    published: list[dict[str, Any]],
    warnings: list[Any],
    *,
    minimum_support_status: str,
) -> dict[str, Any]:
    """Expose audit exclusions separately from the published-item quality.

    ``candidate_count`` deliberately remains the SQL/audit population.  The
    grounded and numeric ratios, however, describe exactly the deterministic,
    max-items-limited wire payload a consumer receives.  Otherwise a consumer
    could not reconcile a ratio with ``items`` after truncation.
    """
    eligible = [
        item for item in candidates if public_strict_eligible(item, minimum_support_status)
    ]
    numeric_published = [item for item in published if is_numeric_claim(item)]
    grounded_published = [item for item in published if externally_grounded(item)]
    numeric_grounded_published = [
        item for item in numeric_published if externally_grounded(item)
    ]
    review_count = sum((item.get("occurrence_review") or {}).get("status") == REVIEW_REQUIRED for item in candidates)
    conflict_count = sum(
        bool((item.get("occurrence_review") or {}).get("reason_codes")) or bool(item.get("contradiction_group_id"))
        for item in candidates
    )
    secondary_only_count = sum(item.get("source_grade") == "SECONDARY" for item in candidates)
    external_truth_not_checked_count = sum(
        item.get("external_truth_status") == "NOT_CHECKED" for item in candidates
    )
    calculated_warnings = {str(value) for value in warnings if str(value).strip()}
    if review_count:
        calculated_warnings.add("HUMAN_REVIEW_REQUIRED_ITEMS_EXCLUDED")
    if conflict_count:
        calculated_warnings.add("CONFLICT_OR_REVIEW_EVIDENCE_PRESENT")
    excluded_count = len(candidates) - len(eligible)
    truncated_count = len(eligible) - len(published)
    if excluded_count:
        calculated_warnings.add("PUBLIC_STRICT_CANDIDATES_EXCLUDED")
    if truncated_count:
        calculated_warnings.add("PUBLIC_STRICT_ELIGIBLE_ITEMS_TRUNCATED")
    if secondary_only_count:
        calculated_warnings.add("SECONDARY_EXTERNAL_FACTS_PRESENT")
    if external_truth_not_checked_count:
        calculated_warnings.add("EXTERNAL_TRUTH_NOT_CHECKED_ITEMS_PRESENT")
    return {
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "excluded_candidate_count": excluded_count,
        "truncated_candidate_count": truncated_count,
        "knowledge_count": len(published),
        "grounded_count": len(grounded_published),
        "numeric_candidate_count": len(numeric_published),
        "numeric_grounded_count": len(numeric_grounded_published),
        "human_review_required_count": review_count,
        "conflict_count": conflict_count,
        "secondary_only_count": secondary_only_count,
        "external_truth_not_checked_count": external_truth_not_checked_count,
        "grounded_ratio": len(grounded_published) / len(published) if published else 0,
        "numeric_grounded_ratio": (
            len(numeric_grounded_published) / len(numeric_published) if numeric_published else 0
        ),
        "warnings": sorted(calculated_warnings),
    }


def sort_v2_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: canonical_json(
            [
                item["primary_domain"],
                item["subject"]["type"],
                item["subject"]["key"],
                item["predicate"],
                item["occurrence_id"],
            ]
        ),
    )

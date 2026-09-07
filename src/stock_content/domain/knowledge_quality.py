"""Deterministic, fail-closed scoring for human-labelled knowledge quality.

This module deliberately consumes JSON files only.  It never reads media,
contacts a model, or resolves a secret-artifact reference.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

METRICS = (
    "precision",
    "recall",
    "entity_accuracy",
    "numeric_unit_grounding",
    "temporal_accuracy",
    "evidence_span_accuracy",
    "duplicate_rate",
    "unsupported_claim_rate",
)
HIGHER_IS_BETTER = set(METRICS[:6])
THRESHOLDS = {
    "precision": 0.90,
    "recall": 0.80,
    "entity_accuracy": 0.95,
    "numeric_unit_grounding": 1.0,
    "temporal_accuracy": 0.90,
    "evidence_span_accuracy": 0.95,
    "duplicate_rate": 0.05,
    "unsupported_claim_rate": 0.0,
}
REQUIRED_CLAIM_STRATA = {"earnings", "risk", "condition", "prediction", "numeric", "date", "industry", "contradiction"}
REQUIRED_SUBTITLE_STRATA = {"manual", "auto", "asr"}
SECRET_KEYS = {"secret", "secret_value", "url", "media_url", "protected_url", "authorization_token", "token"}


class KnowledgeQualityError(ValueError):
    """Input is insufficient or unsafe to support a quality decision."""


def canonical_hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KnowledgeQualityError("JSON_OBJECT_REQUIRED")
    return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load local annotation files, refusing paths outside ``annotations/``."""
    manifest_path = Path(path).resolve()
    manifest = load_json(manifest_path)
    annotations_root = (manifest_path.parent / "annotations").resolve()
    partitions = manifest.get("partitions", {})
    if not isinstance(partitions, Mapping):
        return manifest
    for values in partitions.values():
        if not isinstance(values, list):
            continue
        for record in values:
            if not isinstance(record, dict) or "annotation_file" not in record:
                continue
            relative = record["annotation_file"]
            if not isinstance(relative, str):
                raise KnowledgeQualityError("ANNOTATION_FILE_INVALID")
            resolved = (manifest_path.parent / relative).resolve()
            if annotations_root not in resolved.parents or resolved.suffix != ".json":
                raise KnowledgeQualityError("ANNOTATION_FILE_PATH_FORBIDDEN")
            annotation = load_json(resolved)
            if set(annotation) != {"claims"} or not isinstance(annotation["claims"], list):
                raise KnowledgeQualityError("ANNOTATION_FILE_SCHEMA_INVALID")
            record["claims"] = annotation["claims"]
    return manifest


def _blocked(*codes: str) -> dict[str, Any]:
    return {"status": "BLOCKED", "release_ready": False, "blockers": sorted(set(codes))}


def _safe_source(record: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for key, value in record.items():
        lowered = key.lower()
        if lowered in SECRET_KEYS or ("secret" in lowered and lowered != "secret_artifact_ref"):
            issues.append("SECRET_OR_MEDIA_VALUE_FORBIDDEN")
        if isinstance(value, str) and ("//" in value or "token=" in value.lower()):
            issues.append("SECRET_OR_MEDIA_VALUE_FORBIDDEN")
    return issues


def validate_manifest(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate corpus metadata without dereferencing its annotation files."""
    blockers: list[str] = []
    if manifest.get("schema_version") != "knowledge-golden-manifest.v1":
        blockers.append("MANIFEST_SCHEMA_VERSION_INVALID")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != {"train", "dev", "frozen_oos"}:
        blockers.append("PARTITIONS_REQUIRED")
        return [], blockers
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    hashes: set[str] = set()
    for partition in ("train", "dev", "frozen_oos"):
        values = partitions[partition]
        if not isinstance(values, list):
            blockers.append("PARTITION_NOT_LIST")
            continue
        for item in values:
            if not isinstance(item, Mapping):
                blockers.append("SOURCE_RECORD_INVALID")
                continue
            record = dict(item)
            blockers.extend(_safe_source(record))
            required = {
                "source_identity",
                "content_hash",
                "platform",
                "subtitle_origin",
                "annotation_version",
                "license",
                "authorization",
                "retention_policy",
            }
            if not required.issubset(record):
                blockers.append("SOURCE_LINEAGE_MISSING")
                continue
            if (
                record["platform"] not in {"bilibili", "xiaoe"}
                or record["subtitle_origin"] not in REQUIRED_SUBTITLE_STRATA
            ):
                blockers.append("SOURCE_STRATUM_INVALID")
            if not str(record["content_hash"]).startswith("sha256:"):
                blockers.append("CONTENT_HASH_INVALID")
            if record["platform"] == "xiaoe" and not isinstance(record.get("secret_artifact_ref"), str):
                blockers.append("XIAOE_SECRET_ARTIFACT_REF_REQUIRED")
            identity, content_hash = str(record["source_identity"]), str(record["content_hash"])
            if identity in identities or content_hash in hashes:
                blockers.append("PARTITION_SOURCE_LEAKAGE")
            identities.add(identity)
            hashes.add(content_hash)
            record["partition"] = partition
            records.append(record)
    expected_hash = manifest.get("annotation_set_hash")
    annotation_descriptor = [
        {
            "source_identity": record["source_identity"],
            "content_hash": record["content_hash"],
            "annotation_version": record["annotation_version"],
            "claims": record.get("claims", []),
        }
        for record in records
    ]
    if expected_hash != canonical_hash(annotation_descriptor):
        blockers.append("ANNOTATION_SET_HASH_MISMATCH")
    if not manifest.get("annotation_set_version"):
        blockers.append("ANNOTATION_SET_VERSION_REQUIRED")
    return records, blockers


def _claims(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for record in records:
        annotations = record.get("claims", [])
        if not isinstance(annotations, list):
            raise KnowledgeQualityError("ANNOTATION_CLAIMS_INVALID")
        for item in annotations:
            if not isinstance(item, Mapping) or not isinstance(item.get("claim_id"), str):
                raise KnowledgeQualityError("ANNOTATION_CLAIM_ID_REQUIRED")
            claims.append(
                {
                    **item,
                    "source_identity": record["source_identity"],
                    "partition": record["partition"],
                    "platform": record["platform"],
                    "subtitle_origin": record["subtitle_origin"],
                }
            )
    return claims


def _score(
    golden: Sequence[Mapping[str, Any]], output: Mapping[str, Any]
) -> tuple[dict[str, float | None], dict[str, Any], list[str]]:
    blockers: list[str] = []
    claims = output.get("claims")
    if not isinstance(claims, list):
        return {metric: None for metric in METRICS}, {}, ["PIPELINE_CLAIMS_REQUIRED"]
    golden_by_id = {str(item["claim_id"]): item for item in golden}
    seen_matches: set[str] = set()
    correct = Counter()
    denominators = Counter()
    duplicates = unsupported = 0
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in claims:
        if not isinstance(item, Mapping):
            blockers.append("PIPELINE_CLAIM_INVALID")
            continue
        if (
            item.get("claim_schema_version") != "claim.atomic.v1"
            or item.get("grounding_status") != "GROUNDED"
            or item.get("legacy_grounding_incomplete") is True
        ):
            blockers.append("LEGACY_OR_UNGROUNDED_CLAIM")
        golden_id = item.get("golden_claim_id")
        matched = golden_by_id.get(str(golden_id))
        duplicate = bool(item.get("duplicate", False))
        unsupported_value = bool(item.get("unsupported", False))
        duplicates += duplicate
        unsupported += unsupported_value
        if matched and not duplicate:
            seen_matches.add(str(golden_id))
            for dimension, metric in (
                ("entity", "entity_accuracy"),
                ("numeric", "numeric_unit_grounding"),
                ("date", "temporal_accuracy"),
                ("evidence_span", "evidence_span_accuracy"),
            ):
                if dimension in matched.get("required_dimensions", []):
                    denominators[metric] += 1
                    correct[metric] += bool(item.get(metric, False))
            for stratum in set(matched.get("strata", [])) | {matched["platform"], matched["subtitle_origin"]}:
                strata[str(stratum)].append(item)
    total = len(claims)
    metrics: dict[str, float | None] = {
        "precision": len(seen_matches) / total if total else None,
        "recall": len(seen_matches) / len(golden) if golden else None,
        "entity_accuracy": correct["entity_accuracy"] / denominators["entity_accuracy"]
        if denominators["entity_accuracy"]
        else None,
        "numeric_unit_grounding": correct["numeric_unit_grounding"] / denominators["numeric_unit_grounding"]
        if denominators["numeric_unit_grounding"]
        else None,
        "temporal_accuracy": correct["temporal_accuracy"] / denominators["temporal_accuracy"]
        if denominators["temporal_accuracy"]
        else None,
        "evidence_span_accuracy": correct["evidence_span_accuracy"] / denominators["evidence_span_accuracy"]
        if denominators["evidence_span_accuracy"]
        else None,
        "duplicate_rate": duplicates / total if total else None,
        "unsupported_claim_rate": unsupported / total if total else None,
    }
    return metrics, {key: len(value) for key, value in sorted(strata.items())}, blockers


def evaluate(
    manifest: Mapping[str, Any], candidate: Mapping[str, Any] | None, baseline: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    records, blockers = validate_manifest(manifest)
    result: dict[str, Any] = {
        "contract": "knowledge-quality-report.v1",
        "schema_version": "1.0.0",
        "annotation_set_version": manifest.get("annotation_set_version"),
        "annotation_set_hash": manifest.get("annotation_set_hash"),
        "partition": "frozen_oos",
        "status": "BLOCKED",
        "release_ready": False,
        "metrics": {metric: None for metric in METRICS},
        "strata": {},
        "blockers": [],
    }
    if candidate is None:
        return {**result, **_blocked(*blockers, "MISSING_GOLDEN_OOS", "CANDIDATE_OUTPUT_REQUIRED")}
    golden = _claims(records)
    oos = [claim for claim in golden if claim["partition"] == "frozen_oos"]
    platforms = Counter(item["platform"] for item in oos)
    all_strata = set().union(*(set(item.get("strata", [])) for item in oos)) if oos else set()
    all_strata |= {item["subtitle_origin"] for item in oos}
    if len(oos) < 100:
        blockers.append("MISSING_GOLDEN_OOS")
    if platforms["bilibili"] < 3 or platforms["xiaoe"] < 3:
        blockers.append("MISSING_REQUIRED_PLATFORMS")
    missing_strata = (REQUIRED_CLAIM_STRATA | REQUIRED_SUBTITLE_STRATA) - all_strata
    if missing_strata:
        blockers.extend("MISSING_REQUIRED_STRATUM:" + value for value in sorted(missing_strata))
    required_metadata = {
        "pipeline_version",
        "pipeline_hash",
        "prompt_version",
        "prompt_hash",
        "model_version",
        "model_hash",
        "asr_version",
        "asr_hash",
        "segmenter_version",
        "segmenter_hash",
        "grounder_version",
        "grounder_hash",
        "schema_version",
        "schema_hash",
        "annotation_set_version",
        "annotation_set_hash",
    }
    if not required_metadata.issubset(candidate):
        blockers.append("PIPELINE_LINEAGE_MISSING")
    if candidate.get("annotation_set_version") != manifest.get("annotation_set_version") or candidate.get(
        "annotation_set_hash"
    ) != manifest.get("annotation_set_hash"):
        blockers.append("ANNOTATION_VERSION_OR_HASH_MISMATCH")
    metrics, strata, score_blockers = _score(oos, candidate)
    blockers.extend(score_blockers)
    for metric, value in metrics.items():
        if value is None:
            blockers.append("UNDEFINED_DENOMINATOR:" + metric)
        elif (metric in HIGHER_IS_BETTER and value < THRESHOLDS[metric]) or (
            metric not in HIGHER_IS_BETTER and value > THRESHOLDS[metric]
        ):
            blockers.append("THRESHOLD_NOT_MET:" + metric)
    if baseline is not None:
        if baseline.get("annotation_set_version") != manifest.get("annotation_set_version") or baseline.get(
            "annotation_set_hash"
        ) != manifest.get("annotation_set_hash"):
            blockers.append("BASELINE_ANNOTATION_VERSION_OR_HASH_MISMATCH")
        else:
            baseline_metrics, _, baseline_blockers = _score(oos, baseline)
            blockers.extend("BASELINE_" + code for code in baseline_blockers)
            for metric in METRICS:
                current, prior = metrics[metric], baseline_metrics[metric]
                if (
                    current is not None
                    and prior is not None
                    and (
                        (metric in HIGHER_IS_BETTER and current < prior)
                        or (metric not in HIGHER_IS_BETTER and current > prior)
                    )
                ):
                    blockers.append("REGRESSION:" + metric)
    result.update(metrics=metrics, strata=strata, blockers=sorted(set(blockers)))
    if not result["blockers"]:
        result.update(status="PASS", release_ready=True)
    return result


__all__ = ["KnowledgeQualityError", "canonical_hash", "evaluate", "load_json", "load_manifest", "validate_manifest"]

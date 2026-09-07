from __future__ import annotations

# ruff: noqa: E501
import json
from hashlib import sha256
from pathlib import Path

import yaml

from scripts.evaluate_knowledge_quality import main
from stock_content.domain.knowledge_quality import canonical_hash, evaluate, load_manifest


def _manifest(*, duplicate=False, tampered=False):
    records = []
    labels = ["earnings", "risk", "condition", "prediction", "numeric", "date", "industry", "contradiction"]
    subtitles = ["manual", "auto", "asr"]
    for index in range(6):
        records.append(
            {
                "source_identity": "source-0" if duplicate and index == 1 else f"source-{index}",
                "content_hash": "sha256:" + f"{index + 1:064x}",
                "platform": "bilibili" if index < 3 else "xiaoe",
                "subtitle_origin": subtitles[index % 3],
                "annotation_version": "v1",
                "license": "public" if index < 3 else "authorized",
                "authorization": "public" if index < 3 else "contract-on-file",
                "retention_policy": "policy.v1",
                **({"secret_artifact_ref": "vault-ref-" + str(index)} if index >= 3 else {}),
                "claims": [
                    {
                        "claim_id": f"claim-{index}-{claim}",
                        "strata": [labels[(index * 20 + claim) % len(labels)]],
                        "required_dimensions": ["entity", "numeric", "date", "evidence_span"],
                    }
                    for claim in range(20)
                ],
            }
        )
    descriptor = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "platform",
                "subtitle_origin",
                "license",
                "authorization",
                "retention_policy",
                "secret_artifact_ref",
            }
        }
        for record in records
    ]
    # The production fingerprint uses just identity, content hash, version and claims.
    descriptor = [
        {key: record[key] for key in ("source_identity", "content_hash", "annotation_version", "claims")}
        for record in records
    ]
    return {
        "schema_version": "knowledge-golden-manifest.v1",
        "annotation_set_version": "v1" if not tampered else "v2",
        "annotation_set_hash": canonical_hash(descriptor),
        "partitions": {"train": [], "dev": [], "frozen_oos": records},
    }


def _output(manifest, *, precision_error=False, missing_metadata=False):
    claims = []
    for record in manifest["partitions"]["frozen_oos"]:
        for golden in record["claims"]:
            claims.append(
                {
                    "golden_claim_id": golden["claim_id"],
                    "claim_schema_version": "claim.atomic.v1",
                    "grounding_status": "GROUNDED",
                    "legacy_grounding_incomplete": False,
                    "entity_accuracy": True,
                    "numeric_unit_grounding": True,
                    "temporal_accuracy": True,
                    "evidence_span_accuracy": True,
                    "duplicate": False,
                    "unsupported": False,
                }
            )
    if precision_error:
        claims.extend({**claims[0], "golden_claim_id": "not-present-" + str(index)} for index in range(20))
    result = {
        key: "v1"
        for key in (
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
        )
    }
    result.update(
        annotation_set_version=manifest["annotation_set_version"],
        annotation_set_hash=manifest["annotation_set_hash"],
        claims=claims,
    )
    if missing_metadata:
        result.pop("model_hash")
    return result


def test_frozen_oos_pass_is_deterministic_and_compares_baseline():
    manifest = _manifest()
    output = _output(manifest)
    first = evaluate(manifest, output, output)
    assert first == evaluate(manifest, output, output)
    assert first["status"] == "PASS"
    assert first["metrics"]["numeric_unit_grounding"] == 1.0
    assert first["strata"]["xiaoe"] == 60


def test_failure_boundaries_undefined_and_release_prerequisites():
    manifest = _manifest()
    failed = evaluate(manifest, _output(manifest, precision_error=True))
    assert "THRESHOLD_NOT_MET:precision" in failed["blockers"]
    assert (
        "REGRESSION:precision"
        in evaluate(manifest, _output(manifest, precision_error=True), _output(manifest))["blockers"]
    )
    empty = evaluate(
        {
            "schema_version": "knowledge-golden-manifest.v1",
            "annotation_set_version": "UNPOPULATED",
            "annotation_set_hash": canonical_hash([]),
            "partitions": {"train": [], "dev": [], "frozen_oos": []},
        },
        None,
    )
    assert empty["status"] == "BLOCKED"
    assert "MISSING_GOLDEN_OOS" in empty["blockers"]
    incomplete = evaluate(manifest, _output(manifest, missing_metadata=True))
    assert "PIPELINE_LINEAGE_MISSING" in incomplete["blockers"]


def test_leakage_tamper_secret_and_safe_annotation_file_loading(tmp_path):
    assert (
        "PARTITION_SOURCE_LEAKAGE"
        in evaluate(_manifest(duplicate=True), _output(_manifest(duplicate=True)))["blockers"]
    )
    manifest = _manifest()
    manifest["partitions"]["frozen_oos"][0]["url"] = "https://protected.example/token=secret"
    assert "SECRET_OR_MEDIA_VALUE_FORBIDDEN" in evaluate(manifest, _output(manifest))["blockers"]
    root = tmp_path / "knowledge_golden"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "one.json").write_text('{"claims": []}', encoding="utf-8")
    fixture = {
        "schema_version": "knowledge-golden-manifest.v1",
        "annotation_set_version": "v1",
        "annotation_set_hash": canonical_hash([]),
        "partitions": {"train": [], "dev": [], "frozen_oos": []},
    }
    (root / "manifest.json").write_text(json.dumps(fixture), encoding="utf-8")
    assert load_manifest(root / "manifest.json")["partitions"]
    fixture["partitions"]["frozen_oos"] = [{"annotation_file": "../outside.json"}]
    (root / "manifest.json").write_text(json.dumps(fixture), encoding="utf-8")
    try:
        load_manifest(root / "manifest.json")
    except ValueError as error:
        assert "PATH_FORBIDDEN" in str(error)
    else:  # pragma: no cover
        raise AssertionError("path traversal must fail")


def test_default_cli_is_blocked_and_nonzero(tmp_path, capsys):
    manifest = Path(__file__).parents[1] / "benchmarks" / "knowledge_golden" / "manifest.json"
    assert main(["--manifest", str(manifest)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_report_contract_is_valid_json_and_manifest_checksum_is_pinned():
    root = Path(__file__).parents[1]
    schema = root / "contracts" / "knowledge-quality-report.v1.json"
    assert json.loads(schema.read_text(encoding="utf-8"))["required"] == [
        "contract",
        "schema_version",
        "annotation_set_version",
        "annotation_set_hash",
        "partition",
        "status",
        "release_ready",
        "metrics",
        "strata",
        "blockers",
    ]
    manifest = yaml.safe_load((root / "contracts" / "platform-manifest.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["contracts"] if item["id"] == "knowledge-quality-report.v1")
    assert entry["checksum"] == "sha256:" + sha256(schema.read_bytes()).hexdigest().upper()

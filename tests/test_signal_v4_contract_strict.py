from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from stock_content.domain.signal_contract import upgrade_signal_v4, validate_signal_v4


def _valid_signal() -> dict:
    return {
        "signal_id": "signal-1",
        "decision_id": "decision-1",
        "signal_schema_version": "content-factor-signal.v4",
        "signal_policy_version": "signal_policy.v1",
        "content_snapshot_id": "snapshot-1",
        "claim_id": "claim-1",
        "verification_artifact_id": "verification-1",
        "signal_status": "NORMAL",
        "event_type": None,
        "truth_scope": "FACT",
        "symbol": "600000.SH",
        "signal_type": "FACT",
        "fact_category": "FINANCIAL_METRIC",
        "direction": "NEUTRAL",
        "magnitude": 0.5,
        "confidence": 1,
        "event_time": "2026-01-01T00:00:00Z",
        "available_from": "2026-01-01T00:00:00Z",
        "published_at": "2026-01-01T00:00:00Z",
        "snapshot_id": "snapshot-1",
        "source": {
            "source_artifact_id": "source-1",
            "source_type": "fixture",
            "source_ref": "fixture-1",
        },
        "support": {
            "status": "SUPPORTED",
            "score": 0,
            "evidence_refs": ["evidence-1"],
        },
        "verification": {
            "status": "VERIFIED",
            "provider": "quant",
            "market_snapshot_id": None,
            "market_data_version": None,
            "verification_rule_version": "verification_rule.v1",
        },
        "evidence_refs": ["evidence-1"],
        "producer": {
            "service": "stock_content",
            "service_version": "1.0.0",
            "code_sha": "unknown",
            "container_digest": None,
            "dependency_lock_hash": None,
            "python_lock_hash": None,
            "pipeline_version": "pipeline.v3",
            "model_id": "model-1",
            "prompt_version": "prompt-1",
            "trace_id": "trace-1",
            "decision_id": "decision-1",
            "decision": "VERIFIED_FACT",
        },
        "policy": {
            "signal_policy_version": "signal_policy.v1",
            "forecast_confidence_threshold": 1,
        },
    }


def test_valid_fixture_covers_every_yaml_property_and_required_field():
    contract_path = Path(__file__).parents[1] / "contracts" / "content-factor-signal.v4.yaml"
    schema = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    payload_fields = set(_valid_signal())
    assert payload_fields == set(schema["properties"])
    assert set(schema["required"]) <= payload_fields


def test_v4_valid_payload_and_upgrade_output_match_contract():
    signal = _valid_signal()
    assert validate_signal_v4(signal) is signal

    upgraded = upgrade_signal_v4(
        {
            "signal_type": "FORECAST",
            "fact_category": "FORECAST",
            "confidence": 0.8,
            "published_at": "2026-01-01T00:00:00Z",
        },
        content_snapshot_id="snapshot-1",
        claim_id="claim-1",
        verification_artifact_id="verification-1",
    )
    assert upgraded["truth_scope"] == "AUTHOR_FORECAST"
    assert upgraded["producer"]["decision_id"] == upgraded["decision_id"]
    validate_signal_v4(upgraded)


def test_v4_producer_decision_is_optional_per_yaml_contract():
    signal = _valid_signal()
    del signal["producer"]["decision"]

    contract_path = Path(__file__).parents[1] / "contracts" / "content-factor-signal.v4.yaml"
    schema = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    assert "decision" not in schema["properties"]["producer"]["required"]
    validate_signal_v4(signal)


@pytest.mark.parametrize("decision", [None, ""])
def test_v4_rejects_present_null_or_empty_producer_decision(decision: object):
    signal = _valid_signal()
    signal["producer"]["decision"] = decision
    with pytest.raises(ValueError):
        validate_signal_v4(signal)


def test_upgrade_v4_producer_nulls_do_not_override_defaults_or_decision_id():
    upgraded = upgrade_signal_v4(
        {
            "signal_type": "FACT",
            "producer": {
                "decision": None,
                "model_id": None,
                "decision_id": None,
            },
        },
        content_snapshot_id="snapshot-1",
        claim_id="claim-1",
        verification_artifact_id="verification-1",
    )

    assert upgraded["producer"]["decision"] == "unknown"
    assert upgraded["producer"]["model_id"] == "unknown"
    assert upgraded["decision_id"] == upgraded["producer"]["decision_id"]
    validate_signal_v4(upgraded)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("truth_scope", "NOT_A_SCOPE"),
        ("signal_type", "NOT_A_TYPE"),
        ("direction", "UP"),
        ("signal_status", "PENDING"),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("confidence", True),
        ("confidence", "0.5"),
        ("support", {"status": "SUPPORTED", "score": False, "evidence_refs": []}),
        ("support", {"status": "SUPPORTED", "score": 0.5, "evidence_refs": [1]}),
        (
            "verification",
            {
                "status": "VERIFIED",
                "provider": "",
                "market_snapshot_id": None,
                "market_data_version": None,
                "verification_rule_version": "verification_rule.v1",
            },
        ),
        (
            "verification",
            {
                "status": "VERIFIED",
                "provider": "quant",
                "market_snapshot_id": 1,
                "market_data_version": None,
                "verification_rule_version": "verification_rule.v1",
            },
        ),
        ("policy", {"signal_policy_version": "signal_policy.v1", "forecast_confidence_threshold": 2}),
    ],
)
def test_v4_rejects_yaml_type_enum_and_range_violations(field: str, value: object):
    signal = _valid_signal()
    signal[field] = value
    with pytest.raises(ValueError):
        validate_signal_v4(signal)


@pytest.mark.parametrize(
    ("path",),
    [
        (("truth_scope",),),
        (("verification", "provider"),),
        (("verification", "verification_rule_version"),),
    ],
)
def test_v4_enforces_required_fields(path: tuple[str, ...]):
    signal = _valid_signal()
    target = signal
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(ValueError):
        validate_signal_v4(signal)


def test_v4_rejects_nested_additional_properties_and_wrong_array_types():
    signal = _valid_signal()
    signal["source"]["extra"] = "nope"
    with pytest.raises(ValueError):
        validate_signal_v4(signal)

    signal = _valid_signal()
    signal["evidence_refs"] = "evidence-1"
    with pytest.raises(ValueError):
        validate_signal_v4(signal)

    signal = _valid_signal()
    signal["support"]["evidence_refs"] = [None]
    with pytest.raises(ValueError):
        validate_signal_v4(signal)


def test_v4_allows_only_yaml_nullable_fields_to_be_null():
    signal = _valid_signal()
    signal["event_type"] = None
    signal["verification"]["market_snapshot_id"] = None
    signal["verification"]["market_data_version"] = None
    signal["producer"]["container_digest"] = None
    signal["producer"]["dependency_lock_hash"] = None
    signal["producer"]["python_lock_hash"] = None
    validate_signal_v4(signal)

    for path in (("signal_id",), ("truth_scope",), ("support", "score"), ("policy", "signal_policy_version")):
        invalid = deepcopy(signal)
        target = invalid
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = None
        with pytest.raises(ValueError):
            validate_signal_v4(invalid)

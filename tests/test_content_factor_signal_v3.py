"""content-factor-signal.v3 契约测试（详细修改方案 §7）。"""
from __future__ import annotations

from stock_content.domain.signal_contract import (
    SIGNAL_SCHEMA_VERSION,
    accepts_schema_version,
    signal_major_version,
    upgrade_signal_v3,
)


def _v2_item() -> dict:
    return {
        "signal_id": "sig-1",
        "knowledge_uid": "k-1",
        "symbol": "600519.SH",
        "subject_key": "600519.SH",
        "kind": "CLAIM",
        "knowledge_kind": "STATE",
        "sentiment": "BULLISH",
        "confidence": 0.75,
        "support_status": "SOURCE_SUPPORTED",
        "as_of_time": "2026-08-14T09:30:00+00:00",
        "available_from": "2026-08-14T10:00:00+00:00",
        "event_strength": 0.8,
        "evidence_ids": ["ev-1", "ev-2"],
        "market_snapshot_id": "market-snap-1",
        "market_data_version": "market-data.v1",
        "market_fact_date": "2026-08-14",
        "provenance": {"model": "content-llm.v1", "prompt_version": "prompt.v2"},
    }


def test_signal_v3_contains_required_version_fields():
    signal = upgrade_signal_v3(_v2_item(), code_sha="sha-abc")
    assert signal["signal_schema_version"] == SIGNAL_SCHEMA_VERSION == "content-factor-signal.v3"
    assert signal["producer_version"]
    assert signal["producer"] == {
        "service_version": "1.0.0",
        "code_sha": "sha-abc",
        "model_id": "content-llm.v1",
        "prompt_version": "prompt.v2",
    }


def test_signal_v3_keeps_v2_fields_backward_compatible():
    signal = upgrade_signal_v3(_v2_item(), code_sha="sha-abc")
    for key, value in _v2_item().items():
        if key != "provenance":
            assert signal[key] == value


def test_signal_v3_maps_semantic_fields():
    signal = upgrade_signal_v3(_v2_item(), code_sha="sha-abc")
    assert signal["direction"] == "LONG"
    assert signal["event_time"] == "2026-08-14"
    assert signal["published_at"] == "2026-08-14T10:00:00+00:00"
    assert signal["signal_type"] == "STATE"
    assert signal["magnitude"] == 0.8
    assert signal["evidence_refs"] == ["ev-1", "ev-2"]


def test_factor_can_reject_unsupported_major_version():
    assert signal_major_version("content-factor-signal.v3") == 3
    assert accepts_schema_version("content-factor-signal.v2")
    assert accepts_schema_version("content-factor-signal.v3")
    # 只支持到 v3 的消费方必须拒绝 v4
    assert not accepts_schema_version("content-factor-signal.v4", max_supported_major=3)
    assert not accepts_schema_version("garbage")

"""Quant Snapshot Lineage 测试（详细修改方案 §5 P1-3）。

价格类核验结果必须绑定 market_snapshot_id / market_data_version / fact_date /
adjustment / verification_timestamp / verification_rule_version，禁止只保存 verified=true。
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from stock_content.domain.claims import VerificationResult


def _binding(**overrides) -> dict:
    payload = {
        "market_snapshot_id": "market-snap-1",
        "market_data_version": "market-data.v1",
        "fact_date": date(2026, 8, 14),
        "adjustment": "FORWARD",
        "verification_timestamp": datetime.now(UTC),
        "verification_rule_version": "verification_rule.v1",
    }
    payload.update(overrides)
    return payload


def test_verified_result_requires_full_quant_lineage():
    result = VerificationResult(claim_id="claim-1", status="VERIFIED", **_binding())
    assert result.market_snapshot_id == "market-snap-1"
    assert result.market_data_version == "market-data.v1"
    assert result.verification_rule_version == "verification_rule.v1"


@pytest.mark.parametrize(
    "missing", ["market_snapshot_id", "market_data_version", "fact_date", "verification_timestamp"]
)
def test_bare_verified_true_is_rejected(missing: str):
    payload = _binding()
    payload[missing] = None
    with pytest.raises(ValidationError):
        VerificationResult(claim_id="claim-1", status="VERIFIED", **payload)


def test_contradicted_also_requires_binding():
    with pytest.raises(ValidationError):
        VerificationResult(claim_id="claim-1", status="CONTRADICTED")


def test_pending_and_not_verifiable_do_not_require_binding():
    pending = VerificationResult(claim_id="claim-1", status="VERIFICATION_PENDING")
    assert pending.market_snapshot_id is None
    not_verifiable = VerificationResult(claim_id="claim-1", status="NOT_VERIFIABLE", reason="OPINION_LAYER")
    assert not_verifiable.status == "NOT_VERIFIABLE"


def test_lineage_flows_into_factor_signal(tmp_path):
    """核验绑定的 market_snapshot_id 沿 signal 链路传递给 Factor。"""
    from fastapi.testclient import TestClient

    from stock_content.api.dependencies import build_application
    from stock_content.api.main import create_app

    application = build_application(f"sqlite:///{tmp_path / 'content.db'}", enable_qdrant=False)
    client = TestClient(create_app(application))
    client.post(
        "/api/v1/videos/bilibili/ingest",
        json={
            "bv_id": "BV1lineage",
            "options": {
                "metadata": {"title": "lineage fixture"},
                "transcript": "股票600000今日上涨。",
                "offline_fixture": True,
            },
        },
    )
    application.process_next("lineage-test")

    signals = client.post(
        "/internal/v1/factor-signals",
        json={"symbols": ["600000"], "start": "2026-01-01T00:00:00Z", "end": "2026-12-31T00:00:00Z"},
    ).json()
    item = signals["items"][0]
    # P1-3：signal 携带 quant lineage 字段（未核验时为 None，但字段必须存在）。
    for key in ("market_snapshot_id", "market_data_version", "evidence_refs", "signal_schema_version"):
        assert key in item

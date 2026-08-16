"""QuantFactClient 与外部事实验证测试（设计文档 §11 / §87）。"""
from __future__ import annotations

import httpx

from stock_content.adapters.http.quant_fact_client import QuantExternalFactProvider, QuantFactClient
from stock_content.domain.external_fact_verifier import ExternalFactVerifier


def _bars_response(dates, closes, snapshot_id="mds-abc"):
    return httpx.Response(
        200,
        json={
            "contract_version": "market-data.v1",
            "data": {
                "symbols": ["600519.SH"],
                "dates": dates,
                "bars": {
                    "open": [[close * 0.99 for close in closes]],
                    "high": [[close * 1.01 for close in closes]],
                    "low": [[close * 0.98 for close in closes]],
                    "close": [closes],
                    "volume": [[1000] * len(closes)],
                    "amount": [[None] * len(closes)],
                    "turnover": [[None] * len(closes)],
                },
                "data_version": "v-hash",
                "data_snapshot_id": snapshot_id,
                "source": "qmt",
            },
        },
        request=httpx.Request("POST", "http://quant:8011/api/v1/market/bars/batch"),
    )


def test_quant_fact_client_returns_pit_snapshot(monkeypatch):
    calls = []

    def fake_post(url, json, timeout, headers=None):
        calls.append(url)
        return _bars_response(["2026-08-13", "2026-08-14", "2026-08-15"], [1600.0, 1580.0, 1453.6])

    monkeypatch.setattr(httpx, "post", fake_post)
    client = QuantFactClient()
    snapshot = client.get_price_snapshot("600519.SH", "2026-08-15")
    assert snapshot["close"] == 1453.6
    assert snapshot["previous_close"] == 1580.0
    assert snapshot["data_snapshot_id"] == "mds-abc"
    assert calls[0].endswith("/api/v1/market/bars/batch")


def test_quant_fact_client_falls_back_to_legacy_path(monkeypatch):
    def fake_post(url, json, timeout, headers=None):
        if url.endswith("/api/v1/market/bars/batch"):
            return httpx.Response(404, request=httpx.Request("POST", url))
        return _bars_response(["2026-08-15"], [1453.6])

    monkeypatch.setattr(httpx, "post", fake_post)
    snapshot = QuantFactClient().get_price_snapshot("600519.SH", "2026-08-15")
    assert snapshot["close"] == 1453.6


class _FixedClient:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def get_price_snapshot(self, symbol, event_date):
        return self._snapshot


def _unit(claimed_close=None, claimed_change_pct=None):
    attributes = {"symbol": "600519.SH", "trading_date": "2026-08-15"}
    if claimed_close is not None:
        attributes["claimed_close"] = claimed_close
    if claimed_change_pct is not None:
        attributes["claimed_change_pct"] = claimed_change_pct
    return {"knowledge_uid": "k1", "knowledge_kind": "PRICE_LEVEL", "attributes": attributes}


def test_provider_matches_claim_against_market_fact():
    snapshot = {"close": 1453.6, "previous_close": 1580.0, "data_snapshot_id": "mds-abc"}
    provider = QuantExternalFactProvider(client=_FixedClient(snapshot))
    # 实际跌幅 -8%；claim -8% -> MATCH
    outcome = provider.verify(_unit(claimed_change_pct=-8.0))
    assert outcome["status"] == "MATCH"
    assert outcome["market_fact"]["data_snapshot_id"] == "mds-abc"


def test_provider_conflicts_wrong_claim():
    snapshot = {"close": 1453.6, "previous_close": 1580.0}
    provider = QuantExternalFactProvider(client=_FixedClient(snapshot))
    outcome = provider.verify(_unit(claimed_change_pct=+8.0))
    assert outcome["status"] == "CONFLICT"
    outcome = provider.verify(_unit(claimed_close=2000.0))
    assert outcome["status"] == "CONFLICT"


def test_provider_pending_when_quant_unavailable(monkeypatch):
    provider = QuantExternalFactProvider(client=_FixedClient(None))
    outcome = provider.verify(_unit(claimed_close=1453.6))
    assert outcome["status"] == "PENDING"
    assert outcome["reason"] == "QUANT_UNAVAILABLE_OR_NO_DATA"


def test_verifier_marks_pending_without_blocking_ingestion(monkeypatch):
    """§87：Quant 不可用 -> Verification Pending，不能阻塞内容入库。"""
    monkeypatch.setenv("CONTENT_EXTERNAL_FACT_VERIFICATION", "true")
    verifier = ExternalFactVerifier(provider=QuantExternalFactProvider(client=_FixedClient(None)))
    results = verifier.verify_many([_unit(claimed_close=1453.6)])
    assert results[0]["external_verification_status"] == "VERIFICATION_PENDING"
    # 事实判定权仍在 Content：truth_status 不被外部源改写
    assert results[0]["truth_status"] == "NOT_CHECKED"


def test_verifier_uses_quant_match(monkeypatch):
    monkeypatch.setenv("CONTENT_EXTERNAL_FACT_VERIFICATION", "true")
    snapshot = {"close": 1453.6, "previous_close": 1580.0}
    verifier = ExternalFactVerifier(provider=QuantExternalFactProvider(client=_FixedClient(snapshot)))
    results = verifier.verify_many([_unit(claimed_change_pct=-8.0)])
    assert results[0]["external_verification_status"] == "EXTERNAL_MATCH"
    assert results[0]["truth_status"] == "EXTERNALLY_VERIFIED"

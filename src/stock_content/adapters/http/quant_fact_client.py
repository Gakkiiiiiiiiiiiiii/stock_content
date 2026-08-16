"""QuantFactClient：Content → Quant 事实验证通道（设计文档 §11 / §87）。

- 仅用于 external fact verification（如"茅台今天跌了 8%"对照 PIT 行情）；
- 核心 ingestion 不强依赖 Quant：Quant 不可用时返回 PENDING，
  验证标记为 VERIFICATION_PENDING，绝不阻塞内容入库；
- Quant 只返回事实，claim 是否成立仍由 Content 判定（§11）。
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

import httpx

TOLERANCE_PCT = 1.0


def _trace_headers() -> dict[str, str]:
    # §32：统一 Trace Headers，content → quant 调用透传 trace 与调用方标识。
    return {"X-Trace-Id": uuid4().hex, "X-Caller-Service": "stock_content"}


class QuantFactClient:
    """查询 quant market-data.v1 的 PIT 价格快照。"""

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self._url = (base_url or os.getenv("QUANT_SERVICE_URL", "http://quant:8011")).rstrip("/")
        self._timeout = timeout

    def configured(self) -> bool:
        return bool(os.getenv("QUANT_SERVICE_URL") or os.getenv("CONTENT_EXTERNAL_FACT_PROVIDER") == "quant")

    def get_price_snapshot(self, symbol: str, event_date: str) -> dict[str, Any] | None:
        """返回 event_date 当天（或最近一个交易日）的 PIT 价格快照。"""
        day = date.fromisoformat(str(event_date)[:10])
        payload = {
            "symbols": [symbol],
            "start": (day - timedelta(days=15)).isoformat(),
            "end": day.isoformat(),
            "frequency": "1d",
            "adjust": "none",
        }
        response = self._post("/api/v1/market/bars/batch", payload)
        if response is None or response.status_code == 404:
            response = self._post("/v1/bars/batch", payload)
        if response is None:
            return None
        response.raise_for_status()
        body = response.json()
        data = body.get("data") or {}
        dates = data.get("dates") or []
        bars = data.get("bars") or {}
        if not dates or not bars.get("close") or not bars["close"][0]:
            return None
        # 取不超过 event_date 的最后一个交易日（PIT：不使用未来数据）
        valid_indexes = [index for index, item in enumerate(dates) if item <= day.isoformat()]
        if not valid_indexes:
            return None
        index = valid_indexes[-1]
        return {
            "symbol": symbol,
            "trading_date": dates[index],
            "open": bars["open"][0][index],
            "close": bars["close"][0][index],
            "high": bars["high"][0][index],
            "low": bars["low"][0][index],
            "previous_close": bars["close"][0][valid_indexes[-2]] if len(valid_indexes) >= 2 else None,
            "data_snapshot_id": data.get("data_snapshot_id"),
            "data_version": data.get("data_version"),
        }

    def _post(self, path: str, payload: dict) -> httpx.Response | None:
        try:
            return httpx.post(f"{self._url}{path}", json=payload, headers=_trace_headers(), timeout=self._timeout)
        except httpx.HTTPError:
            return None


class QuantExternalFactProvider:
    """ExternalFactProvider 实现：用 Quant PIT 行情核验价格类 claim。"""

    def __init__(self, client: QuantFactClient | None = None) -> None:
        self._client = client or QuantFactClient()

    def configured(self) -> bool:
        return os.getenv("CONTENT_EXTERNAL_FACT_PROVIDER", "").lower() == "quant" or bool(os.getenv("QUANT_SERVICE_URL"))

    def verify(self, unit: dict[str, Any]) -> dict[str, Any]:
        attributes = unit.get("attributes") or {}
        symbol = attributes.get("symbol") or unit.get("symbol")
        event_date = attributes.get("trading_date") or attributes.get("event_date") or str(unit.get("available_from") or "")[:10]
        claimed_close = attributes.get("claimed_close")
        claimed_change_pct = attributes.get("claimed_change_pct")
        if not symbol or not event_date:
            return {"status": "NOT_FOUND", "reason": "MISSING_CLAIM_FIELDS"}
        try:
            snapshot = self._client.get_price_snapshot(str(symbol), str(event_date))
        except Exception:  # noqa: BLE001
            snapshot = None
        if snapshot is None:
            # §87：Quant 不可用 → Verification Pending，不得阻塞入库
            return {"status": "PENDING", "reason": "QUANT_UNAVAILABLE_OR_NO_DATA"}
        if claimed_close is None and claimed_change_pct is None:
            return {"status": "NOT_FOUND", "reason": "NO_NUMERIC_CLAIM", "market_fact": snapshot}
        status = "MATCH"
        if claimed_close is not None and snapshot.get("close") is not None:
            if abs(float(claimed_close) - float(snapshot["close"])) / max(abs(float(snapshot["close"])), 1e-9) * 100 > TOLERANCE_PCT:
                status = "CONFLICT"
        if claimed_change_pct is not None and snapshot.get("previous_close") and snapshot.get("close"):
            actual_pct = (float(snapshot["close"]) / float(snapshot["previous_close"]) - 1) * 100
            if abs(actual_pct - float(claimed_change_pct)) > TOLERANCE_PCT:
                status = "CONFLICT"
        return {"status": status, "market_fact": snapshot, "source": "quant"}


__all__ = ["QuantExternalFactProvider", "QuantFactClient"]

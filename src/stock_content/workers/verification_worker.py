"""Verification Worker（详细修改方案 §5 P1-2）。

独立进程消费 VERIFICATION_PENDING claim：Quant 短暂不可用只影响核验
延迟，不影响 ingest 主链。达到重试阈值进入 DLQ / manual review。
"""
from __future__ import annotations

import logging
import os
import time

from stock_content.application.verification_service import VerificationService, run_verification_pass

LOGGER = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 30.0


def run_once(service: VerificationService) -> dict:
    """单轮处理：返回本轮统计（供测试与调度器复用）。"""
    processed = run_verification_pass(service)
    return {
        "processed": len(processed),
        "pending": service.pending_count(),
        "dlq": service.dlq(),
    }


def main() -> None:  # pragma: no cover - 进程入口，由部署编排驱动
    logging.basicConfig(level=logging.INFO)
    poll_interval = float(os.getenv("VERIFICATION_POLL_SECONDS", str(DEFAULT_POLL_INTERVAL_SECONDS)))
    service = VerificationService(provider=None)  # provider 由部署装配注入
    LOGGER.info("verification worker started (poll=%.1fs)", poll_interval)
    while True:
        try:
            stats = run_once(service)
            LOGGER.info("verification pass: %s", stats)
        except Exception:  # noqa: BLE001 - worker 永不退出
            LOGGER.exception("verification pass failed")
        time.sleep(poll_interval)


if __name__ == "__main__":  # pragma: no cover
    main()

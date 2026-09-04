"""Verification Worker（详细修改方案 §5 P1-2）。

独立进程消费 VERIFICATION_PENDING claim：Quant 短暂不可用只影响核验
延迟，不影响 ingest 主链。达到重试阈值进入 DLQ / manual review。
"""
from __future__ import annotations

import logging
import os
import time

from stock_content.adapters.http import QuantExternalFactProvider
from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories import (
    ClaimStateEventRepository,
    PostgresVerificationJobRepository,
    SqlClaimRepository,
)
from stock_content.application.verification_refresh import VerificationRefreshService
from stock_content.application.verification_service import VerificationService, run_verification_pass
from stock_content.application.verification_worker import VerificationWorkerApplication
from stock_content.domain.lineage import default_code_sha
from stock_content.domain.worker_capability import TaskKind, WorkerProfile, require_capability

LOGGER = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 30.0
WORKER_PROFILE = WorkerProfile.CORE
QUEUE = TaskKind.CORE


def run_once(service: VerificationService) -> dict:
    """单轮处理：返回本轮统计（供测试与调度器复用）。"""
    processed = run_verification_pass(service)
    return {
        "processed": len(processed),
        "pending": service.pending_count(),
        "dlq": service.dlq(),
    }


def run_db_once(worker_id: str = "verification-worker", limit: int = 10) -> dict[str, int]:
    """Run one durable pass; production authority is PostgreSQL, not memory."""
    default_code_sha()
    database = Database()
    database.create_schema()
    jobs = PostgresVerificationJobRepository(database.session_factory)
    claims = SqlClaimRepository(database.session_factory)
    claim_events = ClaimStateEventRepository(database.session_factory)
    configured = QuantExternalFactProvider()
    provider = configured if configured.configured() else None
    refresh = VerificationRefreshService(database.session_factory, jobs, claims, claim_event_repository=claim_events)
    from stock_content.adapters.postgres.repositories import PostgresTaskRunRepository
    from stock_content.application.task_lease_service import TaskLeaseService

    task_leases = TaskLeaseService(PostgresTaskRunRepository(database.session_factory))
    return VerificationWorkerApplication(jobs, claims, provider, refresh, task_leases).run_once(worker_id, limit)


def main() -> None:  # pragma: no cover - 进程入口，由部署编排驱动
    require_capability(WORKER_PROFILE, QUEUE)
    logging.basicConfig(level=logging.INFO)
    # Validate release identity before entering the retry loop.  Deployment
    # misconfiguration must terminate startup, not be logged forever as a
    # transient database/provider failure.
    default_code_sha()
    poll_interval = float(os.getenv("VERIFICATION_POLL_SECONDS", str(DEFAULT_POLL_INTERVAL_SECONDS)))
    LOGGER.info("verification worker started (poll=%.1fs)", poll_interval)
    while True:
        try:
            stats = run_db_once(os.getenv("VERIFICATION_WORKER_ID", "verification-worker"))
            LOGGER.info("verification pass: %s", stats)
        except Exception:  # noqa: BLE001 - worker 永不退出
            LOGGER.exception("verification pass failed")
        time.sleep(poll_interval)


if __name__ == "__main__":  # pragma: no cover
    main()

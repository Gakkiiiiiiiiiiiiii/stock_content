"""Atomic verification completion -> refresh snapshot -> signal outbox UoW.

The import path remains a stable façade. Transaction orchestration, durable
persistence, and historical closure validation live in dedicated application
modules to keep this use case composable and reviewable.
"""
from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.repositories.snapshot_repository import (
    _compare_candidate_to_existing_row,
    _compare_existing_identity,
    _validate_snapshot_artifacts,
    _validate_snapshot_candidate,
    _validate_snapshot_row,
)
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    PostgresVerificationJobRepository,
)
from stock_content.application.signal_service import SignalService
from stock_content.application.verification.closure import (
    _as_utc,
    _head_is_newer,
    _upsert_source_head,
    _validate_refresh_verification_closure,
)
from stock_content.application.verification.persistence import VerificationPersistenceMixin
from stock_content.application.verification.transaction import VerificationTransactionMixin


class VerificationRefreshService(VerificationTransactionMixin, VerificationPersistenceMixin):
    """Public compatibility façade for verification refresh completion."""

    def __init__(
        self,
        session_factory: sessionmaker,
        jobs: PostgresVerificationJobRepository,
        claims,
        signal_service: SignalService | None = None,
        claim_event_repository=None,
    ) -> None:
        self._sessions = session_factory
        self._jobs = jobs
        self._claims = claims
        self._signals = signal_service or SignalService()
        self._claim_events = claim_event_repository


__all__ = [
    "VerificationRefreshService",
    "_as_utc",
    "_compare_candidate_to_existing_row",
    "_compare_existing_identity",
    "_head_is_newer",
    "_upsert_source_head",
    "_validate_refresh_verification_closure",
    "_validate_snapshot_artifacts",
    "_validate_snapshot_candidate",
    "_validate_snapshot_row",
]

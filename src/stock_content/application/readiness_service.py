"""Read-only readiness projection for facts, formal signals, and search."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


class ReadinessSnapshotPort(Protocol):
    def latest_ready(self) -> "SnapshotReadiness": ...


@dataclass(frozen=True, slots=True)
class SnapshotReadiness:
    snapshot_id: str | None
    state: str = "UNKNOWN"
    published_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ReadinessDependencies:
    postgres_ok: bool = True
    qdrant_ok: bool = True
    outbox_lag_seconds: float = 0.0
    index_lag_events: int = 0
    latest_snapshot: SnapshotReadiness = field(default_factory=lambda: SnapshotReadiness(None))
    contract_inventory: tuple[str, ...] = ()
    required_contracts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentReadiness:
    name: str
    ready: bool
    degraded: bool
    blocking_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    fact: ComponentReadiness
    signal: ComponentReadiness
    search: ComponentReadiness
    latest_ready_snapshot: SnapshotReadiness
    outbox_lag_seconds: float
    index_lag_events: int
    contract_inventory: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.fact.ready and self.signal.ready

    def to_dict(self) -> dict[str, object]:
        def component(value: ComponentReadiness) -> dict[str, object]:
            return {"ready": value.ready, "degraded": value.degraded, "blocking_reasons": list(value.blocking_reasons)}
        return {
            "ready": self.ready,
            "fact": component(self.fact),
            "signal": component(self.signal),
            "search": component(self.search),
            "latest_ready_snapshot": {
                "snapshot_id": self.latest_ready_snapshot.snapshot_id,
                "state": self.latest_ready_snapshot.state,
                "published_at": self.latest_ready_snapshot.published_at.isoformat()
                if self.latest_ready_snapshot.published_at else None,
            },
            "outbox_lag_seconds": self.outbox_lag_seconds,
            "index_lag_events": self.index_lag_events,
            "contract_inventory": list(self.contract_inventory),
        }


class ReadinessService:
    """Compute independently degradable readiness components.

    Qdrant only affects search. It can never make authoritative fact or signal
    readiness false because those projections are SQL-backed.
    """

    def __init__(self, *, max_outbox_lag_seconds: float = 300, max_index_lag_events: int = 1000):
        self.max_outbox_lag_seconds = max_outbox_lag_seconds
        self.max_index_lag_events = max_index_lag_events

    def evaluate(self, dependencies: ReadinessDependencies | None = None) -> ReadinessReport:
        dep = dependencies or ReadinessDependencies()
        fact_reasons = [] if dep.postgres_ok else ["postgres_unavailable"]
        snapshot = dep.latest_snapshot
        signal_reasons = list(fact_reasons)
        if snapshot.state not in {"READY", "PUBLISHING", "PUBLISHED"} or not snapshot.snapshot_id:
            signal_reasons.append("no_ready_snapshot")
        missing_contracts = sorted(set(dep.required_contracts) - set(dep.contract_inventory))
        if missing_contracts:
            signal_reasons.append("contract_inventory_incomplete")
        if dep.outbox_lag_seconds > self.max_outbox_lag_seconds:
            signal_reasons.append("outbox_lag_exceeded")
        # A component that cannot serve its semantic contract is degraded as
        # well as not-ready; callers must be able to distinguish that state
        # from a healthy component whose dependency is merely optional.
        signal = ComponentReadiness("signal", not signal_reasons, bool(signal_reasons), tuple(signal_reasons))
        fact = ComponentReadiness("fact", not fact_reasons, bool(fact_reasons), tuple(fact_reasons))
        search_reasons: list[str] = []
        if not dep.postgres_ok:
            search_reasons.append("postgres_unavailable")
        if not dep.qdrant_ok:
            search_reasons.append("qdrant_unavailable")
        search_degraded = bool(search_reasons) or dep.index_lag_events > self.max_index_lag_events
        if dep.index_lag_events > self.max_index_lag_events:
            search_reasons.append("index_lag_exceeded")
        search = ComponentReadiness("search", not search_reasons, search_degraded, tuple(search_reasons))
        return ReadinessReport(
            fact=fact,
            signal=signal,
            search=search,
            latest_ready_snapshot=snapshot,
            outbox_lag_seconds=dep.outbox_lag_seconds,
            index_lag_events=dep.index_lag_events,
            contract_inventory=tuple(dep.contract_inventory),
        )


__all__ = ["ComponentReadiness", "ReadinessDependencies", "ReadinessReport", "ReadinessService", "SnapshotReadiness"]

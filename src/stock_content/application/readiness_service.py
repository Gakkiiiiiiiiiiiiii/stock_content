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
    # A derived-index watermark belongs to the index/rebuild control plane,
    # not to the formal-signal outbox.  ``None`` is deliberately observable:
    # it means the service cannot make a freshness claim for Qdrant.
    index_lag_events: int | None = None
    index_state: str = "UNKNOWN"
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
    index_lag_events: int | None
    index_state: str
    max_index_lag_events: int
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
            "index_state": self.index_state,
            "index_backlog_slo_events": self.max_index_lag_events,
            "contract_inventory": list(self.contract_inventory),
            # These names make the operational permission distinction explicit
            # without changing any product or wire contract.  Formal publish
            # is fail-closed on SQL authority; read-only facts are not made
            # unavailable by the derived vector index.
            "capabilities": {
                "read_only_facts": component(self.fact),
                "formal_publish": component(self.signal),
                "derived_search": component(self.search),
            },
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
        index_state = _index_state(dep.index_state, qdrant_ok=dep.qdrant_ok)
        if index_state == "UNKNOWN":
            search_reasons.append("index_status_unknown")
        elif index_state == "STALE":
            search_reasons.append("index_stale")
        elif index_state == "REBUILDING":
            search_reasons.append("index_rebuilding")
        elif index_state == "DOWN" and dep.qdrant_ok:
            # Preserve a distinct audit reason for an index control-plane
            # failure even when the Qdrant transport still answers health.
            search_reasons.append("index_unavailable")
        if dep.index_lag_events is None:
            search_reasons.append("index_backlog_unknown")
        elif dep.index_lag_events > self.max_index_lag_events:
            search_reasons.append("index_backlog_exceeded")
        search = ComponentReadiness("search", not search_reasons, bool(search_reasons), tuple(search_reasons))
        return ReadinessReport(
            fact=fact,
            signal=signal,
            search=search,
            latest_ready_snapshot=snapshot,
            outbox_lag_seconds=dep.outbox_lag_seconds,
            index_lag_events=dep.index_lag_events,
            index_state=index_state,
            max_index_lag_events=self.max_index_lag_events,
            contract_inventory=tuple(dep.contract_inventory),
        )


def _index_state(value: str, *, qdrant_ok: bool) -> str:
    """Normalize external rebuild status and fail closed on unknown values."""
    if not qdrant_ok:
        return "DOWN"
    candidate = str(value).upper()
    return candidate if candidate in {"HEALTHY", "STALE", "REBUILDING", "DOWN"} else "UNKNOWN"


__all__ = ["ComponentReadiness", "ReadinessDependencies", "ReadinessReport", "ReadinessService", "SnapshotReadiness"]

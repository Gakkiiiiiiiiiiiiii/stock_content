# Fact authority and readiness SLO

PostgreSQL is authoritative for facts and formal signals. Qdrant is a
rebuildable search derivative. A Qdrant outage therefore degrades `search`
only; it must not report `fact` or `signal` as unavailable.

Readiness exposes the latest READY immutable snapshot, transactional outbox
lag, derived-index backlog, blocking reasons, and the loaded contract
inventory. `read_only_facts` remains available when SQL is healthy; the
`formal_publish` capability fails closed if SQL authority, the formal snapshot,
the required contract inventory (including `content-factor-signal.v5.1`), or
the formal-signal outbox SLO is not ready.

`derived_search` is ready only when Qdrant is reachable, its rebuild state is
`HEALTHY`, and its known backlog is at or below the reported SLO threshold.
`STALE`, `REBUILDING`, `DOWN`, and an unknown rebuild watermark are all
explicitly degraded search states. An unknown watermark is never presented as
a zero backlog; it must be supplied by a durable index/rebuild control plane
before search freshness can be claimed. A durable watermark control plane and
its PostgreSQL-to-Qdrant operational drill remain **BLOCKED**; the readiness
API therefore reports that missing evidence as degraded rather than implying
search freshness.

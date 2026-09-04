# Fact authority and readiness SLO

PostgreSQL is authoritative for facts and formal signals. Qdrant is a
rebuildable search derivative. A Qdrant outage therefore degrades `search`
only; it must not report `fact` or `signal` as unavailable.

Readiness exposes the latest READY immutable snapshot, transactional outbox
lag, index lag, blocking reasons, and the loaded contract inventory.

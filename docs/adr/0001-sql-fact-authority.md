# ADR 0001: SQL is fact authority

Status: accepted.

PostgreSQL owns immutable facts, event lineage, snapshot publication, and
formal-signal outbox state. Search stores are rebuildable derivatives and
cannot decide formal truth. Accordingly, readiness evaluates SQL fact/signal
capabilities separately from derived search capability; a Qdrant failure must
not make SQL facts unavailable.

This decision records repository implementation and deterministic regression
evidence only. A live PostgreSQL-to-Qdrant rebuild drill and durable index
watermark remain **BLOCKED**, so derived-search freshness is fail-closed when
that watermark is unknown.

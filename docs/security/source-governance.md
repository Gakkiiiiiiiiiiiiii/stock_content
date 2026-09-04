# Source governance

Ingestion is permitted only when a versioned `SourcePolicy` allows the
intended use and rate limit. Policy decisions include license/terms metadata,
access classification and retention class. Artifact removal is represented by
an append-only tombstone carrying actor, reason and policy version; derived
records follow the explicit derived retention period.

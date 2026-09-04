-- Additive P0 publication projection.  The delivery outbox remains mutable;
-- these rows are the immutable SQL source for publication retries.
CREATE TABLE IF NOT EXISTS content_publication_manifests (
  publication_run_id VARCHAR(128) PRIMARY KEY
    REFERENCES content_publication_runs(publication_run_id) ON DELETE CASCADE,
  content_snapshot_id VARCHAR(80) NOT NULL,
  query_hash VARCHAR(128) NOT NULL,
  signal_policy_version VARCHAR(80) NOT NULL,
  manifest_hash VARCHAR(128) NOT NULL,
  manifest JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS content_sealed_signals (
  sealed_signal_id VARCHAR(128) PRIMARY KEY,
  publication_run_id VARCHAR(128) NOT NULL
    REFERENCES content_publication_runs(publication_run_id) ON DELETE CASCADE,
  signal_id VARCHAR(128) NOT NULL,
  content_snapshot_id VARCHAR(80) NOT NULL,
  claim_id VARCHAR(96),
  schema_version VARCHAR(48) NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  UNIQUE (publication_run_id, signal_id)
);
CREATE INDEX IF NOT EXISTS ix_content_sealed_signals_snapshot
  ON content_sealed_signals(content_snapshot_id, signal_id);

-- Expand-only P0 historical authority and publication state.
-- Existing claims predate the append-only state stream.  Mark them explicitly
-- so a formal query cannot mistake the absence of history for a valid empty
-- projection.  New claims default to complete and receive events in the
-- snapshot transaction.
ALTER TABLE financial_claim
  ADD COLUMN IF NOT EXISTS legacy_history_incomplete BOOLEAN NOT NULL DEFAULT FALSE;
UPDATE financial_claim
SET legacy_history_incomplete = TRUE
WHERE legacy_history_incomplete IS FALSE
   OR legacy_history_incomplete IS NULL;

CREATE TABLE IF NOT EXISTS claim_state_events (
  claim_state_event_id VARCHAR(128) PRIMARY KEY,
  claim_id VARCHAR(96) NOT NULL,
  event_type VARCHAR(48) NOT NULL,
  business_valid_from TIMESTAMPTZ NULL,
  business_valid_to TIMESTAMPTZ NULL,
  known_from TIMESTAMPTZ NULL,
  known_to TIMESTAMPTZ NULL,
  source_available_from TIMESTAMPTZ NULL,
  payload JSONB NOT NULL,
  previous_event_hash VARCHAR(64) NULL,
  event_hash VARCHAR(64) NOT NULL UNIQUE,
  legacy_history_incomplete BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL,
  CONSTRAINT claim_state_events_known_interval CHECK (known_to IS NULL OR known_from IS NULL OR known_from < known_to),
  CONSTRAINT claim_state_events_business_interval CHECK (business_valid_to IS NULL OR business_valid_from IS NULL OR business_valid_from < business_valid_to),
  CONSTRAINT claim_state_events_claim_fk FOREIGN KEY (claim_id) REFERENCES financial_claim(claim_id)
);
CREATE INDEX IF NOT EXISTS ix_claim_state_event_claim_known ON claim_state_events (claim_id, known_from);
-- A chain tail may have at most one successor. COALESCE closes the NULL root
-- case on both PostgreSQL and SQLite upgrades.
CREATE UNIQUE INDEX IF NOT EXISTS uq_claim_state_event_predecessor
  ON claim_state_events (claim_id, COALESCE(previous_event_hash, '__ROOT__'));

CREATE TABLE IF NOT EXISTS content_publication_runs (
  publication_run_id VARCHAR(128) PRIMARY KEY,
  content_snapshot_id VARCHAR(80) NOT NULL,
  query_hash VARCHAR(128) NOT NULL,
  signal_policy_version VARCHAR(80) NOT NULL,
  state VARCHAR(32) NOT NULL,
  manifest_hash VARCHAR(128) NULL,
  version INTEGER NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  CONSTRAINT content_publication_runs_state_check CHECK (state IN ('ASSEMBLING','PROJECTING','SEALING','READY','PUBLISHING','PUBLISHED','FAILED_RETRYABLE','FAILED_TERMINAL')),
  UNIQUE (content_snapshot_id, query_hash, signal_policy_version)
);

-- Fence every replayable effect through a durable, idempotent intent.
CREATE TABLE IF NOT EXISTS content_task_effect (
  effect_id VARCHAR(128) PRIMARY KEY,
  task_id VARCHAR(64) NOT NULL REFERENCES content_ingest_task(task_id) ON DELETE CASCADE,
  effect_key VARCHAR(160) NOT NULL,
  effect_kind VARCHAR(64) NOT NULL,
  payload_hash VARCHAR(64) NOT NULL,
  fencing_token INTEGER NOT NULL,
  state VARCHAR(24) NOT NULL DEFAULT 'PENDING',
  attempt_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  CONSTRAINT uq_content_task_effect_key UNIQUE (task_id, effect_key)
);

-- No locator, credential, or object key is retained here. The storage adapter
-- maps artifact ids to private storage internally.
CREATE TABLE IF NOT EXISTS retention_execution (
  tombstone_id VARCHAR(80) PRIMARY KEY,
  artifact_id VARCHAR(96) NOT NULL,
  artifact_class VARCHAR(64) NOT NULL,
  content_hash VARCHAR(64) NOT NULL,
  source_identity_hash VARCHAR(64) NOT NULL,
  audit_lineage_id VARCHAR(128) NOT NULL,
  reason VARCHAR(64) NOT NULL,
  expired_at TIMESTAMPTZ NOT NULL,
  state VARCHAR(24) NOT NULL DEFAULT 'DELETE_PENDING',
  attempt_count INTEGER NOT NULL DEFAULT 0,
  last_error_code VARCHAR(64),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finalized_at TIMESTAMPTZ,
  CONSTRAINT uq_retention_execution_artifact UNIQUE (artifact_id)
);

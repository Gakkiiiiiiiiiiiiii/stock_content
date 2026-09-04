-- Durable worker leases and append-only operator audit.
CREATE TABLE IF NOT EXISTS background_task_run (
  task_run_id VARCHAR(128) PRIMARY KEY,
  task_type VARCHAR(64) NOT NULL,
  state VARCHAR(24) NOT NULL,
  owner VARCHAR(128),
  lease_expires_at TIMESTAMPTZ,
  fencing_token INTEGER NOT NULL DEFAULT 0,
  attempt INTEGER NOT NULL DEFAULT 0,
  checkpoints JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_background_task_run_due
  ON background_task_run (state, lease_expires_at);
CREATE TABLE IF NOT EXISTS operator_action_audit (
  audit_id VARCHAR(128) PRIMARY KEY,
  task_run_id VARCHAR(128) NOT NULL,
  action VARCHAR(32) NOT NULL,
  actor VARCHAR(128) NOT NULL,
  reason TEXT NOT NULL,
  request_id VARCHAR(128) NOT NULL,
  fencing_token INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_operator_action_audit_task
  ON operator_action_audit (task_run_id, created_at);

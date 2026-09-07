-- Qdrant is a derived projection.  These fields let an existing durable
-- content_task_effect intent be dispatched after its ingestion task has
-- reached SUCCEEDED, without borrowing or extending that task's lease.
ALTER TABLE content_task_effect
  ADD COLUMN IF NOT EXISTS projection_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS dispatch_owner VARCHAR(128),
  ADD COLUMN IF NOT EXISTS dispatch_expires_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_error_code VARCHAR(64);

CREATE INDEX IF NOT EXISTS ix_content_task_effect_projection_due
  ON content_task_effect (effect_kind, state, next_attempt_at, created_at);

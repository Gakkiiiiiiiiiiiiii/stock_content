-- A content-task claim carries a monotonically increasing fencing token.
-- All worker progress, checkpoint, failure, and terminal effect writes must
-- compare this token with the active, unexpired lease owner.
ALTER TABLE content_ingest_task
  ADD COLUMN IF NOT EXISTS fencing_token INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS ix_content_ingest_task_fenced_claim
  ON content_ingest_task(task_kind, status, lease_expires_at, created_at, fencing_token);

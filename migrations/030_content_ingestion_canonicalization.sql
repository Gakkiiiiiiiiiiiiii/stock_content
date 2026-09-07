-- Canonical source-ingestion command storage.  This is schema-only: legacy
-- row handling is an explicitly invoked operational procedure.
ALTER TABLE content_ingest_task
  ADD COLUMN IF NOT EXISTS request_hash VARCHAR(64),
  ADD COLUMN IF NOT EXISTS task_kind VARCHAR(32) NOT NULL DEFAULT 'legacy_unresolved',
  ADD COLUMN IF NOT EXISTS source_platform VARCHAR(32) NOT NULL DEFAULT 'legacy',
  ADD COLUMN IF NOT EXISTS canonical_source_ref TEXT,
  ADD COLUMN IF NOT EXISTS credential_ref_hash VARCHAR(64),
  ADD COLUMN IF NOT EXISTS locator_secret_hash VARCHAR(64),
  ADD COLUMN IF NOT EXISTS source_identity_hash VARCHAR(64) NOT NULL DEFAULT '0000000000000000000000000000000000000000000000000000000000000000';

CREATE INDEX IF NOT EXISTS ix_content_ingest_task_source_identity
  ON content_ingest_task(source_identity_hash);
CREATE INDEX IF NOT EXISTS ix_content_ingest_task_kind_pending
  ON content_ingest_task(task_kind, status, lease_expires_at, created_at);

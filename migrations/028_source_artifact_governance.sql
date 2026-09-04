-- Source policy metadata and recoverable tombstones. No destructive deletes.
CREATE TABLE IF NOT EXISTS source_artifact_metadata (
  artifact_id VARCHAR(96) PRIMARY KEY,
  source_policy_version VARCHAR(80) NOT NULL,
  retention_class VARCHAR(64) NOT NULL,
  access_classification VARCHAR(32) NOT NULL,
  source_content_hash VARCHAR(64) NOT NULL,
  content_size INTEGER NOT NULL,
  mime_type VARCHAR(128) NOT NULL,
  encryption_key_id VARCHAR(128),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS artifact_tombstone (
  artifact_id VARCHAR(96) PRIMARY KEY,
  reason TEXT NOT NULL,
  actor VARCHAR(128) NOT NULL,
  policy_version VARCHAR(80) NOT NULL,
  request_id VARCHAR(128) NOT NULL,
  deleted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

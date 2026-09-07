-- Private-only locator mapping used by the retention worker.  Public artifact
-- payloads and durable tombstones intentionally contain no storage locator.
CREATE TABLE IF NOT EXISTS retention_artifact_locator (
  artifact_id VARCHAR(96) PRIMARY KEY REFERENCES content_artifact(artifact_id) ON DELETE CASCADE,
  private_root_id VARCHAR(64) NOT NULL,
  relative_locator VARCHAR(512) NOT NULL,
  legal_hold BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

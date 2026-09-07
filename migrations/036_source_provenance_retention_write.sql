-- Additive, normalized public provenance for SQL Bundle reads. Runtime
-- locators and credentials intentionally remain outside this table.
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS canonical_url TEXT;
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS source_type VARCHAR(32);
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS source_id VARCHAR(255);
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS source_part VARCHAR(128);
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS source_identity_hash VARCHAR(64);
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS source_version_id VARCHAR(128);
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS author VARCHAR(255);
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS business_as_of TIMESTAMPTZ;
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS source_available_from TIMESTAMPTZ;
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS pipeline_version VARCHAR(128);
ALTER TABLE source_artifact_metadata ADD COLUMN IF NOT EXISTS service_version VARCHAR(128);
CREATE INDEX IF NOT EXISTS ix_source_artifact_metadata_source_identity_hash
  ON source_artifact_metadata (source_identity_hash);

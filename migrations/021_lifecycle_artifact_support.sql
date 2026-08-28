-- Close snapshot lineage over the append-only lifecycle artifact.
ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS lifecycle_artifact_id varchar(128);
CREATE INDEX IF NOT EXISTS ix_content_snapshot_lifecycle_artifact_id
    ON content_snapshot(lifecycle_artifact_id);
ALTER TABLE content_snapshot_artifact ADD COLUMN IF NOT EXISTS artifact_role varchar(40);
UPDATE content_snapshot_artifact
SET artifact_role = 'LEGACY_MEMBER'
WHERE artifact_role IS NULL;

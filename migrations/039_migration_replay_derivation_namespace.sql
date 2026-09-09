-- Preserve immutable semantic rows when a migration reprocesses one sealed
-- transcript with a different target pipeline. Empty namespace is the
-- legacy/normal-ingestion domain, so existing identities remain unchanged.
ALTER TABLE semantic_segment
    ADD COLUMN IF NOT EXISTS derivation_namespace varchar(48) NOT NULL DEFAULT '';
ALTER TABLE semantic_segment
    DROP CONSTRAINT IF EXISTS semantic_segment_transcript_artifact_id_segment_index_key;
CREATE UNIQUE INDEX IF NOT EXISTS uq_semantic_segment_transcript_namespace_index
    ON semantic_segment(transcript_artifact_id, derivation_namespace, segment_index);

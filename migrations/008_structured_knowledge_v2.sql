-- Upgrade (do not edit 005): preserve verification history while aligning it
-- with the structured verifier contract used by the ORM.
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS verification_id VARCHAR(64);
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS verifier_type VARCHAR(40);
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS decision VARCHAR(40);
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS reason_code VARCHAR(80);
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS model_name VARCHAR(120);
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS model_version VARCHAR(80);
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS prompt_version VARCHAR(80);
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS raw_output JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE knowledge_verification ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
UPDATE knowledge_verification
SET verification_id = COALESCE(verification_id, 'legacy-' || id::text),
    verifier_type = COALESCE(verifier_type, verifier),
    decision = COALESCE(decision, verdict),
    raw_output = CASE WHEN raw_output = '{}'::jsonb THEN COALESCE(detail, '{}'::jsonb) ELSE raw_output END,
    created_at = COALESCE(created_at, verified_at, NOW())
WHERE verification_id IS NULL OR verifier_type IS NULL OR decision IS NULL OR created_at IS NULL;
ALTER TABLE knowledge_verification ALTER COLUMN verification_id SET NOT NULL;
ALTER TABLE knowledge_verification ALTER COLUMN verifier_type SET NOT NULL;
ALTER TABLE knowledge_verification ALTER COLUMN decision SET NOT NULL;
ALTER TABLE knowledge_verification ALTER COLUMN created_at SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_knowledge_verification_id ON knowledge_verification(verification_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_verification_type ON knowledge_verification(verifier_type);
CREATE INDEX IF NOT EXISTS ix_knowledge_verification_decision ON knowledge_verification(decision);

CREATE TABLE IF NOT EXISTS knowledge_cross_video (
    knowledge_uid VARCHAR(64) PRIMARY KEY REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    corroborating_video_count INTEGER NOT NULL DEFAULT 0,
    contradicting_video_count INTEGER NOT NULL DEFAULT 0,
    independent_source_count INTEGER NOT NULL DEFAULT 0,
    author_attention_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    consensus_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    disagreement_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS knowledge_conflict (
    conflict_id VARCHAR(64) PRIMARY KEY,
    knowledge_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    related_knowledge_uid VARCHAR(64) NOT NULL,
    conflict_type VARCHAR(40) NOT NULL,
    resolution VARCHAR(40) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_knowledge_conflict_uid ON knowledge_conflict(knowledge_uid);
CREATE TABLE IF NOT EXISTS knowledge_lifecycle_event (
    event_id VARCHAR(64) PRIMARY KEY,
    knowledge_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    knowledge_version INTEGER NOT NULL,
    from_status VARCHAR(40), to_status VARCHAR(40) NOT NULL,
    reason TEXT NOT NULL, trigger VARCHAR(80) NOT NULL, actor VARCHAR(80) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_knowledge_lifecycle_event_uid ON knowledge_lifecycle_event(knowledge_uid);
CREATE TABLE IF NOT EXISTS analysis_document (
    document_id VARCHAR(64) PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    markdown TEXT NOT NULL DEFAULT '', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_analysis_document_video ON analysis_document(video_id);

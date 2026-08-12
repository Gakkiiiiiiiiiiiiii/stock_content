CREATE TABLE IF NOT EXISTS knowledge_verification (
    id BIGSERIAL PRIMARY KEY,
    knowledge_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    verifier VARCHAR(80) NOT NULL,
    verdict VARCHAR(40) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_verification_uid ON knowledge_verification(knowledge_uid);

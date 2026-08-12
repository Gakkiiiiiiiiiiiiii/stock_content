CREATE TABLE IF NOT EXISTS knowledge_lifecycle_audit (
    id BIGSERIAL PRIMARY KEY,
    knowledge_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    previous_status VARCHAR(40),
    new_status VARCHAR(40) NOT NULL,
    reason TEXT NOT NULL,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_unit (
    knowledge_uid VARCHAR(64) PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    chapter_id VARCHAR(64),
    statement TEXT NOT NULL,
    kind VARCHAR(40) NOT NULL DEFAULT 'CLAIM',
    subject VARCHAR(255),
    ticker VARCHAR(32),
    sentiment VARCHAR(20) NOT NULL DEFAULT 'NEUTRAL',
    support_status VARCHAR(40) NOT NULL DEFAULT 'SOURCE_SUPPORTED',
    review_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    confidence DOUBLE PRECISION NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    available_from TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_evidence (
    id BIGSERIAL PRIMARY KEY,
    knowledge_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    evidence_type VARCHAR(40) NOT NULL,
    evidence_text TEXT NOT NULL,
    start_seconds DOUBLE PRECISION,
    end_seconds DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS knowledge_entity_relation (
    id BIGSERIAL PRIMARY KEY,
    knowledge_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    entity_type VARCHAR(40) NOT NULL,
    entity_value VARCHAR(255) NOT NULL,
    relation_type VARCHAR(40) NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_unit_relation (
    id BIGSERIAL PRIMARY KEY,
    source_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    target_uid VARCHAR(64) NOT NULL REFERENCES knowledge_unit(knowledge_uid) ON DELETE CASCADE,
    relation_type VARCHAR(40) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    UNIQUE(source_uid, target_uid, relation_type)
);

CREATE INDEX IF NOT EXISTS ix_knowledge_factor_signal
    ON knowledge_unit(ticker, available_from, support_status);

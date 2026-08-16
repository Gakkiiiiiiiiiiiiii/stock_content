-- 详细修改方案 §5 P1-1/P1-2/P1-3：FinancialClaim 与验证生命周期
CREATE TABLE IF NOT EXISTS financial_claim (
    claim_id VARCHAR(80) PRIMARY KEY,
    claim_type VARCHAR(40) NOT NULL,
    fact_category VARCHAR(20) NOT NULL,
    subject_type VARCHAR(40) NOT NULL,
    subject_id VARCHAR(128) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    value JSONB,
    unit VARCHAR(40),
    fact_time TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    evidence_refs JSONB NOT NULL DEFAULT '[]',
    source_confidence DOUBLE PRECISION NOT NULL,
    extractor_confidence DOUBLE PRECISION NOT NULL,
    video_id VARCHAR(64),
    content_snapshot_id VARCHAR(80),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_claim_subject ON financial_claim(subject_type, subject_id, predicate);
CREATE INDEX IF NOT EXISTS idx_claim_snapshot ON financial_claim(content_snapshot_id);

CREATE TABLE IF NOT EXISTS claim_verification_lifecycle (
    claim_id VARCHAR(80) PRIMARY KEY REFERENCES financial_claim(claim_id) ON DELETE CASCADE,
    status VARCHAR(40) NOT NULL DEFAULT 'EXTRACTED',
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TIMESTAMPTZ,
    market_snapshot_id VARCHAR(128),
    market_data_version VARCHAR(80),
    fact_date DATE,
    adjustment VARCHAR(20),
    verification_timestamp TIMESTAMPTZ,
    verification_rule_version VARCHAR(80) NOT NULL DEFAULT 'verification_rule.v1',
    result JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_claim_verification_status ON claim_verification_lifecycle(status, next_retry_at);

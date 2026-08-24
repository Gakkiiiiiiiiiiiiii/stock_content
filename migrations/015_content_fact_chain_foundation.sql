-- Source -> Artifact -> Claim -> Verification -> Snapshot -> Signal foundation.
-- Additive migration; 012/013 tables and APIs remain compatible.
CREATE TABLE IF NOT EXISTS content_artifact (
    artifact_id VARCHAR(96) PRIMARY KEY,
    artifact_type VARCHAR(40) NOT NULL,
    schema_version VARCHAR(40) NOT NULL,
    producer_stage VARCHAR(80) NOT NULL,
    producer_version VARCHAR(80) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    parent_artifact_ids JSONB NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_content_artifact_type ON content_artifact(artifact_type);
-- Content-addressed artifacts are immutable and canonical: one payload per
-- artifact type/hash.  Existing duplicate rows intentionally fail this
-- upgrade instead of being silently merged.
CREATE UNIQUE INDEX IF NOT EXISTS uq_content_artifact_type_hash
    ON content_artifact(artifact_type, content_hash);
CREATE TABLE IF NOT EXISTS content_artifact_edge (
    edge_id VARCHAR(128) PRIMARY KEY,
    artifact_id VARCHAR(96) NOT NULL REFERENCES content_artifact(artifact_id) ON DELETE CASCADE,
    parent_artifact_id VARCHAR(96) NOT NULL,
    relation VARCHAR(40) NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_content_artifact_edge_artifact ON content_artifact_edge(artifact_id);
CREATE TABLE IF NOT EXISTS content_stage_checkpoint (
    checkpoint_id VARCHAR(128) PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL REFERENCES content_ingest_task(task_id) ON DELETE CASCADE,
    stage VARCHAR(80) NOT NULL,
    stage_version VARCHAR(80) NOT NULL,
    status VARCHAR(32) NOT NULL,
    artifact_ids JSONB NOT NULL,
    artifact_hashes JSONB NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS financial_claim (
    claim_id VARCHAR(96) PRIMARY KEY,
    claim_type VARCHAR(48) NOT NULL,
    fact_category VARCHAR(32) NOT NULL,
    subject_type VARCHAR(80) NOT NULL,
    subject_id VARCHAR(255) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    value JSONB,
    unit VARCHAR(40), currency VARCHAR(16),
    fact_time TIMESTAMPTZ, period_start TIMESTAMPTZ, period_end TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    source_confidence DOUBLE PRECISION NOT NULL,
    extractor_confidence DOUBLE PRECISION NOT NULL,
    extraction_model_id VARCHAR(120) NOT NULL, extraction_prompt_version VARCHAR(120) NOT NULL,
    condition_text TEXT, invalidation_text TEXT,
    claim_schema_version VARCHAR(40) NOT NULL, normalization_version VARCHAR(40) NOT NULL,
    source_support_status VARCHAR(24) NOT NULL DEFAULT 'UNSUPPORTED',
    payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_financial_claim_subject ON financial_claim(subject_id);
CREATE TABLE IF NOT EXISTS claim_evidence (
    member_id VARCHAR(128) PRIMARY KEY,
    claim_id VARCHAR(96) NOT NULL REFERENCES financial_claim(claim_id) ON DELETE CASCADE,
    evidence_id VARCHAR(96) NOT NULL, relation VARCHAR(32) NOT NULL,
    UNIQUE(claim_id, evidence_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_claim_evidence_claim_evidence
    ON claim_evidence(claim_id, evidence_id);
CREATE TABLE IF NOT EXISTS claim_artifact_member (
    member_id VARCHAR(128) PRIMARY KEY,
    artifact_id VARCHAR(96) NOT NULL REFERENCES content_artifact(artifact_id) ON DELETE CASCADE,
    claim_id VARCHAR(96) NOT NULL REFERENCES financial_claim(claim_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS claim_verification_job (
    job_id VARCHAR(96) PRIMARY KEY,
    claim_id VARCHAR(96) NOT NULL REFERENCES financial_claim(claim_id) ON DELETE CASCADE,
    provider VARCHAR(40) NOT NULL, status VARCHAR(32) NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0, max_retries INTEGER NOT NULL DEFAULT 5,
    next_retry_at TIMESTAMPTZ, lease_owner VARCHAR(128), lease_expires_at TIMESTAMPTZ,
    last_error TEXT, trace_id VARCHAR(96), created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE(claim_id, provider)
);
CREATE TABLE IF NOT EXISTS claim_verification_result (
    verification_id VARCHAR(96) PRIMARY KEY,
    claim_id VARCHAR(96) NOT NULL REFERENCES financial_claim(claim_id) ON DELETE CASCADE,
    provider VARCHAR(40) NOT NULL, status VARCHAR(32) NOT NULL,
    market_snapshot_id VARCHAR(128), market_data_version VARCHAR(80),
    result_payload JSONB NOT NULL, trace_id VARCHAR(96), created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS content_snapshot_artifact (
    member_id VARCHAR(128) PRIMARY KEY,
    content_snapshot_id VARCHAR(80) NOT NULL REFERENCES content_snapshot(content_snapshot_id) ON DELETE CASCADE,
    artifact_id VARCHAR(96) NOT NULL REFERENCES content_artifact(artifact_id), slot VARCHAR(40) NOT NULL
);
CREATE TABLE IF NOT EXISTS content_signal_outbox (
    outbox_id VARCHAR(128) PRIMARY KEY, signal_id VARCHAR(128) NOT NULL UNIQUE,
    payload JSONB NOT NULL, status VARCHAR(24) NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ, last_error TEXT, created_at TIMESTAMPTZ NOT NULL, published_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS content_source_head (
    source_identity_hash VARCHAR(64) PRIMARY KEY,
    latest_snapshot_id VARCHAR(80) NOT NULL,
    latest_verified_snapshot_id VARCHAR(80), updated_at TIMESTAMPTZ NOT NULL
);

-- Compatibility upgrades for databases already created from migrations 012/013.
-- PostgreSQL accepts ADD COLUMN IF NOT EXISTS; all additions are nullable or
-- have safe defaults so existing rows remain valid.
ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS source_artifact_id VARCHAR(96);
ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS artifact_root_hash VARCHAR(64);
ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS snapshot_kind VARCHAR(32) NOT NULL DEFAULT 'INITIAL';
ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS parent_snapshot_id VARCHAR(80);
ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS supersedes_snapshot_id VARCHAR(80);
ALTER TABLE content_snapshot ADD COLUMN IF NOT EXISTS producer_manifest JSONB NOT NULL DEFAULT '{}';

ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS currency VARCHAR(16);
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS period_start TIMESTAMPTZ;
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS period_end TIMESTAMPTZ;
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS extraction_model_id VARCHAR(120) NOT NULL DEFAULT 'unknown';
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS extraction_prompt_version VARCHAR(120) NOT NULL DEFAULT 'unknown';
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS condition_text TEXT;
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS invalidation_text TEXT;
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS claim_schema_version VARCHAR(40) NOT NULL DEFAULT 'claim.v2';
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS normalization_version VARCHAR(40) NOT NULL DEFAULT 'normalization.v1';
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS source_support_status VARCHAR(24) NOT NULL DEFAULT 'UNSUPPORTED';
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}';
ALTER TABLE financial_claim ADD COLUMN IF NOT EXISTS evidence_refs JSONB NOT NULL DEFAULT '[]';

-- PostgreSQL does not change an existing column's type when ADD COLUMN IF
-- NOT EXISTS is used, so explicitly normalize every core JSON column before
-- any JSONB operators below are evaluated. JSON is validated by PostgreSQL on
-- write, making these casts safe for every existing value.
ALTER TABLE content_snapshot
    ALTER COLUMN identity DROP DEFAULT,
    ALTER COLUMN identity TYPE JSONB USING identity::jsonb,
    ALTER COLUMN identity SET DEFAULT '{}'::jsonb,
    ALTER COLUMN artifact_ids DROP DEFAULT,
    ALTER COLUMN artifact_ids TYPE JSONB USING artifact_ids::jsonb,
    ALTER COLUMN artifact_ids SET DEFAULT '{}'::jsonb,
    ALTER COLUMN quant_market_snapshot_ids DROP DEFAULT,
    ALTER COLUMN quant_market_snapshot_ids TYPE JSONB USING quant_market_snapshot_ids::jsonb,
    ALTER COLUMN quant_market_snapshot_ids SET DEFAULT '[]'::jsonb,
    ALTER COLUMN producer_manifest DROP DEFAULT,
    ALTER COLUMN producer_manifest TYPE JSONB USING producer_manifest::jsonb,
    ALTER COLUMN producer_manifest SET DEFAULT '{}'::jsonb;
ALTER TABLE content_artifact
    ALTER COLUMN parent_artifact_ids TYPE JSONB USING parent_artifact_ids::jsonb,
    ALTER COLUMN payload TYPE JSONB USING payload::jsonb;
ALTER TABLE content_stage_checkpoint
    ALTER COLUMN artifact_ids TYPE JSONB USING artifact_ids::jsonb,
    ALTER COLUMN artifact_hashes TYPE JSONB USING artifact_hashes::jsonb,
    ALTER COLUMN payload TYPE JSONB USING payload::jsonb;
ALTER TABLE financial_claim
    ALTER COLUMN payload DROP DEFAULT,
    ALTER COLUMN payload TYPE JSONB USING payload::jsonb,
    ALTER COLUMN payload SET DEFAULT '{}'::jsonb,
    ALTER COLUMN value TYPE JSONB USING value::jsonb,
    ALTER COLUMN evidence_refs DROP DEFAULT,
    ALTER COLUMN evidence_refs TYPE JSONB USING evidence_refs::jsonb,
    ALTER COLUMN evidence_refs SET DEFAULT '[]'::jsonb;
ALTER TABLE claim_verification_result
    ALTER COLUMN result_payload DROP DEFAULT,
    ALTER COLUMN result_payload TYPE JSONB USING result_payload::jsonb,
    ALTER COLUMN result_payload SET DEFAULT '{}'::jsonb;
ALTER TABLE content_signal_outbox
    ALTER COLUMN payload TYPE JSONB USING payload::jsonb;

-- Rows created by migration 013 predate the canonical payload and evidence
-- membership tables.  Reconstruct a complete FinancialClaim payload from the
-- authoritative indexed columns instead of leaving ``payload = {}``, which
-- cannot pass domain validation after an upgrade.
UPDATE financial_claim
SET payload = jsonb_build_object(
    'claim_id', claim_id,
    'claim_type', claim_type,
    'fact_category', fact_category,
    'subject_type', subject_type,
    'subject_id', subject_id,
    'predicate', predicate,
    'value', value,
    'unit', unit,
    'currency', currency,
    'fact_time', fact_time,
    'period_start', period_start,
    'period_end', period_end,
    'published_at', published_at,
    'evidence_refs', COALESCE(evidence_refs, '[]'::jsonb),
    'source_support_status', COALESCE(source_support_status, 'UNSUPPORTED'),
    'source_confidence', source_confidence,
    'extractor_confidence', extractor_confidence,
    'extraction_model_id', COALESCE(extraction_model_id, 'unknown'),
    'extraction_prompt_version', COALESCE(extraction_prompt_version, 'unknown'),
    'condition_text', condition_text,
    'invalidation_text', invalidation_text,
    'claim_schema_version', COALESCE(claim_schema_version, 'claim.v2'),
    'normalization_version', COALESCE(normalization_version, 'normalization.v1')
)
WHERE payload IS NULL OR NOT (payload ? 'claim_type');

INSERT INTO claim_evidence(member_id, claim_id, evidence_id, relation)
SELECT md5(fc.claim_id || ':' || evidence_id), fc.claim_id, evidence_id, 'SUPPORTS'
FROM financial_claim AS fc
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(fc.evidence_refs, '[]'::jsonb)) AS refs(evidence_id)
ON CONFLICT(member_id) DO NOTHING;

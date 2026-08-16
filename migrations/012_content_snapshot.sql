-- 详细修改方案 §4 P0-2：ContentSnapshot 权威存储
-- PostgreSQL = Source of Truth；Qdrant 仅为可重建的检索投影。
CREATE TABLE IF NOT EXISTS content_snapshot (
    content_snapshot_id VARCHAR(80) PRIMARY KEY,
    source_type VARCHAR(32) NOT NULL,
    source_ref TEXT NOT NULL,
    source_content_hash VARCHAR(64) NOT NULL,
    identity JSONB NOT NULL DEFAULT '{}',
    artifact_ids JSONB NOT NULL DEFAULT '{}',
    quant_market_snapshot_ids JSONB NOT NULL DEFAULT '[]',
    pipeline_version VARCHAR(40) NOT NULL DEFAULT 'pipeline.v2',
    schema_version VARCHAR(40) NOT NULL DEFAULT 'content.snapshot.v1',
    code_sha VARCHAR(64),
    config_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_content_snapshot_source ON content_snapshot(source_type, source_content_hash);
CREATE INDEX IF NOT EXISTS idx_content_snapshot_created ON content_snapshot(created_at);

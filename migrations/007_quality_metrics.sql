CREATE TABLE IF NOT EXISTS knowledge_extraction_run (
    run_id VARCHAR(64) PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    extractor_version VARCHAR(80) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS knowledge_extraction_quality_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES knowledge_extraction_run(run_id) ON DELETE CASCADE,
    metric_name VARCHAR(80) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS video_asset (
    video_id VARCHAR(64) PRIMARY KEY,
    source_type VARCHAR(32) NOT NULL,
    source_ref TEXT NOT NULL,
    title TEXT NOT NULL,
    author VARCHAR(255),
    duration_seconds DOUBLE PRECISION,
    transcript_text TEXT NOT NULL DEFAULT '',
    source_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_segment (
    id BIGSERIAL PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    start_seconds DOUBLE PRECISION NOT NULL,
    end_seconds DOUBLE PRECISION NOT NULL,
    text TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    UNIQUE(video_id, segment_index)
);

CREATE TABLE IF NOT EXISTS video_chapter (
    chapter_id VARCHAR(64) PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    chapter_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    start_seconds DOUBLE PRECISION NOT NULL,
    end_seconds DOUBLE PRECISION NOT NULL,
    chapter_type VARCHAR(40) NOT NULL DEFAULT 'ANALYSIS',
    UNIQUE(video_id, chapter_index)
);

CREATE TABLE IF NOT EXISTS video_summary (
    video_id VARCHAR(64) PRIMARY KEY REFERENCES video_asset(video_id) ON DELETE CASCADE,
    core_summary TEXT NOT NULL,
    markdown TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS content_ingest_task (
    task_id VARCHAR(64) PRIMARY KEY,
    source_type VARCHAR(32) NOT NULL,
    source_ref TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    stage VARCHAR(40) NOT NULL DEFAULT 'queued',
    progress INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    lease_owner VARCHAR(128),
    lease_expires_at TIMESTAMPTZ,
    options JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_content_task_claim
    ON content_ingest_task(status, lease_expires_at, created_at);
CREATE INDEX IF NOT EXISTS ix_video_asset_source_hash ON video_asset(source_hash);

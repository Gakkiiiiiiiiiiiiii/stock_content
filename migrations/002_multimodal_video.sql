CREATE TABLE IF NOT EXISTS video_frame (
    id BIGSERIAL PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp_seconds DOUBLE PRECISION NOT NULL,
    storage_uri TEXT NOT NULL,
    ocr_text TEXT,
    vision_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(video_id, frame_index)
);

CREATE TABLE IF NOT EXISTS video_analysis_document (
    video_id VARCHAR(64) PRIMARY KEY REFERENCES video_asset(video_id) ON DELETE CASCADE,
    markdown TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

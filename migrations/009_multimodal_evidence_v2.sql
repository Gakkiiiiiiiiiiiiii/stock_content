CREATE TABLE IF NOT EXISTS video_frame (
    frame_id VARCHAR(64) PRIMARY KEY, video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    timestamp_ms INTEGER NOT NULL, image_hash VARCHAR(64) NOT NULL, extraction_reason VARCHAR(40) NOT NULL,
    storage_ref TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(video_id, timestamp_ms, image_hash)
);
CREATE TABLE IF NOT EXISTS ocr_evidence (
    id BIGSERIAL PRIMARY KEY, frame_id VARCHAR(64) NOT NULL REFERENCES video_frame(frame_id) ON DELETE CASCADE,
    timestamp_ms INTEGER NOT NULL, text TEXT NOT NULL, bbox JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION, ocr_engine VARCHAR(80) NOT NULL, engine_version VARCHAR(80)
);
CREATE TABLE IF NOT EXISTS vision_evidence (
    id BIGSERIAL PRIMARY KEY, frame_id VARCHAR(64) NOT NULL REFERENCES video_frame(frame_id) ON DELETE CASCADE,
    timestamp_ms INTEGER NOT NULL, label VARCHAR(80) NOT NULL, payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION, model_name VARCHAR(120), model_version VARCHAR(80)
);
CREATE TABLE IF NOT EXISTS temporal_window (
    window_id VARCHAR(64) PRIMARY KEY, video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, transcript TEXT NOT NULL DEFAULT '',
    speaker_ids JSONB NOT NULL DEFAULT '[]'::jsonb, frame_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ocr_items JSONB NOT NULL DEFAULT '[]'::jsonb, vision_items JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE TABLE IF NOT EXISTS financial_entity (
    entity_id VARCHAR(64) PRIMARY KEY, video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    raw_mention VARCHAR(255) NOT NULL, entity_type VARCHAR(40) NOT NULL, canonical_name VARCHAR(255),
    canonical_key VARCHAR(255), ticker VARCHAR(32), exchange VARCHAR(24), confidence DOUBLE PRECISION, resolution_source VARCHAR(80)
);
CREATE TABLE IF NOT EXISTS financial_numeric_fact (
    numeric_id VARCHAR(64) PRIMARY KEY, video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    metric VARCHAR(80), value DOUBLE PRECISION, unit VARCHAR(40), period VARCHAR(40), currency VARCHAR(16),
    comparison_type VARCHAR(40), qualifier VARCHAR(40), confidence DOUBLE PRECISION, evidence_ref VARCHAR(64)
);
CREATE TABLE IF NOT EXISTS financial_event (
    event_id VARCHAR(64) PRIMARY KEY, video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    event_type VARCHAR(60) NOT NULL, subject_key VARCHAR(255), objects JSONB NOT NULL DEFAULT '[]'::jsonb,
    event_time TIMESTAMPTZ, effective_time TIMESTAMPTZ, available_from TIMESTAMPTZ NOT NULL,
    direction VARCHAR(20), strength DOUBLE PRECISION, numeric_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb, confidence DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_financial_event_subject ON financial_event(subject_key);

CREATE TABLE IF NOT EXISTS financial_event (
    event_id VARCHAR(64) PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL REFERENCES video_asset(video_id) ON DELETE CASCADE,
    event_type VARCHAR(64) NOT NULL,
    subject VARCHAR(255),
    ticker VARCHAR(32),
    statement TEXT NOT NULL,
    event_time TIMESTAMPTZ,
    confidence DOUBLE PRECISION NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_financial_event_ticker_time ON financial_event(ticker, event_time);

CREATE TABLE IF NOT EXISTS event_evidence (
    id BIGSERIAL PRIMARY KEY,
    event_id VARCHAR(64) NOT NULL REFERENCES financial_event(event_id) ON DELETE CASCADE,
    evidence_type VARCHAR(40) NOT NULL,
    evidence_text TEXT NOT NULL,
    start_seconds DOUBLE PRECISION,
    end_seconds DOUBLE PRECISION
);

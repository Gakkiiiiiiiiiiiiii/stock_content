-- Legacy video_segment rows are backfilled by the application with explicit
-- legacy_trseg_ identifiers; video_id + segment_index is never authoritative.
ALTER TABLE video_segment ADD COLUMN IF NOT EXISTS segment_id varchar(128);
ALTER TABLE video_segment ADD COLUMN IF NOT EXISTS raw_text text;
UPDATE video_segment
SET segment_id = 'legacy_trseg_' || md5(
    coalesce(video_id, '') || '|' || segment_index::text || '|' ||
    start_seconds::text || '|' || end_seconds::text || '|' || coalesce(raw_text, text)
)
WHERE segment_id IS NULL OR segment_id = '';
CREATE UNIQUE INDEX IF NOT EXISTS uq_video_segment_segment_id ON video_segment(segment_id);
ALTER TABLE video_segment ALTER COLUMN segment_id SET NOT NULL;

CREATE TABLE IF NOT EXISTS semantic_segment (
    semantic_segment_id varchar(64) PRIMARY KEY,
    transcript_artifact_id varchar(128) NOT NULL,
    video_id varchar(64) NOT NULL,
    segment_index integer NOT NULL,
    start_segment_index integer NOT NULL,
    end_segment_index integer NOT NULL,
    start_segment_id varchar(128) NOT NULL,
    end_segment_id varchar(128) NOT NULL,
    start_ms bigint NOT NULL,
    end_ms bigint NOT NULL,
    topic text,
    subject varchar(255),
    segment_type varchar(40) NOT NULL,
    model_id varchar(160) NOT NULL,
    prompt_version varchar(80) NOT NULL,
    confidence double precision,
    artifact_id varchar(128),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(transcript_artifact_id, segment_index),
    CHECK (start_segment_index <= end_segment_index),
    CHECK (start_ms <= end_ms)
);

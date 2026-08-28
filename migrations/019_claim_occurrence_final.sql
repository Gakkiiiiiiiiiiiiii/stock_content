CREATE TABLE IF NOT EXISTS claim_occurrence (
    occurrence_id varchar(128) PRIMARY KEY, claim_id varchar(128) NOT NULL,
    source_artifact_id varchar(128) NOT NULL, transcript_artifact_id varchar(128) NOT NULL,
    semantic_segment_id varchar(64) NOT NULL, assertion_locator_hash varchar(128) NOT NULL,
    asserted_at timestamptz, source_published_at timestamptz, source_available_at timestamptz,
    source_availability_quality varchar(32) NOT NULL, ingested_at timestamptz NOT NULL,
    extraction_completed_at timestamptz NOT NULL, snapshot_committed_at timestamptz NOT NULL,
    available_from timestamptz NOT NULL, source_support_status varchar(40) NOT NULL,
    source_confidence double precision NOT NULL, extractor_confidence double precision NOT NULL,
    raw_temporal_expressions jsonb NOT NULL DEFAULT '[]', provenance jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (available_from >= ingested_at), CHECK (available_from >= extraction_completed_at),
    CHECK (available_from >= snapshot_committed_at)
);
CREATE TABLE IF NOT EXISTS claim_occurrence_evidence (
    occurrence_id varchar(128) NOT NULL, evidence_id varchar(128) NOT NULL,
    evidence_role varchar(32) NOT NULL, ordinal integer NOT NULL DEFAULT 0,
    PRIMARY KEY (occurrence_id, evidence_id, evidence_role)
);

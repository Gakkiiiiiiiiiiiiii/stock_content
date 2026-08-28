CREATE TABLE IF NOT EXISTS claim_temporal_binding (
    temporal_binding_id varchar(128) PRIMARY KEY, claim_id varchar(128) NOT NULL,
    role varchar(40) NOT NULL, scope varchar(32) NOT NULL, value_type varchar(16) NOT NULL,
    start_time timestamptz, end_time timestamptz, earliest_start_time timestamptz,
    latest_start_time timestamptz, earliest_end_time timestamptz, latest_end_time timestamptz,
    start_date date, end_date date, earliest_start_date date, latest_start_date date,
    earliest_end_date date, latest_end_date date, period_label varchar(64), raw_expression text,
    expression_key varchar(128), precision varchar(32) NOT NULL, granularity varchar(32),
    assertion_status varchar(32) NOT NULL, metric_temporal_nature varchar(32), confidence double precision,
    timezone varchar(64), calendar_type varchar(32), calendar_id varchar(64), market_session varchar(32),
    recurrence jsonb, normalization_status varchar(32) NOT NULL, normalization_reason text,
    normalization_version varchar(64) NOT NULL, source_evidence_refs jsonb NOT NULL DEFAULT '[]',
    reference_snapshot_id varchar(128), reference_data_version varchar(128), reference_available_at timestamptz,
    CHECK (value_type IN ('NONE','DATE','TIMESTAMP')),
    CHECK (value_type <> 'DATE' OR (start_time IS NULL AND end_time IS NULL AND earliest_start_time IS NULL AND latest_start_time IS NULL AND earliest_end_time IS NULL AND latest_end_time IS NULL)),
    CHECK (value_type <> 'TIMESTAMP' OR (start_date IS NULL AND end_date IS NULL AND earliest_start_date IS NULL AND latest_start_date IS NULL AND earliest_end_date IS NULL AND latest_end_date IS NULL)),
    CHECK (value_type <> 'NONE' OR (start_time IS NULL AND end_time IS NULL AND earliest_start_time IS NULL AND latest_start_time IS NULL AND earliest_end_time IS NULL AND latest_end_time IS NULL AND start_date IS NULL AND end_date IS NULL AND earliest_start_date IS NULL AND latest_start_date IS NULL AND earliest_end_date IS NULL AND latest_end_date IS NULL))
);
CREATE TABLE IF NOT EXISTS claim_temporal_relation (
    temporal_relation_id varchar(128) PRIMARY KEY, claim_id varchar(128) NOT NULL,
    relation_type varchar(32) NOT NULL, from_binding_id varchar(128) NOT NULL,
    to_binding_id varchar(128) NOT NULL, lag_value double precision, lag_unit varchar(32),
    lag_min double precision, lag_max double precision, confidence double precision
);

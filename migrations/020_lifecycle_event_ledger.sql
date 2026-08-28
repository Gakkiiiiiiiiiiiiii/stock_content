-- 008 created a legacy knowledge-only table. Preserve its rows under an
-- explicit compatibility name before taking the contractual table name. The
-- guarded block is safe both for a 001-019 upgrade and for a database where
-- create_schema has already materialised the authoritative table.
DO $$
BEGIN
    IF to_regclass('knowledge_lifecycle_event') IS NOT NULL
       AND to_regclass('knowledge_lifecycle_event_legacy') IS NULL THEN
        ALTER TABLE knowledge_lifecycle_event RENAME TO knowledge_lifecycle_event_legacy;
    END IF;
END $$;
CREATE TABLE IF NOT EXISTS knowledge_lifecycle_event (
    lifecycle_event_id varchar(128) PRIMARY KEY, target_type varchar(32) NOT NULL,
    target_id varchar(128) NOT NULL, from_status varchar(32), to_status varchar(32) NOT NULL,
    effective_at timestamptz NOT NULL, recorded_at timestamptz NOT NULL,
    reason_code varchar(80) NOT NULL, policy_version varchar(64) NOT NULL,
    supersedes_event_id varchar(128), created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (target_type IN ('CLAIM', 'OCCURRENCE'))
);
COMMENT ON TABLE knowledge_lifecycle_event IS 'Append-only lifecycle ledger; application must not UPDATE or DELETE events.';

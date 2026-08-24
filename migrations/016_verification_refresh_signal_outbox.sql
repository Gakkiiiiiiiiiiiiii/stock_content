-- C2: verification refresh transaction and durable v4 signal outbox.
ALTER TABLE claim_verification_result ADD COLUMN IF NOT EXISTS fact_date TIMESTAMPTZ;
ALTER TABLE claim_verification_result ADD COLUMN IF NOT EXISTS adjustment VARCHAR(32);
ALTER TABLE claim_verification_result ADD COLUMN IF NOT EXISTS verification_timestamp TIMESTAMPTZ;
ALTER TABLE claim_verification_result ADD COLUMN IF NOT EXISTS verification_rule_version VARCHAR(64);
ALTER TABLE claim_verification_result ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ;

ALTER TABLE content_signal_outbox ADD COLUMN IF NOT EXISTS content_snapshot_id VARCHAR(80);
ALTER TABLE content_signal_outbox ADD COLUMN IF NOT EXISTS claim_id VARCHAR(96);
ALTER TABLE content_signal_outbox ADD COLUMN IF NOT EXISTS schema_version VARCHAR(48) NOT NULL DEFAULT 'content-factor-signal.v4';
ALTER TABLE content_signal_outbox ADD COLUMN IF NOT EXISTS lease_owner VARCHAR(128);
ALTER TABLE content_signal_outbox ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_content_signal_outbox_due
    ON content_signal_outbox(status, next_attempt_at, lease_expires_at);

-- Preserve the old mutable lifecycle view in the new append-only result
-- ledger.  This runs after the additive result columns above, so a clean
-- 013 -> 015 -> 016 PostgreSQL upgrade never references a column too early.
-- The deterministic ID makes rerunning this migration harmless and keeps the
-- old status visible through the new verifications API.
DO $$
BEGIN
    IF to_regclass('public.claim_verification_lifecycle') IS NOT NULL THEN
        INSERT INTO claim_verification_result(
            verification_id, claim_id, provider, status, market_snapshot_id,
            market_data_version, result_payload, trace_id, fact_date, adjustment,
            verification_timestamp, verification_rule_version, verified_at, created_at
        )
        SELECT
            'legacy-' || md5(lifecycle.claim_id),
            lifecycle.claim_id,
            'legacy_lifecycle',
            lifecycle.status,
            lifecycle.market_snapshot_id,
            lifecycle.market_data_version,
            COALESCE(lifecycle.result, '{}'::jsonb)
                || jsonb_build_object(
                    'legacy_lifecycle', true,
                    'retry_count', lifecycle.retry_count,
                    'next_retry_at', lifecycle.next_retry_at
                ),
            NULL,
            lifecycle.fact_date,
            lifecycle.adjustment,
            lifecycle.verification_timestamp,
            COALESCE(lifecycle.verification_rule_version, 'verification_rule.v1'),
            lifecycle.verification_timestamp,
            COALESCE(lifecycle.updated_at, lifecycle.verification_timestamp, NOW())
        FROM claim_verification_lifecycle AS lifecycle
        ON CONFLICT(verification_id) DO NOTHING;
    END IF;
END $$;

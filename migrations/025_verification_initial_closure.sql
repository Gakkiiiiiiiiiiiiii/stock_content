-- P0-3: make verification result visibility an explicit PIT envelope.
-- NULL is preserved for old rows and is deliberately ineligible for reuse.
ALTER TABLE claim_verification_result
    ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_verification_result_claim_provider_available
    ON claim_verification_result (claim_id, provider, available_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS ux_verification_job_claim_provider
    ON claim_verification_job (claim_id, provider);

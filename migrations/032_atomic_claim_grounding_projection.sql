-- SC-07B: explicit accepted-draft grounding fields.  This migration is
-- additive and must run before application code writes the new projection.
ALTER TABLE financial_claim
  ADD COLUMN IF NOT EXISTS normalized_statement text,
  ADD COLUMN IF NOT EXISTS grounding_status varchar(32) NOT NULL DEFAULT 'LEGACY_UNGROUNDED',
  ADD COLUMN IF NOT EXISTS grounding_reason_codes jsonb NOT NULL DEFAULT '[]',
  ADD COLUMN IF NOT EXISTS contradiction_group_id varchar(96),
  ADD COLUMN IF NOT EXISTS legacy_grounding_incomplete boolean NOT NULL DEFAULT true;

ALTER TABLE claim_occurrence
  ADD COLUMN IF NOT EXISTS primary_quote text,
  ADD COLUMN IF NOT EXISTS normalized_statement text,
  ADD COLUMN IF NOT EXISTS grounding_status varchar(32) NOT NULL DEFAULT 'LEGACY_UNGROUNDED',
  ADD COLUMN IF NOT EXISTS grounding_reason_codes jsonb NOT NULL DEFAULT '[]',
  ADD COLUMN IF NOT EXISTS contradiction_group_id varchar(96),
  ADD COLUMN IF NOT EXISTS claim_schema_version varchar(40) NOT NULL DEFAULT 'claim.legacy.v1',
  ADD COLUMN IF NOT EXISTS legacy_grounding_incomplete boolean NOT NULL DEFAULT true;

CREATE INDEX IF NOT EXISTS ix_financial_claim_formal_grounding
  ON financial_claim(grounding_status, legacy_grounding_incomplete, claim_schema_version, claim_id);
CREATE INDEX IF NOT EXISTS ix_claim_occurrence_formal_grounding
  ON claim_occurrence(grounding_status, legacy_grounding_incomplete, claim_schema_version, claim_id);

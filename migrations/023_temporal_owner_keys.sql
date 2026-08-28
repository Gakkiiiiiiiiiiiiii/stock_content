-- Temporal identities are semantic/content identities and may be shared by
-- multiple claims.  Claim ownership and per-claim provenance are therefore
-- part of the persistence key, not the domain identity.
DO $$
BEGIN
    IF to_regclass('claim_temporal_binding') IS NOT NULL THEN
        ALTER TABLE claim_temporal_binding
            DROP CONSTRAINT IF EXISTS claim_temporal_binding_pkey;
        ALTER TABLE claim_temporal_binding
            ADD CONSTRAINT claim_temporal_binding_pkey
            PRIMARY KEY (claim_id, temporal_binding_id);
    END IF;

    IF to_regclass('claim_temporal_relation') IS NOT NULL THEN
        ALTER TABLE claim_temporal_relation
            DROP CONSTRAINT IF EXISTS claim_temporal_relation_pkey;
        ALTER TABLE claim_temporal_relation
            ADD CONSTRAINT claim_temporal_relation_pkey
            PRIMARY KEY (claim_id, temporal_relation_id);
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS ix_claim_temporal_binding_identity
    ON claim_temporal_binding (temporal_binding_id);
CREATE INDEX IF NOT EXISTS ix_claim_temporal_relation_identity
    ON claim_temporal_relation (temporal_relation_id);

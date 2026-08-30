-- P0-1: canonical final claims never own source-specific evidence.
-- The precheck is intentionally fail-closed: an unmigrated membership must
-- not be silently discarded when no authoritative occurrence lineage exists.
DO $$
BEGIN
    IF to_regclass('financial_claim') IS NULL
       OR to_regclass('claim_evidence') IS NULL THEN
        RETURN;
    END IF;

    IF to_regclass('claim_occurrence') IS NULL
       OR to_regclass('claim_occurrence_evidence') IS NULL THEN
        IF EXISTS (
            SELECT 1
            FROM claim_evidence ce
            JOIN financial_claim fc ON fc.claim_id = ce.claim_id
            WHERE fc.claim_schema_version = 'claim.final.v1'
        ) THEN
            RAISE EXCEPTION
                'cannot remove final claim_evidence: authoritative occurrence lineage is missing';
        END IF;
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM claim_evidence ce
        JOIN financial_claim fc ON fc.claim_id = ce.claim_id
        WHERE fc.claim_schema_version = 'claim.final.v1'
          AND NOT EXISTS (
              SELECT 1
              FROM claim_occurrence co
              JOIN claim_occurrence_evidence coe
                ON coe.occurrence_id = co.occurrence_id
              WHERE co.claim_id = ce.claim_id
                AND coe.evidence_id = ce.evidence_id
          )
    ) THEN
        RAISE EXCEPTION
            'cannot remove final claim_evidence: authoritative occurrence lineage is incomplete';
    END IF;

    DELETE FROM claim_evidence ce
    USING financial_claim fc
    WHERE ce.claim_id = fc.claim_id
      AND fc.claim_schema_version = 'claim.final.v1';
END
$$;

CREATE OR REPLACE FUNCTION reject_final_claim_evidence()
RETURNS trigger AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM financial_claim
        WHERE claim_id = NEW.claim_id
          AND claim_schema_version = 'claim.final.v1'
    ) THEN
        RAISE EXCEPTION
            'claim.final.v1 cannot own source-specific claim_evidence';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF to_regclass('claim_evidence') IS NOT NULL
       AND to_regclass('financial_claim') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_reject_final_claim_evidence ON claim_evidence;
        CREATE TRIGGER trg_reject_final_claim_evidence
        BEFORE INSERT OR UPDATE ON claim_evidence
        FOR EACH ROW
        EXECUTE FUNCTION reject_final_claim_evidence();
    END IF;
END
$$;

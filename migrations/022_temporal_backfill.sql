-- Conservative legacy backfill. Migration time is the only honest recorded_at.
INSERT INTO knowledge_lifecycle_event (
    lifecycle_event_id, target_type, target_id, from_status, to_status,
    effective_at, recorded_at, reason_code, policy_version, created_at
)
SELECT
    'le_legacy_' || md5('CLAIM|' || fc.claim_id || '|LEGACY_IMPORTED'),
    'CLAIM', fc.claim_id, NULL,
    COALESCE((SELECT old.to_status FROM knowledge_lifecycle_event_legacy old
              WHERE old.knowledge_uid = fc.claim_id ORDER BY old.event_time DESC LIMIT 1), 'LEGACY_IMPORTED'),
    COALESCE(fc.created_at, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP,
    'LEGACY_IMPORTED', 'migration.022', CURRENT_TIMESTAMP
FROM financial_claim fc
WHERE NOT EXISTS (
    SELECT 1 FROM knowledge_lifecycle_event le
    WHERE le.target_type = 'CLAIM' AND le.target_id = fc.claim_id
      AND le.reason_code = 'LEGACY_IMPORTED'
);
-- Method/concept has no defensible date: add only a timeless NONE binding.
INSERT INTO claim_temporal_binding (
    temporal_binding_id, claim_id, role, scope, value_type, precision,
    assertion_status, normalization_status, normalization_version, source_evidence_refs
)
SELECT 'tb_legacy_' || md5(fc.claim_id || '|TIMELESS'), fc.claim_id,
       'VALID_AT', 'TIMELESS', 'NONE', 'UNKNOWN', 'ACTUAL', 'NORMALIZED',
       'temporal-normalization.final.v1', '[]'
FROM financial_claim fc
WHERE upper(coalesce(fc.fact_category, '')) IN ('METHOD', 'CONCEPT')
  AND NOT EXISTS (SELECT 1 FROM claim_temporal_binding tb
                  WHERE tb.claim_id = fc.claim_id AND tb.normalization_status = 'PARTIAL');

-- Forecast as_of is not observed. Preserve it as a label, with no guessed
-- calendar precision and value_type NONE.
INSERT INTO claim_temporal_binding (
    temporal_binding_id, claim_id, role, scope, value_type, period_label,
    precision, assertion_status, normalization_status, normalization_version,
    source_evidence_refs
)
SELECT 'tb_legacy_' || md5(fc.claim_id || '|FORECAST_TARGET'), fc.claim_id,
       'FORECAST_TARGET', 'FORECAST', 'NONE', cast(fc.fact_time AS text),
       'UNKNOWN', 'EXPECTED', 'PARTIAL', 'temporal-normalization.final.v1', '[]'
FROM financial_claim fc
WHERE upper(coalesce(fc.fact_category, '')) = 'FORECAST'
  AND fc.fact_time IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM claim_temporal_binding tb
                  WHERE tb.claim_id = fc.claim_id AND tb.role = 'FORECAST_TARGET');

INSERT INTO claim_temporal_binding (
    temporal_binding_id, claim_id, role, scope, value_type, precision,
    assertion_status, normalization_status, normalization_version,
    source_evidence_refs
)
SELECT 'tb_legacy_' || md5(fc.claim_id || '|FORECAST_TARGET'), fc.claim_id,
       'FORECAST_TARGET', 'FORECAST', 'NONE', 'UNKNOWN', 'EXPECTED', 'UNRESOLVED',
       'temporal-normalization.final.v1', '[]'
FROM financial_claim fc
WHERE upper(coalesce(fc.fact_category, '')) = 'FORECAST'
  AND fc.fact_time IS NULL
  AND NOT EXISTS (SELECT 1 FROM claim_temporal_binding tb
                  WHERE tb.claim_id = fc.claim_id AND tb.role = 'FORECAST_TARGET');

-- FACT as_of/fact_time is intentionally not rewritten as OBSERVED_AT.
-- Do not map legacy valid_to to FORECAST_TARGET.

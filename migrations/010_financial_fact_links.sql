ALTER TABLE financial_numeric_fact ADD COLUMN IF NOT EXISTS knowledge_uid VARCHAR(64);
ALTER TABLE financial_numeric_fact ADD COLUMN IF NOT EXISTS raw_text TEXT;
ALTER TABLE financial_numeric_fact ADD COLUMN IF NOT EXISTS as_of_time TIMESTAMPTZ;
ALTER TABLE financial_numeric_fact ADD COLUMN IF NOT EXISTS available_from TIMESTAMPTZ;
ALTER TABLE financial_event ADD COLUMN IF NOT EXISTS knowledge_uid VARCHAR(64);
CREATE INDEX IF NOT EXISTS ix_financial_numeric_fact_knowledge_uid ON financial_numeric_fact(knowledge_uid);
CREATE INDEX IF NOT EXISTS ix_financial_event_knowledge_uid ON financial_event(knowledge_uid);

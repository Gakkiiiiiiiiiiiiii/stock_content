-- A Bundle is immutable and can exist without an HTTP retry key.  A separate
-- mapping table is therefore required: adding a key onto the Bundle row would
-- either violate its immutability trigger or make an older unkeyed Bundle
-- unreplayable.  The key itself is domain-separated and hashed by the adapter.
CREATE TABLE IF NOT EXISTS content_knowledge_bundle_idempotency (
  idempotency_key_hash varchar(80) PRIMARY KEY,
  idempotency_request_hash varchar(80) NOT NULL,
  bundle_id varchar(80) NOT NULL REFERENCES content_knowledge_bundle(bundle_id) ON DELETE RESTRICT,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_content_knowledge_bundle_idempotency_bundle
  ON content_knowledge_bundle_idempotency(bundle_id);

CREATE OR REPLACE FUNCTION reject_content_knowledge_bundle_idempotency_update() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'content_knowledge_bundle_idempotency is immutable'; END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_content_knowledge_bundle_idempotency_immutable
  ON content_knowledge_bundle_idempotency;
CREATE TRIGGER trg_content_knowledge_bundle_idempotency_immutable
  BEFORE UPDATE ON content_knowledge_bundle_idempotency
  FOR EACH ROW EXECUTE FUNCTION reject_content_knowledge_bundle_idempotency_update();

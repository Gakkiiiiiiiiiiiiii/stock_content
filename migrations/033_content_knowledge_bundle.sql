CREATE TABLE IF NOT EXISTS content_knowledge_bundle (
  bundle_id varchar(80) PRIMARY KEY,
  bundle_hash varchar(80) NOT NULL UNIQUE,
  content_snapshot_id varchar(80) NOT NULL,
  request_hash varchar(80) NOT NULL,
  contract_version varchar(80) NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  producer_git_commit varchar(128) NOT NULL,
  pipeline_version varchar(80) NOT NULL,
  UNIQUE(content_snapshot_id, request_hash, bundle_hash)
);
CREATE OR REPLACE FUNCTION reject_content_knowledge_bundle_update() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'content_knowledge_bundle is immutable; create a new bundle'; END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_content_knowledge_bundle_immutable ON content_knowledge_bundle;
CREATE TRIGGER trg_content_knowledge_bundle_immutable BEFORE UPDATE ON content_knowledge_bundle
FOR EACH ROW EXECUTE FUNCTION reject_content_knowledge_bundle_update();

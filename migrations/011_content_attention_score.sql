ALTER TABLE knowledge_cross_video ADD COLUMN IF NOT EXISTS content_attention_score DOUBLE PRECISION NOT NULL DEFAULT 0;

-- 详细修改方案 §5 P1-6：向量投影状态（Qdrant 可从 PostgreSQL 全量重建）
CREATE TABLE IF NOT EXISTS vector_projection_state (
    collection VARCHAR(128) PRIMARY KEY,
    collection_version VARCHAR(40) NOT NULL,
    embedding_model VARCHAR(120) NOT NULL,
    embedding_version VARCHAR(40) NOT NULL,
    chunk_version VARCHAR(40) NOT NULL,
    indexed_count INTEGER NOT NULL DEFAULT 0,
    source_of_truth VARCHAR(20) NOT NULL DEFAULT 'postgres',
    rebuilt_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

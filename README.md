# stock_content

Financial Content Intelligence Service. This repository owns content ingestion,
transcription, chaptering, knowledge/evidence, summaries, retrieval, and the
`content-factor-signal.v3` producer contract. It never imports `stock_agent` or
`stock_factor`.

## 可信、可追溯、可验证、可重放的金融内容事实层

```text
Raw Financial Content -> Source Snapshot -> Evidence Artifact -> FinancialClaim
    -> Fact Verification -> Knowledge Unit -> ContentSnapshot
    -> Search / Factor Signal / Agent Evidence
```

- 强类型 Artifact Pipeline（`domain/artifacts.py`），Stage 间不再用字符串 key 传核心对象；
- ContentSnapshot 内容寻址身份（`domain/lineage.py`）：源内容 + 模型 + Prompt + code SHA + config + Quant 快照；
- 三概念幂等：`request_idempotency_key` / `source_identity_hash` / `source_content_hash`；
- Artifact Checkpoint v2：断点恢复前校验 artifact 哈希与 stage 版本；
- FinancialClaim 区分 Fact / Forecast / Opinion，核验必须绑定 Quant market_snapshot_id；
- Quant 不可用时核验进入 VERIFICATION_PENDING 退避重试，不影响 ingest 主链；
- PostgreSQL = Source of Truth，Qdrant 可用 `scripts/rebuild_vector_index.py --from-postgres` 全量重建。

## Current migration status

The first independent vertical slice is operational:

```text
POST ingest -> PostgreSQL task -> leased worker -> source/media/ASR
            -> transcript -> chapter -> knowledge/evidence -> verification
            -> summary -> PostgreSQL + Qdrant -> search/factor-signals
```

Implemented source adapters are Bilibili (yt-dlp) and authorized Xiaoe HLS
(ffmpeg). ASR uses faster-whisper. Tests may provide a transcript fixture in
`options.transcript`, keeping the complete post-ASR pipeline deterministic.

OCR, vision, external fact verification, conflict resolution, lifecycle jobs,
and full golden video-accuracy coverage remain subsequent migration slices.

## Run

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The API listens on `http://localhost:8100`; the worker is a separate process.
For a lightweight local run without infrastructure, omit `CONTENT_QDRANT_URL`
and use `CONTENT_DATABASE_URL=sqlite:///./stock_content.db`.

```powershell
python -m uvicorn stock_content.api.main:app --host 0.0.0.0 --port 8100
stock-content-worker
```

## Migrations

Numbered PostgreSQL migrations in `migrations/` establish content-owned tables.
The service also calls SQLAlchemy `create_all` so SQLite-backed development and
tests are self-contained; production deployments should apply the SQL files in
order before starting API and worker processes.

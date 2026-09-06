# stock_content

Financial Content Intelligence Service. This repository owns content ingestion,
transcription, chaptering, knowledge/evidence, summaries, retrieval, and the
`content-factor-signal.v5.1` formal producer contract. It never imports `stock_agent` or
`stock_factor`.

## 可信、可追溯、可验证、可重放的金融内容事实层

```text
Raw Financial Content -> Source Snapshot -> Evidence Artifact -> FinancialClaim
    -> Fact Verification -> Knowledge Unit -> ContentSnapshot
    -> Search / Formal v5.1 Factor Signal / Agent Evidence
```

- 强类型 Artifact Pipeline（`domain/artifacts.py`），Stage 间不再用字符串 key 传核心对象；
- ContentSnapshot 内容寻址身份（`domain/lineage.py`）：源内容 + 模型 + Prompt + code SHA + config + Quant 快照；
- 三概念幂等：`request_idempotency_key` / `source_identity_hash` / `source_content_hash`；
- Artifact Checkpoint v2：断点恢复前校验 artifact 哈希与 stage 版本；
- FinancialClaim 区分 Fact / Forecast / Opinion，核验必须绑定 Quant market_snapshot_id；
- Quant 不可用时核验进入 VERIFICATION_PENDING 退避重试，不影响 ingest 主链；
- PostgreSQL = Source of Truth；Qdrant 是可从 SQL 重建的搜索派生物。

## Implemented, local evidence

The first independent vertical slice is implemented and locally testable:

```text
POST ingest -> PostgreSQL task -> leased worker -> source/media/ASR
            -> transcript -> chapter -> knowledge/evidence -> verification
            -> summary -> PostgreSQL + Qdrant -> search/factor-signals
```

Implemented source adapters are Bilibili (yt-dlp) and authorized Xiaoe HLS
(ffmpeg). ASR uses faster-whisper. Tests may provide a transcript fixture in
`options.transcript`, keeping the complete post-ASR pipeline deterministic.

OCR, vision, external fact verification, conflict resolution and lifecycle
jobs are implemented as explicit pipeline capabilities. The formal producer
contract is `content-factor-signal.v5.1`: every formal request binds
`business_as_of`, `knowledge_as_of`, `availability_as_of`, and a
`content_snapshot_id`. v3/v4/v5 responses remain compatibility-only and are
not formal evidence.

Readiness reports three independently evaluated capabilities: SQL-backed
`read_only_facts`, SQL-backed `formal_publish`, and derived Qdrant
`derived_search`. A Qdrant failure or an unknown index watermark degrades
search; it does not relabel SQL fact or formal-signal authority as unavailable.
Source intake is governed by `source-policy.v1`; immutable source metadata
contains matching `source-governance-evidence.v1` and `pii-redaction.v1`
records. Validation fails closed if those versioned records are absent or drift
from the artifact metadata. See the [source-governance policy](docs/security/source-governance.md)
and [readiness SLO](docs/slo/fact-authority.md).

Run the strict contract and quality checks with:

```powershell
python scripts/contracts/verify_manifest.py
python scripts/generate_sbom.py --profile core
python -m pytest tests/test_p1_p2_foundations.py -q
```

Replay and vector rebuild operations are documented in the
[runbooks](docs/runbooks/README.md), and the
[v3-to-v5.1 migration guide](docs/migrations/signal-v3-to-v5_1.md) describes
the compatibility boundary.

## Evidence boundary

This repository has deterministic contract, replay, readiness, and governance
coverage. The following external exercises are **BLOCKED**, so this README
does not claim their completion: a cross-repository v5.1 consumer exercise, a
live PostgreSQL-to-Qdrant rebuild drill, and a durable Qdrant index-watermark
control plane. Their prerequisites are recorded in
[pending work](docs/pending-work-2026-09-04.md).

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

# stock_content

Financial Content Intelligence Service. This repository owns content ingestion,
transcription, chaptering, knowledge/evidence, summaries, retrieval, and the
`content-factor-signal.v1` producer contract. It never imports `stock_agent` or
`stock_factor`.

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

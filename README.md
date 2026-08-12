# stock_content

Financial Content Intelligence Service.

This repository owns video/audio ingestion, multimodal understanding, atomic
knowledge, evidence/verification, lifecycle management, knowledge search, and
the `content-factor-signal.v1` producer contract. It never imports
`stock_agent` or `stock_factor`.

## Run

```powershell
python -m uvicorn stock_content.api.main:app --host 0.0.0.0 --port 8100
```

The initial scaffold deliberately exposes stable contracts and the pipeline
boundary before migrated media adapters and repositories are connected.

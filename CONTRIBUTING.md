# Contributing

Install test dependencies with `python -m pip install -e ".[test]"`.
Before opening a change, run:

```powershell
python -m ruff check src tests scripts
python scripts/contracts/verify_manifest.py
python -m pytest tests/test_architecture.py tests/test_p1_p2_foundations.py -q
```

Formal contract changes must update the platform manifest checksum and retain
compatibility metadata. Domain code must remain independent of FastAPI,
SQLAlchemy and Qdrant; SQL is the fact authority and Qdrant is derivative.

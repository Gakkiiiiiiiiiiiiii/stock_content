# Qdrant rebuild runbook

PostgreSQL is the authoritative source and Qdrant is a derived index. Run the
installed command from the service image so its dependency profile matches the
deployed code:

```sh
docker compose exec -T stock-content stock-content-rebuild-index \
  --from-postgres --collection knowledge_v3 --dry-run
```

For a host execution, install the declared profiles first:

```sh
python -m pip install -e ".[test,postgres,search]"
stock-content-rebuild-index --from-postgres --collection knowledge_v3
```

The JSON output is the rebuild report. A Qdrant outage must not roll back SQL;
rerun the command after the index is healthy and retain the report with the
deployment artifacts.

# Content database migration

`content-migrate` is the only production process allowed to create, alter, or
backfill the PostgreSQL content schema.  It takes a transaction-scoped
PostgreSQL advisory lock, applies the numbered `migrations/*.sql` files in
order, and records each file's SHA-256 in `content_schema_migrations`.

Deploy the one-shot `content-migrate` job before starting the API or any
worker.  Runtime processes only inspect the catalog and require the ledger to
contain exactly the migrations packaged by their release.  A missing,
incomplete, unexpected, or checksum-mismatched ledger fails startup closed.

On an empty PostgreSQL schema, the job creates the current mapped baseline and
then executes and catalog-verifies the SQL-only authority from migrations 024
(the final-claim evidence trigger) and 026 (the single-successor expression
index) before recording the historical ledger.  The job is repeatable: a
matching ledger and verified catalog produce no DDL.  A duplicate legacy key
is reported before a simple unique-index migration is attempted; resolve that
conflict from a backup/change plan and retry the job.  Do not delete or edit
ledger rows to force a version: runtime also rejects a complete ledger when a
required catalog guard is absent or malformed.

For explicit local fixtures only, `Database(...sqlite...).create_schema()`
creates a SQLite schema.  It rejects PostgreSQL URLs and is never an API or
worker startup operation.

The migrations are expand/contract schema history and do not include automatic
destructive rollback or legacy-row rewrites.  Roll forward with a corrected
numbered migration; use the deployment backup and an approved restore
procedure for a rollback.

## EPIC-043 legacy rows

Migrations 030 and 032 are DDL-only.  Their defaults classify rows without
canonical EPIC-043 evidence as unresolved or legacy-un-grounded, so runtime
Bundle eligibility remains fail-closed.  If an approved change plan needs the
same explicit safety markers for a pre-EPIC-043 database, run the separate,
idempotent procedure only after schema migration and backup review:

```text
python scripts/backfill_epic043_legacy_rows.py --database-url <approved-postgres-url> --confirm
```

That script is not called by `content-migrate`, schema bootstrap, the API, or
workers.  It never promotes historical claims; re-running SC-07A is required
before a claim can become `GROUNDED` and Bundle-eligible.

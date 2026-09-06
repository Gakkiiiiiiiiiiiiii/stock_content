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

The migrations are expand/contract history and do not include automatic
destructive rollback.  Roll forward with a corrected numbered migration; use
the deployment backup and an approved restore procedure for a rollback.

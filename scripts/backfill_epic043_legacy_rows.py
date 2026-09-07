"""Explicit, idempotent safety marking for pre-EPIC-043 legacy rows.

This operational procedure is intentionally separate from numbered schema
migrations.  It is never invoked by ``content-migrate``, the API, or workers.
Run it only after the numbered schema migration has completed and after a
reviewed backup/change plan has approved the supplied PostgreSQL database.
"""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import create_engine, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True, help="Approved PostgreSQL database URL")
    parser.add_argument("--confirm", action="store_true", help="Apply the idempotent safety markers")
    args = parser.parse_args(argv)
    if not args.confirm:
        parser.error("refusing to mutate rows without --confirm")

    engine = create_engine(args.database_url, pool_pre_ping=True)
    if engine.dialect.name != "postgresql":
        parser.error("EPIC-043 legacy backfill requires PostgreSQL")

    with engine.begin() as connection:
        task_count = connection.execute(
            text(
                "UPDATE content_ingest_task SET task_kind = 'legacy_unresolved' "
                "WHERE status = 'PENDING' AND (request_hash IS NULL OR request_hash = '') "
                "AND task_kind <> 'legacy_unresolved'"
            )
        ).rowcount
        claim_count = connection.execute(
            text(
                "UPDATE financial_claim SET legacy_grounding_incomplete = true, "
                "grounding_status = 'LEGACY_UNGROUNDED' "
                "WHERE grounding_status <> 'GROUNDED' OR normalized_statement IS NULL"
            )
        ).rowcount
        occurrence_count = connection.execute(
            text(
                "UPDATE claim_occurrence SET legacy_grounding_incomplete = true, "
                "grounding_status = 'LEGACY_UNGROUNDED' "
                "WHERE grounding_status <> 'GROUNDED' OR primary_quote IS NULL "
                "OR normalized_statement IS NULL"
            )
        ).rowcount
    sys.stdout.write(
        json.dumps(
            {
                "content_ingest_task": task_count,
                "financial_claim": claim_count,
                "claim_occurrence": occurrence_count,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The sole production entry point permitted to mutate the content schema."""

from __future__ import annotations

import os

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.migration_ledger import apply_migrations


def main() -> None:
    database = Database(os.environ.get("CONTENT_DATABASE_URL"))
    applied = apply_migrations(database.engine)
    print("content migrations verified" if not applied else f"content migrations applied: {', '.join(applied)}")


if __name__ == "__main__":  # pragma: no cover - installed command entry point
    main()

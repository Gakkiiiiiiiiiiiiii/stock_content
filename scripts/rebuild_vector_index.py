"""Backward-compatible wrapper for the installed rebuild command."""
from stock_content.cli.rebuild_index import (  # noqa: F401
    _load_knowledge_from_postgres,
    main,
    rebuild_vector_index,
)

__all__ = ["main", "rebuild_vector_index"]

if __name__ == "__main__":
    main()

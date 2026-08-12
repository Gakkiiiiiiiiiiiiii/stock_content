from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from stock_content.adapters.postgres.models import Base


class Database:
    """Owns the Content database engine and transaction boundary."""

    def __init__(self, url: str | None = None) -> None:
        resolved_url = url or os.getenv("CONTENT_DATABASE_URL", "sqlite:///./stock_content.db")
        connect_args = {"check_same_thread": False} if resolved_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(resolved_url, pool_pre_ping=True, connect_args=connect_args)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

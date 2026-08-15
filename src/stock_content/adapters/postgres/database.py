from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect
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
        # The split service can be deployed against the early minimal schema.
        # Keep upgrades additive so persisted videos/tasks are never discarded.
        additions = {
            "content_ingest_task": {
                "checkpoint": "JSON",
                "input_hash": "VARCHAR(64)",
                "idempotency_key": "VARCHAR(128)",
                "trace_id": "VARCHAR(64)",
            },
            "video_asset": {
                "canonical_url": "TEXT",
                "published_at": "TIMESTAMP",
                "source_version": "VARCHAR(128)",
                "metadata": "JSON",
                "resolved_at": "TIMESTAMP",
            },
            "video_segment": {
                "raw_text": "TEXT",
                "normalized_text": "TEXT",
                "speaker_id": "VARCHAR(64)",
                "speaker_confidence": "FLOAT",
                "correction_records": "JSON",
            },
            "knowledge_unit": {
                "knowledge_kind": "VARCHAR(40)",
                "knowledge_version": "INTEGER",
                "predicate_key": "VARCHAR(255)",
                "subject_key": "VARCHAR(255)",
                "truth_status": "VARCHAR(40)",
                "lifecycle_status": "VARCHAR(20)",
                "valid_from": "TIMESTAMP",
                "valid_to": "TIMESTAMP",
                "source_statement_hash": "VARCHAR(64)",
                "content_hash": "VARCHAR(64)",
                "attributes": "JSON",
                "provenance": "JSON",
            },
            "knowledge_evidence": {
                "source_id": "VARCHAR(128)",
                "video_id": "VARCHAR(64)",
                "frame_id": "VARCHAR(64)",
                "structured_payload": "JSON",
                "confidence": "FLOAT",
                "source_reliability": "FLOAT",
            },
            "knowledge_cross_video": {"author_attention_score": "FLOAT"},
        }
        with self.engine.begin() as connection:
            inspector = inspect(connection)
            for table, columns in additions.items():
                if table not in inspector.get_table_names():
                    continue
                existing = {column["name"] for column in inspector.get_columns(table)}
                for name, declaration in columns.items():
                    if name not in existing:
                        connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

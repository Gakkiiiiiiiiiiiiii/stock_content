from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from stock_content.adapters.postgres.legacy_ids import (
    legacy_evidence_member_id,
    legacy_verification_id,
)
from stock_content.adapters.postgres.models import Base
from stock_content.domain.artifacts import legacy_transcript_segment_id


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
        json_declaration = "JSONB" if self.engine.dialect.name == "postgresql" else "JSON"
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
                "segment_id": "VARCHAR(128)",
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
            "knowledge_cross_video": {"author_attention_score": "FLOAT", "content_attention_score": "FLOAT"},
            "content_snapshot": {
                "source_artifact_id": "VARCHAR(96)",
                "artifact_root_hash": "VARCHAR(64)",
                "snapshot_kind": "VARCHAR(32)",
                "parent_snapshot_id": "VARCHAR(80)",
                "supersedes_snapshot_id": "VARCHAR(80)",
                "producer_manifest": json_declaration,
                "lifecycle_artifact_id": "VARCHAR(128)",
            },
            "content_snapshot_artifact": {
                "artifact_role": "VARCHAR(40)",
            },
            "financial_claim": {
                "currency": "VARCHAR(16)",
                "period_start": "TIMESTAMP",
                "period_end": "TIMESTAMP",
                "extraction_model_id": "VARCHAR(120)",
                "extraction_prompt_version": "VARCHAR(120)",
                "condition_text": "TEXT",
                "invalidation_text": "TEXT",
                "claim_schema_version": "VARCHAR(40)",
                "normalization_version": "VARCHAR(40)",
                "source_support_status": "VARCHAR(24)",
                "payload": json_declaration,
            },
            "claim_verification_result": {
                "fact_date": "TIMESTAMP",
                "adjustment": "VARCHAR(32)",
                "verification_timestamp": "TIMESTAMP",
                "verification_rule_version": "VARCHAR(64)",
                "verified_at": "TIMESTAMP",
                "available_at": "TIMESTAMP",
            },
            "content_signal_outbox": {
                "content_snapshot_id": "VARCHAR(80)",
                "claim_id": "VARCHAR(96)",
                "schema_version": "VARCHAR(48)",
                "lease_owner": "VARCHAR(128)",
                "lease_expires_at": "TIMESTAMP",
            },
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

            # ``create_all`` adds the constraint for fresh databases, while
            # additive upgrades must protect pre-existing 015 tables too.
            # Let duplicate legacy rows fail explicitly here; silently
            # selecting a winner would make content-addressed identity
            # ambiguous.
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_content_artifact_type_hash "
                "ON content_artifact (artifact_type, content_hash)"
            )
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_claim_evidence_claim_evidence "
                "ON claim_evidence (claim_id, evidence_id)"
            )
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_video_segment_segment_id ON video_segment (segment_id)"
            )
            # P0-3 PIT lookup and one durable job per canonical claim/provider.
            # Base.metadata covers fresh databases; these additive statements
            # also upgrade databases created before migration 025.
            if "claim_verification_result" in inspector.get_table_names():
                connection.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_verification_result_claim_provider_available "
                    "ON claim_verification_result (claim_id, provider, available_at DESC)"
                )
            if "claim_verification_job" in inspector.get_table_names():
                connection.exec_driver_sql(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_verification_job_claim_provider "
                    "ON claim_verification_job (claim_id, provider)"
                )

            # Some application-created PostgreSQL databases received these
            # columns as JSON before migration 015. Normalize them here too,
            # so JSONB operators and the ORM agree regardless of entrypoint.
            if connection.dialect.name == "postgresql":
                jsonb_columns = (
                    ("content_snapshot", "identity", "'{}'::jsonb"),
                    ("content_snapshot", "artifact_ids", "'{}'::jsonb"),
                    ("content_snapshot", "quant_market_snapshot_ids", "'[]'::jsonb"),
                    ("content_snapshot", "producer_manifest", "'{}'::jsonb"),
                    ("content_artifact", "parent_artifact_ids", None),
                    ("content_artifact", "payload", None),
                    ("content_stage_checkpoint", "artifact_ids", None),
                    ("content_stage_checkpoint", "artifact_hashes", None),
                    ("content_stage_checkpoint", "payload", None),
                    ("financial_claim", "value", None),
                    ("financial_claim", "payload", "'{}'::jsonb"),
                    ("financial_claim", "evidence_refs", "'[]'::jsonb"),
                    ("claim_verification_result", "result_payload", "'{}'::jsonb"),
                    ("content_signal_outbox", "payload", None),
                )
                for table, column, default in jsonb_columns:
                    if table not in inspect(connection).get_table_names():
                        continue
                    columns = {item["name"]: item["type"] for item in inspect(connection).get_columns(table)}
                    if column not in columns or "JSONB" in str(columns[column]).upper():
                        continue
                    alter = (
                        f"ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT, "
                        f"ALTER COLUMN {column} TYPE JSONB USING {column}::jsonb"
                    )
                    if default:
                        alter += f", ALTER COLUMN {column} SET DEFAULT {default}"
                    connection.exec_driver_sql(alter)

            # 015 is additive for databases that were initialized from 013.
            # The old financial_claim table has no payload column and keeps
            # evidence_refs separately.  Backfill the new canonical payload
            # before repositories attempt to validate a legacy row.  This is
            # deliberately done here as well as in the PostgreSQL migration so
            # SQLite development databases have the same upgrade semantics.
            self._upgrade_legacy_claims(connection)
            self._upgrade_legacy_segments(connection)

    @staticmethod
    def _upgrade_legacy_segments(connection) -> None:
        """Backfill legacy rows without using video/index as authoritative ID."""
        inspector = inspect(connection)
        if "video_segment" not in inspector.get_table_names():
            return
        columns = {column["name"] for column in inspector.get_columns("video_segment")}
        if "segment_id" not in columns:
            return
        rows = connection.execute(
            text(
                "SELECT id, video_id, segment_index, start_seconds, end_seconds, text, raw_text "
                "FROM video_segment WHERE segment_id IS NULL OR segment_id = ''"
            )
        ).mappings()
        for row in rows:
            segment_id = legacy_transcript_segment_id(
                int(row["segment_index"]),
                float(row["start_seconds"]),
                float(row["end_seconds"]),
                row["raw_text"] if row["raw_text"] is not None else row["text"],
                legacy_namespace=row["video_id"],
            )
            connection.execute(
                text("UPDATE video_segment SET segment_id = :segment_id WHERE id = :id"),
                {"segment_id": segment_id, "id": row["id"]},
            )

    @staticmethod
    def _upgrade_legacy_claims(connection) -> None:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if "financial_claim" not in tables:
            return

        claim_columns = {column["name"] for column in inspector.get_columns("financial_claim")}
        required = {
            "claim_id", "claim_type", "fact_category", "subject_type", "subject_id",
            "predicate", "value", "unit", "fact_time", "published_at",
            "source_confidence", "extractor_confidence", "extraction_model_id",
            "extraction_prompt_version", "condition_text", "invalidation_text",
            "claim_schema_version", "normalization_version", "source_support_status", "payload",
        }
        if not required <= claim_columns:
            return

        def json_value(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            return value

        # Select the legacy evidence_refs column only when it exists.  New
        # 015 tables intentionally do not carry that denormalized column.
        evidence_expression = "evidence_refs" if "evidence_refs" in claim_columns else "NULL"
        rows = connection.execute(
            text(
                "SELECT claim_id, claim_type, fact_category, subject_type, subject_id, predicate, "
                "value, unit, currency, fact_time, period_start, period_end, published_at, "
                "source_confidence, extractor_confidence, extraction_model_id, "
                "extraction_prompt_version, condition_text, invalidation_text, claim_schema_version, "
                "normalization_version, source_support_status, payload, " + evidence_expression + " "
                "FROM financial_claim"
            )
        ).mappings()
        for row in rows:
            payload = json_value(row.get("payload"))
            if isinstance(payload, dict) and payload.get("claim_type"):
                canonical = payload
            else:
                evidence_refs = json_value(row.get("evidence_refs")) or []
                if not isinstance(evidence_refs, list):
                    evidence_refs = [str(evidence_refs)]
                canonical = {
                    "claim_id": row["claim_id"],
                    "claim_type": row["claim_type"],
                    "fact_category": row["fact_category"] or "FACT",
                    "subject_type": row["subject_type"],
                    "subject_id": row["subject_id"],
                    "predicate": row["predicate"],
                    "value": json_value(row["value"]),
                    "unit": row["unit"],
                    "currency": row.get("currency"),
                    "fact_time": json_value(row.get("fact_time")),
                    "period_start": json_value(row.get("period_start")),
                    "period_end": json_value(row.get("period_end")),
                    "published_at": json_value(row.get("published_at")),
                    "evidence_refs": evidence_refs,
                    "source_support_status": row.get("source_support_status") or "UNSUPPORTED",
                    "source_confidence": row["source_confidence"],
                    "extractor_confidence": row["extractor_confidence"],
                    "extraction_model_id": row.get("extraction_model_id") or "unknown",
                    "extraction_prompt_version": row.get("extraction_prompt_version") or "unknown",
                    "condition_text": row.get("condition_text"),
                    "invalidation_text": row.get("invalidation_text"),
                    "claim_schema_version": row.get("claim_schema_version") or "claim.v2",
                    "normalization_version": row.get("normalization_version") or "normalization.v1",
                }
                connection.execute(
                    text("UPDATE financial_claim SET payload = :payload WHERE claim_id = :claim_id"),
                    {"payload": json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), default=str),
                     "claim_id": row["claim_id"]},
                )

            is_final = (
                canonical.get("claim_schema_version") == "claim.final.v1"
                or row.get("claim_schema_version") == "claim.final.v1"
            )
            if is_final and canonical.get("evidence_refs"):
                # Final claims do not own source evidence.  Normalize an old
                # payload projection during bootstrap; authoritative evidence
                # remains in claim_occurrence_evidence.
                canonical = {**canonical, "evidence_refs": []}
                connection.execute(
                    text("UPDATE financial_claim SET payload = :payload WHERE claim_id = :claim_id"),
                    {"payload": json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), default=str),
                     "claim_id": row["claim_id"]},
                )
            evidence_refs = canonical.get("evidence_refs") or []
            if is_final:
                # Do not recreate claim_evidence for final claims while
                # upgrading a legacy database.  Migration 024 performs the
                # guarded cleanup and installs the PostgreSQL trigger.
                continue
            if "claim_evidence" not in tables or not isinstance(evidence_refs, list):
                continue
            for evidence_id in evidence_refs:
                evidence_id = str(evidence_id)
                member_id = legacy_evidence_member_id(row["claim_id"], evidence_id)
                connection.execute(
                    text(
                        "INSERT INTO claim_evidence(member_id, claim_id, evidence_id, relation) "
                        "VALUES (:member_id, :claim_id, :evidence_id, 'SUPPORTS') "
                        "ON CONFLICT(member_id) DO NOTHING"
                    ),
                    {"member_id": member_id, "claim_id": row["claim_id"], "evidence_id": evidence_id},
                )

        # 013's lifecycle table is retained for old deployments.  Copy each
        # row into the append-only result ledger with a deterministic ID, so
        # the new verifications API exposes the old status without requiring
        # callers to know which schema version created it.
        if "claim_verification_lifecycle" not in tables or "claim_verification_result" not in tables:
            return
        lifecycle_columns = {column["name"] for column in inspector.get_columns("claim_verification_lifecycle")}
        expected_lifecycle = {
            "claim_id", "status", "retry_count", "next_retry_at", "market_snapshot_id",
            "market_data_version", "fact_date", "adjustment", "verification_timestamp",
            "verification_rule_version", "result", "updated_at",
        }
        if not expected_lifecycle <= lifecycle_columns:
            return
        lifecycle_rows = connection.execute(text("SELECT * FROM claim_verification_lifecycle")).mappings()
        for row in lifecycle_rows:
            verification_id = legacy_verification_id(row["claim_id"])
            result = json_value(row.get("result")) or {}
            if not isinstance(result, dict):
                result = {"legacy_result": result}
            result = {
                **result,
                "legacy_lifecycle": True,
                "retry_count": row.get("retry_count", 0),
                "next_retry_at": json_value(row.get("next_retry_at")),
            }
            connection.execute(
                text(
                    "INSERT INTO claim_verification_result "
                    "(verification_id, claim_id, provider, status, market_snapshot_id, market_data_version, "
                    "result_payload, trace_id, fact_date, adjustment, verification_timestamp, "
                    "verification_rule_version, verified_at, created_at) "
                    "VALUES (:verification_id, :claim_id, 'legacy_lifecycle', :status, :market_snapshot_id, "
                    ":market_data_version, :result_payload, NULL, :fact_date, :adjustment, "
                    ":verification_timestamp, :verification_rule_version, :verified_at, :created_at) "
                    "ON CONFLICT(verification_id) DO NOTHING"
                )
            , {
                "verification_id": verification_id,
                "claim_id": row["claim_id"],
                "status": row["status"],
                "market_snapshot_id": row.get("market_snapshot_id"),
                "market_data_version": row.get("market_data_version"),
                "result_payload": json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str),
                "fact_date": row.get("fact_date"),
                "adjustment": row.get("adjustment"),
                "verification_timestamp": row.get("verification_timestamp"),
                "verification_rule_version": row.get("verification_rule_version") or "verification_rule.v1",
                "verified_at": row.get("verification_timestamp"),
                "created_at": row.get("updated_at") or row.get("verification_timestamp") or datetime.now(),
            })

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

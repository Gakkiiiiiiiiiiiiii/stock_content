from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import FinancialEventRow, FinancialNumericFactRow, KnowledgeEvidenceRow


class PostgresFinancialRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def replace(self, video_id: str, numeric_facts: list[dict], events: list[dict]) -> None:
        with self._sessions.begin() as session:
            session.execute(delete(FinancialNumericFactRow).where(FinancialNumericFactRow.video_id == video_id))
            session.execute(delete(FinancialEventRow).where(FinancialEventRow.video_id == video_id))
            for index, item in enumerate(numeric_facts):
                digest = hashlib.sha256(
                    f"{video_id}:{item.get('knowledge_uid')}:{index}:{item.get('raw_expression')}".encode()
                ).hexdigest()[:32]
                session.add(
                    FinancialNumericFactRow(
                        numeric_id=str(item.get("numeric_id") or f"num_{digest}"),
                        video_id=video_id,
                        knowledge_uid=item.get("knowledge_uid"),
                        raw_text=item.get("raw_expression"),
                        metric=item.get("metric"),
                        value=item.get("value"),
                        unit=item.get("unit"),
                        period=item.get("period"),
                        currency=item.get("currency"),
                        comparison_type=item.get("comparator"),
                        qualifier="APPROX" if item.get("approximate") else None,
                        confidence=item.get("confidence"),
                        evidence_ref=item.get("evidence_ref"),
                        as_of_time=self._timestamp(item.get("as_of_time")),
                        available_from=self._timestamp(item.get("available_from")),
                    )
                )
            for item in events:
                # Financial facts reference the immutable KnowledgeEvidence
                # rows, not transient stage-local names.  Source ids are kept
                # as a fallback for old ingests that pre-date the evidence
                # table.
                evidence_rows = session.scalars(
                    select(KnowledgeEvidenceRow).where(KnowledgeEvidenceRow.knowledge_uid == item.get("knowledge_uid"))
                ).all()
                source_ids = set(str(value) for value in item.get("evidence_refs") or [])
                evidence_refs = [
                    str(row.id) for row in evidence_rows if not source_ids or str(row.source_id) in source_ids
                ]
                if not evidence_refs:
                    evidence_refs = [str(row.id) for row in evidence_rows]
                session.add(
                    FinancialEventRow(
                        event_id=str(item["event_id"]),
                        video_id=video_id,
                        knowledge_uid=item.get("knowledge_uid"),
                        event_type=str(item["event_type"]),
                        subject_key=item.get("subject_key"),
                        objects=[],
                        event_time=self._timestamp(item.get("event_time")),
                        effective_time=self._timestamp(item.get("effective_time")),
                        available_from=self._timestamp(item.get("available_from")) or datetime.now(UTC),
                        direction=item.get("direction"),
                        strength=item.get("strength"),
                        numeric_refs=list(item.get("numeric_refs") or []),
                        evidence_refs=evidence_refs,
                        confidence=item.get("confidence"),
                    )
                )

    @staticmethod
    def _timestamp(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return None

    def list_events(self, video_id: str) -> list[dict]:
        with self._sessions() as session:
            return [
                {
                    "event_id": row.event_id,
                    "knowledge_uid": row.knowledge_uid,
                    "event_type": row.event_type,
                    "numeric_refs": row.numeric_refs,
                    "evidence_refs": row.evidence_refs,
                }
                for row in session.scalars(select(FinancialEventRow).where(FinancialEventRow.video_id == video_id))
            ]

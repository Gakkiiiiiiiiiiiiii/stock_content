from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import KnowledgeCrossVideoRow, KnowledgeEvidenceRow, KnowledgeUnitRow
from stock_content.domain.cross_video_corroboration import CrossVideoCorroborationService
from stock_content.domain.models import KnowledgeUnit

_SUPPORT_RANK = {"UNSUPPORTED": 0, "SOURCE_SUPPORTED": 1, "CROSS_VERIFIED": 2, "FACT_VERIFIED": 3}


class PostgresKnowledgeRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    @staticmethod
    def _payload(row: KnowledgeUnitRow) -> dict:
        return {
            "knowledge_uid": row.knowledge_uid,
            "video_id": row.video_id,
            "chapter_id": row.chapter_id,
            "statement": row.statement,
            "kind": row.kind,
            "knowledge_kind": row.knowledge_kind,
            "knowledge_version": row.knowledge_version,
            "subject": row.subject,
            "subject_key": row.subject_key,
            "predicate_key": row.predicate_key,
            "ticker": row.ticker,
            "sentiment": row.sentiment,
            "support_status": row.support_status,
            "truth_status": row.truth_status,
            "review_status": row.review_status,
            "lifecycle_status": row.lifecycle_status,
            "confidence": row.confidence,
            "as_of": row.as_of.isoformat(),
            "as_of_time": row.as_of.isoformat(),
            "available_from": row.available_from.isoformat(),
            "valid_from": row.valid_from.isoformat() if row.valid_from else None,
            "valid_to": row.valid_to.isoformat() if row.valid_to else None,
            "provenance": dict(row.provenance or {}),
        }

    @staticmethod
    def _apply_filters(query, filters: dict):
        for field in ("kind", "ticker", "subject", "support_status", "review_status", "video_id"):
            value = filters.get(field)
            if value:
                query = query.where(getattr(KnowledgeUnitRow, field) == value)
        return query

    def replace_for_video(self, video_id: str, units: list[KnowledgeUnit]) -> None:
        with self._sessions.begin() as session:
            existing = {
                row.knowledge_uid: row
                for row in session.scalars(select(KnowledgeUnitRow).where(KnowledgeUnitRow.video_id == video_id))
            }
            incoming = {unit.knowledge_uid for unit in units}
            for knowledge_uid, row in existing.items():
                if knowledge_uid not in incoming:
                    row.lifecycle_status = "SUPERSEDED"
            for unit in units:
                values = vars(unit)
                row = existing.get(unit.knowledge_uid)
                if row is None:
                    row = KnowledgeUnitRow(**values)
                    session.add(row)
                else:
                    for name, value in values.items():
                        setattr(row, name, value)
                session.execute(
                    delete(KnowledgeEvidenceRow).where(KnowledgeEvidenceRow.knowledge_uid == unit.knowledge_uid)
                )
                evidence_items = list((unit.attributes or {}).get("evidence") or [])
                for evidence in evidence_items:
                    session.add(
                        KnowledgeEvidenceRow(
                            knowledge_uid=unit.knowledge_uid,
                            evidence_type=str(
                                evidence.get("evidence_type") or evidence.get("source_type") or "TRANSCRIPT"
                            ),
                            source_id=str(evidence.get("source_id") or "") or None,
                            video_id=video_id,
                            frame_id=str(evidence.get("frame_id") or "") or None,
                            evidence_text=str(evidence.get("text") or evidence.get("evidence_text") or ""),
                            start_seconds=evidence.get("start_seconds"),
                            end_seconds=evidence.get("end_seconds"),
                            structured_payload=dict(evidence.get("structured_payload") or {}),
                            confidence=evidence.get("confidence") or evidence.get("confidence_score"),
                            source_reliability=evidence.get("source_reliability"),
                        )
                    )
            self._refresh_cross_video(session, units)

    @staticmethod
    def _refresh_cross_video(session, units: list[KnowledgeUnit]) -> None:
        """Persist corroboration once at write time; read APIs never invent it."""
        affected = {
            (unit.subject_key or unit.ticker or "", unit.predicate_key or unit.knowledge_kind) for unit in units
        }
        rows = session.scalars(select(KnowledgeUnitRow).where(KnowledgeUnitRow.lifecycle_status == "ACTIVE")).all()
        payload = [
            {
                "knowledge_uid": row.knowledge_uid,
                "video_id": row.video_id,
                "subject_key": row.subject_key,
                "predicate_key": row.predicate_key,
                "sentiment": row.sentiment,
                "support_status": row.support_status,
                "lifecycle_status": row.lifecycle_status,
                "evidence_ids": [
                    str(value)
                    for value in session.scalars(
                        select(KnowledgeEvidenceRow.id).where(KnowledgeEvidenceRow.knowledge_uid == row.knowledge_uid)
                    ).all()
                ],
            }
            for row in rows
            if (row.subject_key or row.ticker or "", row.predicate_key or row.knowledge_kind) in affected
        ]
        for knowledge_uid, values in CrossVideoCorroborationService().calculate(payload).items():
            state = session.get(KnowledgeCrossVideoRow, knowledge_uid)
            # Keep the legacy column as an alias until all clients move to the
            # unambiguous content_attention_score contract.
            values["author_attention_score"] = values["content_attention_score"]
            if state is None:
                session.add(KnowledgeCrossVideoRow(knowledge_uid=knowledge_uid, **values))
            else:
                for name, value in values.items():
                    setattr(state, name, value)

    def get(self, knowledge_uid: str) -> dict | None:
        with self._sessions() as session:
            row = session.get(KnowledgeUnitRow, knowledge_uid)
            return self._payload(row) if row else None

    def list_for_video(self, video_id: str, limit: int) -> list[dict]:
        with self._sessions() as session:
            rows = session.scalars(
                select(KnowledgeUnitRow)
                .where(KnowledgeUnitRow.video_id == video_id)
                .order_by(KnowledgeUnitRow.available_from, KnowledgeUnitRow.knowledge_uid)
                .limit(limit)
            ).all()
            return [self._payload(row) for row in rows]

    def search(self, query: str, filters: dict, limit: int) -> list[dict]:
        with self._sessions() as session:
            statement = select(KnowledgeUnitRow).where(
                or_(
                    KnowledgeUnitRow.statement.ilike(f"%{query}%"),
                    KnowledgeUnitRow.subject.ilike(f"%{query}%"),
                    KnowledgeUnitRow.ticker.ilike(f"%{query}%"),
                )
            )
            statement = self._apply_filters(statement, filters)
            rows = session.scalars(statement.order_by(KnowledgeUnitRow.confidence.desc()).limit(limit)).all()
            return [self._payload(row) for row in rows]

    def hydrate(self, knowledge_uids: list[str], filters: dict) -> list[dict]:
        if not knowledge_uids:
            return []
        with self._sessions() as session:
            statement = select(KnowledgeUnitRow).where(KnowledgeUnitRow.knowledge_uid.in_(knowledge_uids))
            statement = self._apply_filters(statement, filters)
            rows = session.scalars(statement).all()
            by_uid = {row.knowledge_uid: self._payload(row) for row in rows}
            return [by_uid[uid] for uid in knowledge_uids if uid in by_uid]

    def factor_signals(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        minimum_support_status: str,
    ) -> list[dict]:
        with self._sessions() as session:
            statement = select(KnowledgeUnitRow).where(
                KnowledgeUnitRow.available_from >= start,
                KnowledgeUnitRow.available_from <= end,
                KnowledgeUnitRow.review_status != "REJECTED",
                KnowledgeUnitRow.lifecycle_status == "ACTIVE",
            )
            rows = session.scalars(statement.order_by(KnowledgeUnitRow.available_from)).all()
            minimum = _SUPPORT_RANK.get(minimum_support_status, 1)
            requested_codes = {_ticker_code(symbol) for symbol in symbols if _ticker_code(symbol)}
            eligible = [
                row
                for row in rows
                if _SUPPORT_RANK.get(row.support_status, 0) >= minimum
                and (not requested_codes or _ticker_code(row.ticker or row.subject_key or "") in requested_codes)
            ]
            items = []
            for row in eligible:
                cross = session.get(KnowledgeCrossVideoRow, row.knowledge_uid)
                evidence_ids = [
                    str(value)
                    for value in session.scalars(
                        select(KnowledgeEvidenceRow.id).where(KnowledgeEvidenceRow.knowledge_uid == row.knowledge_uid)
                    ).all()
                ]
                items.append(
                    {
                        "signal_id": row.knowledge_uid,
                        "knowledge_uid": row.knowledge_uid,
                        "knowledge_version": row.knowledge_version,
                        "symbol": row.subject_key or row.ticker,
                        "subject": row.subject,
                        "subject_key": row.subject_key or row.ticker or row.subject,
                        "kind": row.kind,
                        "knowledge_kind": row.knowledge_kind,
                        "event_type": (row.attributes or {}).get("event_type"),
                        "sentiment": row.sentiment,
                        "confidence": row.confidence,
                        "support_status": row.support_status,
                        "truth_status": row.truth_status,
                        "review_status": row.review_status,
                        "lifecycle_status": row.lifecycle_status,
                        "as_of_time": row.as_of.isoformat(),
                        "available_from": row.available_from.isoformat(),
                        "content_attention_score": cross.content_attention_score if cross else 0.0,
                        "author_attention_score": cross.content_attention_score if cross else 0.0,
                        "event_strength": (row.attributes or {}).get("event_strength", row.confidence),
                        "cross_video_consensus": cross.consensus_score if cross else 0.0,
                        "cross_video_disagreement": cross.disagreement_score if cross else 0.0,
                        "source_video_id": row.video_id,
                        "evidence_ids": evidence_ids,
                        "provenance": dict(row.provenance or {}),
                    }
                )
            return items


def _ticker_code(value: str) -> str | None:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", str(value))
    return match.group(1) if match else None

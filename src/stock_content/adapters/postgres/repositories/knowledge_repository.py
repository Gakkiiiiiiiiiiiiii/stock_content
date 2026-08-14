from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import KnowledgeEvidenceRow, KnowledgeUnitRow
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
            "subject": row.subject,
            "ticker": row.ticker,
            "sentiment": row.sentiment,
            "support_status": row.support_status,
            "review_status": row.review_status,
            "confidence": row.confidence,
            "as_of": row.as_of.isoformat(),
            "available_from": row.available_from.isoformat(),
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
            existing = select(KnowledgeUnitRow.knowledge_uid).where(KnowledgeUnitRow.video_id == video_id)
            session.execute(delete(KnowledgeEvidenceRow).where(KnowledgeEvidenceRow.knowledge_uid.in_(existing)))
            session.execute(delete(KnowledgeUnitRow).where(KnowledgeUnitRow.video_id == video_id))
            for unit in units:
                session.add(KnowledgeUnitRow(**vars(unit)))
                session.add(
                    KnowledgeEvidenceRow(
                        knowledge_uid=unit.knowledge_uid,
                        evidence_type="TRANSCRIPT",
                        evidence_text=unit.statement,
                    )
                )

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
            )
            if symbols:
                statement = statement.where(KnowledgeUnitRow.ticker.in_(symbols))
            rows = session.scalars(statement.order_by(KnowledgeUnitRow.available_from)).all()
            minimum = _SUPPORT_RANK.get(minimum_support_status, 1)
            eligible = [row for row in rows if _SUPPORT_RANK.get(row.support_status, 0) >= minimum]
            by_subject: dict[str, list[KnowledgeUnitRow]] = {}
            for row in eligible:
                subject_key = row.ticker or (row.subject or "").strip().casefold()
                if subject_key:
                    by_subject.setdefault(subject_key, []).append(row)

            items = []
            for row in eligible:
                subject_key = row.ticker or (row.subject or "").strip().casefold()
                peers = by_subject.get(subject_key, [row])
                comparable = [peer for peer in peers if peer.sentiment != "NEUTRAL"]
                same_sentiment = sum(peer.sentiment == row.sentiment for peer in comparable)
                consensus = same_sentiment / len(comparable) if comparable else 0.0
                items.append(
                    {
                        "signal_id": row.knowledge_uid,
                        "knowledge_uid": row.knowledge_uid,
                        "symbol": row.ticker,
                        "subject": row.subject,
                        "subject_key": subject_key,
                        "kind": row.kind,
                        "knowledge_kind": _knowledge_kind(row.kind),
                        "sentiment": row.sentiment,
                        "confidence": row.confidence,
                        "support_status": row.support_status,
                        "truth_status": "NOT_CHECKED",
                        "review_status": _review_status(row.review_status),
                        "as_of": row.as_of.isoformat(),
                        "as_of_time": row.as_of.isoformat(),
                        "available_from": row.available_from.isoformat(),
                        "source_video_id": row.video_id,
                        "author_attention_score": 1.0,
                        "cross_video_consensus": consensus,
                        "cross_video_disagreement": 1.0 - consensus if comparable else 0.0,
                    }
                )
            return items


def _knowledge_kind(kind: str) -> str:
    return {
        "CATALYST": "CAUSAL_THESIS",
        "RISK": "RISK_CONDITION",
        "EARNINGS": "FINANCIAL_METRIC",
    }.get(kind, kind if kind in {"VALUATION", "FACT", "POLICY_FACT", "STATE"} else "STATE")


def _review_status(status: str) -> str:
    return "UNREVIEWED" if status == "PENDING" else status

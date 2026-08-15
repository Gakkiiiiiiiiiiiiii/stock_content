from __future__ import annotations

import hashlib

from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import FinancialEntityRow
from stock_content.domain.financial_entity_normalizer import FinancialEntityNormalizer
from stock_content.domain.models import KnowledgeUnit


class PostgresFinancialEntityRepository:
    """The canonical entity authority; callers never infer symbols on read."""

    def __init__(self, sessions: sessionmaker, normalizer: FinancialEntityNormalizer | None = None) -> None:
        self._sessions, self._normalizer = sessions, normalizer or FinancialEntityNormalizer()

    def replace(self, video_id: str, units: list[KnowledgeUnit]) -> None:
        with self._sessions.begin() as session:
            session.execute(delete(FinancialEntityRow).where(FinancialEntityRow.video_id == video_id))
            seen: set[str] = set()
            for unit in units:
                for entity in self._normalizer.extract_entities(unit.statement):
                    raw = str(entity["name"])
                    key = self._canonical_key(str(entity.get("ticker") or raw))
                    identity = f"{raw}:{key}"
                    if identity in seen:
                        continue
                    seen.add(identity)
                    entity_id = "ent_" + hashlib.sha256(f"{video_id}:{identity}".encode()).hexdigest()[:32]
                    session.add(
                        FinancialEntityRow(
                            entity_id=entity_id,
                            video_id=video_id,
                            raw_mention=raw,
                            entity_type=str(entity.get("entity_type") or "UNKNOWN"),
                            canonical_name=raw,
                            canonical_key=key,
                            ticker=str(entity.get("ticker") or "") or None,
                            exchange=key.split(".", 1)[0] if key else None,
                            confidence=1.0,
                            resolution_source="financial_entity_normalizer",
                        )
                    )

    def list_for_video(self, video_id: str) -> list[dict]:
        with self._sessions() as session:
            return [
                {"entity_id": row.entity_id, "canonical_key": row.canonical_key, "ticker": row.ticker}
                for row in session.scalars(select(FinancialEntityRow).where(FinancialEntityRow.video_id == video_id))
            ]

    @staticmethod
    def _canonical_key(value: str) -> str:
        token = value.upper().strip()
        if token.isdigit() and len(token) == 6:
            return f"CN.A.{token}"
        if token.endswith(".HK"):
            return f"HK.{token.split('.', 1)[0]}"
        if token.endswith((".SH", ".SZ")):
            return f"CN.A.{token.split('.', 1)[0]}"
        if token.isalpha() and 1 < len(token) <= 5:
            return f"US.{token}"
        return token

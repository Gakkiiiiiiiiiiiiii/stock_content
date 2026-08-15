from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import KnowledgeVerificationRow
from stock_content.domain.models import KnowledgeUnit


class PostgresVerificationRepository:
    """Append-only audit trail for each knowledge verification decision."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    def append(self, units: list[KnowledgeUnit], trace_id: str | None = None) -> None:
        with self._sessions.begin() as session:
            for unit in units:
                verification = dict((unit.attributes or {}).get("verification") or {})
                judge = dict(verification.get("judge") or {})
                reason_codes = list(verification.get("reason_codes") or [])
                decision = str(verification.get("support_status") or unit.support_status)
                session.add(
                    KnowledgeVerificationRow(
                        verification_id=f"ver_{uuid4().hex}",
                        knowledge_uid=unit.knowledge_uid,
                        verifier_type=str(verification.get("verifier_type") or "CLAIM_EVIDENCE"),
                        decision=decision,
                        confidence=_number(verification.get("support_probability"), unit.confidence),
                        reason_code=reason_codes[0] if reason_codes else None,
                        model_name=judge.get("model"),
                        model_version=judge.get("version"),
                        prompt_version=judge.get("prompt_version"),
                        raw_output={
                            "support_status": decision,
                            "support_probability": _number(verification.get("support_probability"), unit.confidence),
                            "reason_codes": reason_codes,
                            "checks": dict(verification.get("checks") or {}),
                            "judge": judge,
                            "truth_status": unit.truth_status,
                            "review_status": unit.review_status,
                            "knowledge_version": unit.knowledge_version,
                            "available_from": unit.available_from.isoformat(),
                            "trace_id": trace_id,
                        },
                        created_at=datetime.now(UTC),
                    )
                )


def _number(value: object, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default

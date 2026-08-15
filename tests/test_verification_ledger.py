from datetime import UTC, datetime

from sqlalchemy import select

from stock_content.adapters.postgres import Database
from stock_content.adapters.postgres.models import KnowledgeVerificationRow
from stock_content.adapters.postgres.repositories.knowledge_repository import PostgresKnowledgeRepository
from stock_content.adapters.postgres.repositories.verification_repository import PostgresVerificationRepository
from stock_content.domain.models import KnowledgeUnit


def test_verification_decisions_are_appended_with_audit_context(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'content.db'}")
    database.create_schema()
    now = datetime(2026, 8, 15, tzinfo=UTC)
    unit = KnowledgeUnit(
        knowledge_uid="knowledge-1", video_id="video-1", chapter_id=None, statement="来源支持该结论。",
        available_from=now,
        attributes={"verification": {"support_status": "SOURCE_SUPPORTED", "support_probability": 0.91,
                                     "reason_codes": ["SOURCE_LOCATED"], "checks": {"source_located": True}}},
    )
    PostgresKnowledgeRepository(database.session_factory).replace_for_video("video-1", [unit])
    repository = PostgresVerificationRepository(database.session_factory)
    repository.append([unit], "trace-1")
    repository.append([unit], "trace-2")
    with database.session_factory() as session:
        rows = session.scalars(select(KnowledgeVerificationRow)).all()
    assert len(rows) == 2
    assert {row.raw_output["trace_id"] for row in rows} == {"trace-1", "trace-2"}
    assert all(row.raw_output["knowledge_version"] == 1 for row in rows)

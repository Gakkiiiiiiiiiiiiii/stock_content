from datetime import UTC, datetime

from sqlalchemy import select

from stock_content.adapters.postgres import Database
from stock_content.adapters.postgres.models import FinancialEntityRow, FinancialEventRow, FinancialNumericFactRow
from stock_content.adapters.postgres.repositories.entity_repository import PostgresFinancialEntityRepository
from stock_content.adapters.postgres.repositories.financial_repository import PostgresFinancialRepository
from stock_content.application.pipeline import PipelineContext
from stock_content.application.stages import FinancialEnrichmentStage
from stock_content.domain.models import KnowledgeUnit


def test_financial_facts_keep_visibility_boundary_and_persist_entity_authority(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'content.db'}")
    database.create_schema()
    available_from = datetime(2026, 8, 15, 9, 30, tzinfo=UTC)
    unit = KnowledgeUnit(
        knowledge_uid="knowledge-1",
        video_id="video-1",
        chapter_id=None,
        statement="宁德时代300750业绩增长10%，营收达到100亿元。",
        subject_key="300750",
        ticker="300750",
        as_of=available_from,
        available_from=available_from,
    )
    context = PipelineContext(task_id="financial-test", source={}, options={}, data={"knowledge": [unit]})
    FinancialEnrichmentStage().execute(context)
    PostgresFinancialEntityRepository(database.session_factory).replace("video-1", [unit])
    PostgresFinancialRepository(database.session_factory).replace(
        "video-1", context.data["financial_numeric_facts"], context.data["financial_events"]
    )
    with database.session_factory() as session:
        numerics = session.scalars(select(FinancialNumericFactRow)).all()
        events = session.scalars(select(FinancialEventRow)).all()
        entities = session.scalars(select(FinancialEntityRow)).all()
    assert numerics and all(item.as_of_time and item.available_from for item in numerics)
    assert all(item.available_from.replace(tzinfo=UTC) == available_from for item in numerics)
    assert events and all(item.numeric_refs is not None and item.evidence_refs is not None for item in events)
    assert any(item.canonical_key == "CN.A.300750" for item in entities)

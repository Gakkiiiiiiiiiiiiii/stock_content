"""Knowledge 生命周期 TTL 测试（详细修改方案 §5 P1-5）。"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stock_content.domain.knowledge_enums import LifecycleStatus
from stock_content.domain.knowledge_lifecycle import KNOWLEDGE_TTL, evaluate_lifecycle, ttl_for

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


def test_lifecycle_statuses_include_required_states():
    values = LifecycleStatus.values()
    for status in ("ACTIVE", "STALE", "SUPERSEDED", "RETRACTED", "INVALID"):
        assert status in values


def test_price_knowledge_goes_stale_after_ttl():
    evaluation = evaluate_lifecycle(
        claim_type="PRICE",
        current_status="ACTIVE",
        as_of=NOW,
        available_from=NOW - timedelta(hours=2),
    )
    assert evaluation.action == "MARK_STALE"


def test_financial_report_fact_is_long_lived():
    assert ttl_for("FINANCIAL_METRIC") == timedelta(days=3650)
    evaluation = evaluate_lifecycle(
        claim_type="FINANCIAL_METRIC",
        current_status="ACTIVE",
        as_of=NOW,
        available_from=NOW - timedelta(days=30),
    )
    assert evaluation.action == "KEEP"


def test_industry_relation_is_event_driven():
    assert KNOWLEDGE_TTL["INDUSTRY_RELATION"] is None
    evaluation = evaluate_lifecycle(
        claim_type="INDUSTRY_RELATION", current_status="ACTIVE", as_of=NOW, available_from=NOW - timedelta(days=999)
    )
    assert evaluation.action == "KEEP"
    assert evaluation.reason == "EVENT_DRIVEN"


def test_forecast_enters_outcome_review_at_target_date():
    evaluation = evaluate_lifecycle(
        claim_type="FORECAST",
        current_status="ACTIVE",
        as_of=NOW,
        fact_time=(NOW - timedelta(days=1)).date(),
    )
    assert evaluation.action == "OUTCOME_REVIEW"

    pending = evaluate_lifecycle(
        claim_type="FORECAST",
        current_status="ACTIVE",
        as_of=NOW,
        fact_time=(NOW + timedelta(days=30)).date(),
    )
    assert pending.action == "KEEP"


def test_non_active_knowledge_is_not_touched():
    evaluation = evaluate_lifecycle(
        claim_type="PRICE", current_status="SUPERSEDED", as_of=NOW, available_from=NOW - timedelta(days=30)
    )
    assert evaluation.action == "KEEP"
    assert evaluation.reason == "NOT_ACTIVE"

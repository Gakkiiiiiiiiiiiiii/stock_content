from stock_content.domain.financial_event_extractor import FinancialEventExtractor


def test_financial_event_extraction_keeps_source_and_availability():
    events = FinancialEventExtractor().extract(
        [
            {
                "knowledge_uid": "k1",
                "statement": "公司上调全年业绩指引",
                "subject_key": "CN.A.600519",
                "available_from": "2026-08-15T09:20:00+08:00",
                "sentiment": "BULLISH",
                "confidence": 0.8,
            }
        ]
    )
    assert events[0]["event_type"] == "GUIDANCE"
    assert events[0]["knowledge_uid"] == "k1"

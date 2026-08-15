from datetime import UTC, datetime

from stock_content.domain.cross_video_corroboration import CrossVideoCorroborationService
from stock_content.domain.knowledge_enums import SupportStatus, support_rank


def test_canonical_support_rank_keeps_cross_modal_signal_eligible():
    assert support_rank(SupportStatus.CROSS_MODAL_SUPPORTED.value) > support_rank(SupportStatus.SOURCE_SUPPORTED.value)


def test_cross_video_excludes_source_located_narrative_from_reliable_consensus():
    rows = [
        {
            "knowledge_uid": "located",
            "video_id": "v1",
            "subject_key": "CN.A.600519",
            "predicate_key": "earnings",
            "sentiment": "BULLISH",
            "support_status": SupportStatus.SOURCE_LOCATED.value,
            "lifecycle_status": "ACTIVE",
            "evidence_ids": ["1"],
            "available_from": datetime.now(UTC).isoformat(),
        },
        {
            "knowledge_uid": "supported",
            "video_id": "v2",
            "subject_key": "CN.A.600519",
            "predicate_key": "earnings",
            "sentiment": "BULLISH",
            "support_status": SupportStatus.SOURCE_SUPPORTED.value,
            "lifecycle_status": "ACTIVE",
            "evidence_ids": ["2"],
        },
    ]
    result = CrossVideoCorroborationService().calculate(rows)
    assert result["supported"]["independent_source_count"] == 1
    assert "located" not in result

"""Opt-in account-backed smoke tests; never run from the deterministic CI suite."""

from __future__ import annotations

import os

import pytest

from stock_content.adapters.http.model_client import ContentModelClient
from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem
from stock_content.domain.atomic_claim_extractor import AtomicClaimExtractor
from stock_content.domain.semantic_segment import SemanticBoundary
from stock_content.domain.semantic_segmenter import SemanticSegmenter

pytestmark = pytest.mark.skipif(
    os.getenv("CONTENT_RUN_LIVE_CODEX_TEST") != "1", reason="requires logged-in Codex GPT-6 Sol"
)


def _transcript(rows: list[str]) -> TranscriptArtifact:
    return TranscriptArtifact(
        artifact_id="live-topic-transcript",
        artifact_type="transcript",
        media_artifact_id="live-topic-media",
        asr_model="self-test",
        asr_model_version="1",
        segments=[
            TranscriptSegmentItem(
                segment_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 10),
                text=text,
                raw_text=text,
                media_artifact_id="live-topic-media",
                asr_model="self-test",
                asr_model_version="1",
            )
            for index, text in enumerate(rows)
        ],
    )


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [
                "先讨论宏观通胀：居民消费价格连续回落，需求仍偏弱。",
                "货币政策关注实际利率和就业，不能把一次降准等同于复苏。",
                "财政支出形成的需求传导也有时滞，这仍是宏观政策判断。",
                "宏观结论是观察物价和就业的共同变化，不急于给个股目标价。",
                "下面转到人工智能算力，重点是训练集群所需的高带宽存储器。",
                "HBM 的产能爬坡与封装良率会制约 GPU 的实际交付。",
                "算力需求增长不等于每家芯片供应商利润都增长，要看订单。",
                "这一段的风险是新一代芯片量产节奏和客户资本开支变化。",
                "最后转到光伏组件行业。当前讨论的是库存和组件价格。",
                "渠道库存高企会压低组件厂开工率，与 HBM 供给不是一回事。",
                "观察库存去化、终端装机和企业现金流，才能判断光伏拐点。",
                "光伏结论是价格企稳仍需验证，不能仅凭政策口号下判断。",
            ],
            {3, 7},
        ),
        (
            [
                "贵州茅台 600519 的核心问题是渠道动销和批价是否稳定。",
                "若批价回落，收入增长也可能伴随库存上升。",
                "观察经销商库存与现金回款，才能检验茅台的增长质量。",
                "这一观点的失效条件是动销持续走弱和现金回款恶化。",
                "现在换到平安银行 000001，讨论净息差与不良贷款。",
                "银行的资产收益率下行会压低净息差，和白酒批价无关。",
                "平安银行还需要检查拨备覆盖与资本充足情况。",
                "银行部分的风险是信用成本上行，不能套用白酒的库存指标。",
            ],
            {3},
        ),
        (
            [
                "算力行业需要观察 GPU 供给、网络互连与机柜交付。",
                "高带宽存储器的良率会限制服务器实际交付。",
                "主持人还没来，稍等一下。",
                "所以这些算力环节需要一起核对订单和交付。",
                "接下来讨论黄金，问题是实际利率与金价的关系。",
                "实际利率下降可能支持金价，但美元变化也重要。",
                "黄金的风险在于利率反弹，不能套用算力订单指标。",
            ],
            {3},
        ),
        (
            [
                "先解释 AI 制药的干湿实验室闭环：算法提出候选分子，实体实验负责验证。",
                "干实验用于模拟和筛选，湿实验测量真实反应，两者往复改进设计。",
                "闭环的关键是实验结果能否反馈给模型，而非只生成分子名称。",
                "这套工作流还要核对验证成本和实验自动化能力。",
                "接着讨论临床数据瓶颈，问题已从实验工作流转向训练与验证材料。",
                "高质量临床数据稀缺，尤其是带结局和随访的完整数据。",
                "缺少可用数据会限制模型判断疗效，不能用实验自动化替代。",
                "数据授权、质量控制和跨机构标准化是这部分的独立约束。",
                "最后讨论药物上游耗材的商业弹性，与前面的数据瓶颈不同。",
                "耗材订单取决于实验通量和客户采购，不直接等于新药获批。",
                "观察订单复购、毛利率和产能利用率，才能判断耗材景气度。",
                "耗材公司的风险是客户研发预算收缩和竞争导致价格下行。",
            ],
            {3, 7},
        ),
    ],
)
def test_gpt6_sol_finds_subject_changes_without_fixed_time_windows(rows, expected):
    model = ContentModelClient()
    assert model.available()
    result = SemanticSegmenter(
        model, model_id="gpt-6-sol", allow_offline_fixture=False,
        require_model_identity=True, require_initial_topic=True, refine_segments=True,
        verify_brief_topic_labels=True,
    ).segment(_transcript(rows))
    assert {item.end_segment_index for item in result.segments[:-1]} == expected
    assert result.segments[0].start_segment_index == 0
    assert result.segments[0].topic
    assert result.segments[-1].end_segment_index == len(rows) - 1


def test_gpt6_sol_keeps_spoken_stock_name_and_code_in_objective_claims():
    transcript_text = "贵州茅台600519的渠道动销改善，但如果批价连续回落，估值修复就难以持续。"
    drafts = AtomicClaimExtractor(
        ContentModelClient(), model_id="gpt-6-sol",
        allow_offline_fixture=False, require_model_identity=True,
    ).extract({
        "semantic_segment_id": "stock-opinion-smoke",
        "transcript_text": transcript_text,
        "transcript_segments": [{"segment_index": 0, "raw_text": transcript_text, "text": transcript_text}],
    })
    assert drafts
    assert all(draft.subject_name == "贵州茅台" and draft.subject_key == "600519" for draft in drafts)
    assert all(draft.evidence_segment_indices == [0] for draft in drafts)
    assert all("讲者" not in draft.conclusion and "老师" not in draft.conclusion for draft in drafts)


def test_gpt6_sol_preserves_new_negotiation_thesis_after_related_security_story():
    rows = [
        "先分析北极航线附近的军事部署，安全准入是这项协议的主要目的。",
        "港口位置关系到航线和地缘安全，不能简单等同于开采矿产。",
        "这一安全协议的结论是航线控制权比资源开采更重要。",
        "另一个问题是两国即将谈判关键矿产的供应安排。",
        "若谈判取得较好结果，市场风险偏好可能改善。",
        "矿产供应的变化还会影响人工智能产业链，这形成独立市场判断。",
    ]
    segmenter = SemanticSegmenter(
        ContentModelClient(), model_id="gpt-6-sol", require_model_identity=True,
    )
    accepted, repairs = segmenter._adjudicate_internal_boundaries(
        _transcript(rows).segments, 0, len(rows),
        [SemanticBoundary(2, next_topic="关键矿产谈判与市场影响", next_subject="两国谈判")],
        parent_topic="北极航线安全协议", parent_subject="北极航线",
    )
    assert [item.after_segment_index for item in accepted] == [2]
    assert repairs == 0

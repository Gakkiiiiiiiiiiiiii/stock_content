"""视频知识模块统一枚举（单一事实来源）。

API / MCP / DB / Retriever / Admin / Schema 全部从这里 import，
禁止在其他文件各写一份 set（P0-11 / P1-7）。

状态模型（§79 三轴状态）：

- lifecycle_status: 生命周期
- support_status:   Axis 1 证据支持（来源内支持）
- truth_status:     Axis 2 外部真实性
- review_status:    Axis 3 人工审核

旧字段 verification_status 已 deprecated，仅做兼容映射。
"""

from __future__ import annotations

from enum import Enum


class KnowledgeKind(str, Enum):
    METHOD = "METHOD"
    CONCEPT = "CONCEPT"
    CAUSAL_THESIS = "CAUSAL_THESIS"
    FACT = "FACT"
    STATE = "STATE"
    FORECAST = "FORECAST"
    TECHNICAL_SIGNAL = "TECHNICAL_SIGNAL"
    ACTION = "ACTION"
    RISK_CONDITION = "RISK_CONDITION"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    VALUATION = "VALUATION"
    FINANCIAL_METRIC = "FINANCIAL_METRIC"
    PRICE_LEVEL = "PRICE_LEVEL"
    POLICY_FACT = "POLICY_FACT"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class SupportStatus(str, Enum):
    """Axis 1：来源内证据支持等级。"""

    UNSUPPORTED = "UNSUPPORTED"
    SOURCE_LOCATED = "SOURCE_LOCATED"
    SOURCE_SUPPORTED = "SOURCE_SUPPORTED"
    CROSS_MODAL_SUPPORTED = "CROSS_MODAL_SUPPORTED"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


# 新枚举的等级顺序（下标即等级）。
SUPPORT_ORDER: tuple[str, ...] = (
    SupportStatus.UNSUPPORTED.value,
    SupportStatus.SOURCE_LOCATED.value,
    SupportStatus.SOURCE_SUPPORTED.value,
    SupportStatus.CROSS_MODAL_SUPPORTED.value,
)

# 仅兼容读取旧数据用：遗留值 -> 最接近的新枚举值。
LEGACY_SUPPORT_ALIASES: dict[str, str] = {
    "NEEDS_REVIEW": SupportStatus.SOURCE_LOCATED.value,
    "EXTERNALLY_VERIFIED": SupportStatus.CROSS_MODAL_SUPPORTED.value,
    "VALIDATED": SupportStatus.CROSS_MODAL_SUPPORTED.value,
}

# 遗留值也参与等级比较：
# NEEDS_REVIEW 排在 SOURCE_LOCATED 与 SOURCE_SUPPORTED 之间，
# EXTERNALLY_VERIFIED / VALIDATED 排在 CROSS_MODAL_SUPPORTED 之上。
_SUPPORT_RANK: dict[str, int] = {
    SupportStatus.UNSUPPORTED.value: 0,
    SupportStatus.SOURCE_LOCATED.value: 1,
    "NEEDS_REVIEW": 2,
    SupportStatus.SOURCE_SUPPORTED.value: 3,
    SupportStatus.CROSS_MODAL_SUPPORTED.value: 4,
    "EXTERNALLY_VERIFIED": 5,
    "VALIDATED": 5,
}


def support_rank(status: str | None) -> int:
    """返回 support_status 的等级整数，未知/None 视为最低（0）。

    兼容旧数据中的 NEEDS_REVIEW / EXTERNALLY_VERIFIED / VALIDATED，
    保证检索层旧逻辑可平滑迁移。
    """

    if not status:
        return 0
    return _SUPPORT_RANK.get(str(status).strip().upper(), 0)


class TruthStatus(str, Enum):
    """Axis 2：外部真实性。"""

    NOT_CHECKED = "NOT_CHECKED"
    NOT_FOUND = "NOT_FOUND"
    EXTERNALLY_VERIFIED = "EXTERNALLY_VERIFIED"
    EXTERNAL_CONFLICT = "EXTERNAL_CONFLICT"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


LEGACY_TRUTH_ALIASES: dict[str, str] = {
    "NOT_EXTERNALLY_VERIFIED": TruthStatus.NOT_CHECKED.value,
}


class ReviewStatus(str, Enum):
    """Axis 3：人工审核。"""

    UNREVIEWED = "UNREVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    OVERRIDDEN = "OVERRIDDEN"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class LifecycleStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    ACTIVE = "ACTIVE"
    VALIDATED = "VALIDATED"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


TERMINAL_LIFECYCLE_STATUSES: set[str] = {
    LifecycleStatus.REJECTED.value,
    LifecycleStatus.RETIRED.value,
}


class TemporalClass(str, Enum):
    DURABLE = "DURABLE"
    CYCLICAL = "CYCLICAL"
    SNAPSHOT = "SNAPSHOT"
    EVENT_BOUND = "EVENT_BOUND"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class VerificationStatus(str, Enum):
    """旧字段，deprecated，仅做兼容映射，不再作为新逻辑核心字段。"""

    UNVERIFIED = "UNVERIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    SOURCE_LOCATED = "SOURCE_LOCATED"
    SOURCE_SUPPORTED = "SOURCE_SUPPORTED"
    CROSS_MODAL_SUPPORTED = "CROSS_MODAL_SUPPORTED"
    EXTERNALLY_VERIFIED = "EXTERNALLY_VERIFIED"
    SOURCE_CONFIRMED = "SOURCE_CONFIRMED"
    VERIFIED = "VERIFIED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class EvidenceQualityStatus(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class TransformationType(str, Enum):
    FORMAT_NORMALIZATION = "FORMAT_NORMALIZATION"
    SCRIPT_CONVERSION = "SCRIPT_CONVERSION"
    DICTIONARY_CORRECTION = "DICTIONARY_CORRECTION"
    ENTITY_RESOLUTION = "ENTITY_RESOLUTION"
    MODEL_CORRECTION = "MODEL_CORRECTION"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class SpeakerMode(str, Enum):
    SINGLE_SPEAKER = "SINGLE_SPEAKER"
    DIARIZE = "DIARIZE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class VerifierType(str, Enum):
    SOURCE_LOCATION = "SOURCE_LOCATION"
    ASR_QUALITY = "ASR_QUALITY"
    ENTITY_RESOLUTION = "ENTITY_RESOLUTION"
    NUMERIC_BINDING = "NUMERIC_BINDING"
    SEMANTIC_ENTAILMENT = "SEMANTIC_ENTAILMENT"
    CROSS_MODAL = "CROSS_MODAL"
    EXTERNAL_FACT = "EXTERNAL_FACT"
    MANUAL_REVIEW = "MANUAL_REVIEW"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


HIGH_RISK_KINDS: set[str] = {
    KnowledgeKind.FACT.value,
    KnowledgeKind.VALUATION.value,
    KnowledgeKind.FINANCIAL_METRIC.value,
    KnowledgeKind.PRICE_LEVEL.value,
    KnowledgeKind.POLICY_FACT.value,
}


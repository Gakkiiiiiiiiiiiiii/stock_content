"""ClaimEvidenceVerifier V2（设计文档 §10-17，P0-4）。

两阶段结构：

- Stage A：确定性硬门禁（structured slot checks）。从 claim 与 source
  分别抽取结构化槽位（数字 / 主体 / 方向 / 否定 / 条件 / 时间），逐项判定，
  修复 V1 的四类缺陷：
  - §11 跨主体误绑定：数字匹配限定在 claim 主体所在子句内；
  - §12 否定不对称：claim 与 source 的谓语否定状态必须一致；
  - §13 条件丢失：source 有条件而 claim 没有 → CONDITION_DROPPED；
  - §15 score>1：score = passed_hard/N * 0.75 + overlap * 0.25，clamp [0,1]。
- Stage B：可选语义裁判（judge 注入）。judge 不可覆盖 Stage A 硬失败；
  judge 判 CONTRADICTED / NOT_ENOUGH_EVIDENCE 时可降级。

score 语义（§17）：support_score 是未校准分数（uncalibrated），
``support_probability`` 仅为兼容旧链路（normalizer / repository / retrieval）
保留的同名键，值与 support_score 相同，不代表校准后的概率。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from stock_content.domain.financial_numeric import (
    FinancialNumericValue,
    numeric_values_match,
    parse_financial_numerics,
)
from stock_content.domain.knowledge_enums import HIGH_RISK_KINDS
from stock_content.domain.semantic_entailment_judge import INFRA_FAILURE_REASONS

# Stage B 语义裁判签名：输入 claim/evidence/structured_checks，
# 输出 {"label": SUPPORTED|CONTRADICTED|NOT_ENOUGH_EVIDENCE, "score": float, "reason_codes": list}，
# 可附带 provider/model/version 元数据（用于 Verification Ledger 入账）。
JudgeFn = Callable[[dict[str, Any]], dict[str, Any]]

HARD_CHECKS: tuple[str, ...] = (
    "number_match",
    "entity_match",
    "direction_match",
    "negation_match",
    "condition_match",
    "unit_match",
    "time_match",
)

_CLAUSE_SPLIT_RE = re.compile(r"[,，。；;!！?？、\n]")
_CONDITION_RE = re.compile(r"如果|倘若|若是|若非|除非|一旦|只要|只有|跌破|突破|当.{0,10}时")


class ClaimEvidenceVerifier:
    """Conservative claim-to-evidence support checker (V2 two-stage).

    Stage A hard guards are deterministic and auditable; an optional semantic
    judge (Stage B) can downgrade but never override a Stage A hard failure.
    """

    POSITIVE = ("上涨", "增长", "改善", "看多", "看好", "偏强", "突破", "加仓")
    NEGATIVE = ("下跌", "下降", "恶化", "看空", "偏弱", "跌破", "减仓")
    # 谓语词：用于否定邻近检查（§12）。
    PREDICATES = POSITIVE + NEGATIVE + ("便宜", "高估", "低估", "景气", "回暖", "走弱")
    NEGATIONS = ("没有", "没", "未", "并非", "并不", "不是", "无", "不")
    # 含 "不" 但为肯定语境的例外。
    NEGATION_EXCEPTIONS = ("不断", "不得不", "不得已")
    TIME_GROUPS: tuple[tuple[str, ...], ...] = (
        ("当前", "目前", "现在", "当下", "现阶段"),
        ("今年", "去年", "明年", "前年"),
        ("短期", "中期", "长期"),
        ("近期", "未来", "远期"),
        ("上半年", "下半年"),
        ("一季度", "二季度", "三季度", "四季度"),
        ("本周", "上周", "下周"),
        ("本月", "上月", "下月"),
        ("今天", "昨天", "明天"),
    )

    def __init__(
        self,
        judge: JudgeFn | None = None,
        high_risk_threshold: float = 0.82,
        default_threshold: float = 0.65,
        overlap_min: float = 0.18,
    ) -> None:
        self.judge = judge
        self.high_risk_threshold = high_risk_threshold
        self.default_threshold = default_threshold
        self.overlap_min = overlap_min

    def verify(self, unit: dict[str, Any]) -> dict[str, Any]:
        evidence = list(unit.get("evidence") or [])
        primary = next((item for item in evidence if item.get("is_primary")), evidence[0] if evidence else None)
        if not primary or primary.get("start_ms") is None or primary.get("end_ms") is None:
            return self._result("UNSUPPORTED", 0.0, ["EVIDENCE_NOT_LOCATED"], {})
        source = " ".join(str(primary.get(key) or "") for key in ("raw_text", "normalized_text", "evidence_text"))
        claim = str(unit.get("statement") or "")
        if not source.strip() or not claim:
            return self._result("UNSUPPORTED", 0.0, ["EMPTY_CLAIM_OR_EVIDENCE"], {})

        source_norm, claim_norm = self._compact(source), self._compact(claim)
        names = self._subject_names(unit)
        # §11：数字匹配限定在 claim 主体所在子句，杜绝跨实体误绑定。
        scope = self._scope_text(source_norm, names)
        claim_numbers = parse_financial_numerics(claim_norm)
        scope_numbers = parse_financial_numerics(scope)

        condition_ok, condition_dropped = self._condition_check(unit, claim_norm, source_norm)
        checks: dict[str, Any] = {
            "number_match": self._number_match(claim_numbers, scope_numbers),
            "entity_match": self._entity_match(names, source_norm),
            "direction_match": self._direction_match(claim_norm, source_norm),
            "negation_match": self._negation_match(claim_norm, source_norm),
            "condition_match": condition_ok,
            "unit_match": self._unit_match(claim_numbers, scope_numbers),
            "time_match": self._time_match(claim_norm, source_norm),
            "source_located": True,
        }
        if condition_dropped:
            checks["condition_dropped"] = True

        overlap = self._keyword_overlap(claim_norm, source_norm)
        # bigram 重叠只作为软信号进入 score，不作为硬门禁（§14）。
        checks["semantic_overlap"] = overlap >= self.overlap_min

        hard_failed = [key for key in HARD_CHECKS if not checks[key]]
        reasons = [f"{key.upper()}_FAILED" for key in hard_failed]
        if condition_dropped:
            reasons.append("CONDITION_DROPPED")
        if not checks["semantic_overlap"]:
            reasons.append("SEMANTIC_OVERLAP_LOW")

        score = (sum(1 for key in HARD_CHECKS if checks[key]) / len(HARD_CHECKS)) * 0.75 + overlap * 0.25
        score = max(0.0, min(1.0, score))  # §15：永不超过 1

        high_risk = str(unit.get("knowledge_kind") or "").upper() in HIGH_RISK_KINDS
        threshold = self.high_risk_threshold if high_risk else self.default_threshold
        status = "SOURCE_SUPPORTED" if not hard_failed and score >= threshold else "NEEDS_REVIEW"

        judge_meta: dict[str, Any] | None = None
        if self.judge is not None:
            status, reasons, judge_meta = self._apply_judge(
                claim=claim,
                source=source,
                checks=checks,
                status=status,
                reasons=reasons,
                hard_failed=bool(hard_failed),
                threshold=threshold,
            )

        result = self._result(status, score, reasons, checks)
        if judge_meta is not None:
            # judge 元数据随 verification 流出，供 Verification Ledger 入账（§4 验收）。
            result["judge"] = judge_meta
        return result

    # ------------------------------------------------------------------
    # Stage B
    # ------------------------------------------------------------------

    def _apply_judge(
        self,
        claim: str,
        source: str,
        checks: dict[str, Any],
        status: str,
        reasons: list[str],
        hard_failed: bool,
        threshold: float,
    ) -> tuple[str, list[str], dict[str, Any]]:
        verdict = self.judge({"claim": claim, "evidence": source, "structured_checks": dict(checks)}) or {}
        label = str(verdict.get("label") or "").upper()
        reason_codes = list(verdict.get("reason_codes") or [])
        checks["judge"] = {
            "label": label or None,
            "score": verdict.get("score"),
            "reason_codes": reason_codes,
        }
        judge_meta = {
            "provider": verdict.get("provider"),
            "model": verdict.get("model"),
            "version": verdict.get("version"),
            "label": label or None,
            "score": verdict.get("score"),
        }
        if label == "CONTRADICTED":
            reasons.append("JUDGE_CONTRADICTED")
            return ("NEEDS_REVIEW" if status == "SOURCE_SUPPORTED" else status), reasons, judge_meta
        if label == "NOT_ENOUGH_EVIDENCE":
            if INFRA_FAILURE_REASONS & set(reason_codes):
                # 裁判自身失效（不可用/调用异常/非法输出）不等于证据不足：弃权，不降级。
                return status, reasons, judge_meta
            if status == "SOURCE_SUPPORTED":
                reasons.append("JUDGE_NOT_ENOUGH_EVIDENCE")
                return "NEEDS_REVIEW", reasons, judge_meta
            return status, reasons, judge_meta
        if label == "SUPPORTED" and not hard_failed and status != "SOURCE_SUPPORTED":
            # Stage A 无硬失败时才允许 judge 升级；硬失败不可被覆盖（§16.2）。
            try:
                judge_score = float(verdict.get("score") or 0.0)
            except (TypeError, ValueError):
                judge_score = 0.0
            if judge_score >= threshold:
                return "SOURCE_SUPPORTED", reasons, judge_meta
        return status, reasons, judge_meta

    # ------------------------------------------------------------------
    # Stage A slot checks
    # ------------------------------------------------------------------

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"\s+", "", value).lower()

    @classmethod
    def _subject_names(cls, unit: dict[str, Any]) -> list[str]:
        names = [str(unit.get(key) or "").strip() for key in ("subject_name", "subject_key")]
        names.extend(
            str(item.get("entity_name") or item.get("ticker") or "").strip() for item in unit.get("entities") or []
        )
        seen: list[str] = []
        for name in names:
            compacted = cls._compact(name)
            if len(compacted) >= 2 and compacted not in seen:
                seen.append(compacted)
        return seen

    @staticmethod
    def _scope_text(source: str, names: list[str]) -> str:
        """取主体名出现的子句（按 ，。； 切分）作为数字匹配的作用域。"""

        if not names:
            return source
        clauses = [clause for clause in _CLAUSE_SPLIT_RE.split(source) if clause]
        scoped = [clause for clause in clauses if any(name in clause for name in names)]
        return "。".join(scoped) if scoped else source

    @classmethod
    def _number_match(
        cls, claim_numbers: list[FinancialNumericValue], scope_numbers: list[FinancialNumericValue]
    ) -> bool:
        for claim_num in claim_numbers:
            if not any(
                numeric_values_match(claim_num, source_num) and cls._metric_compatible(claim_num, source_num)
                for source_num in scope_numbers
            ):
                return False
        return True

    @staticmethod
    def _metric_compatible(claim_num: FinancialNumericValue, source_num: FinancialNumericValue) -> bool:
        # 双方都能推断出 metric 且不同（利润 vs 营收）→ 跨指标误绑定，不匹配。
        return not claim_num.metric or not source_num.metric or claim_num.metric == source_num.metric

    @staticmethod
    def _unit_match(claim_numbers: list[FinancialNumericValue], scope_numbers: list[FinancialNumericValue]) -> bool:
        for claim_num in claim_numbers:
            if claim_num.unit and not any(source_num.unit == claim_num.unit for source_num in scope_numbers):
                return False
        return True

    @staticmethod
    def _entity_match(names: list[str], source: str) -> bool:
        # 无可用名称时保持 V1 语义（True）；normalizer 会传规范化后的实体。
        return not names or any(name in source for name in names)

    @classmethod
    def _direction_match(cls, claim: str, source: str) -> bool:
        def polarity(text: str) -> int:
            return int(any(token in text for token in cls.POSITIVE)) - int(any(token in text for token in cls.NEGATIVE))

        return polarity(claim) == 0 or polarity(claim) == polarity(source)

    @classmethod
    def _has_negation(cls, text: str) -> bool:
        cleaned = text
        for exception in cls.NEGATION_EXCEPTIONS:
            cleaned = cleaned.replace(exception, "")
        return any(token in cleaned for token in cls.NEGATIONS)

    @classmethod
    def _negation_match(cls, claim: str, source: str) -> bool:
        """§12 对称化：claim 与 source 对同一谓语的否定状态必须一致。"""

        claim_negated = cls._has_negation(claim)
        for predicate in cls.PREDICATES:
            if predicate not in claim:
                continue
            for match in re.finditer(re.escape(predicate), source):
                near = source[max(0, match.start() - 4) : match.start()]
                source_negated = cls._has_negation(near)
                if claim_negated != source_negated:
                    return False
        # claim 有否定但 source 通篇无否定 → 不一致。
        if claim_negated and not cls._has_negation(source):
            return False
        return True

    @classmethod
    def _has_condition(cls, text: str) -> bool:
        if _CONDITION_RE.search(text):
            return True
        return "若" in text and "若干" not in text

    @classmethod
    def _condition_check(cls, unit: dict[str, Any], claim: str, source: str) -> tuple[bool, bool]:
        """返回 (condition_match, condition_dropped)（§13）。"""

        condition = cls._compact(str(unit.get("condition_text") or ""))
        if condition:
            return condition in source, False
        # source 含条件而 claim 既无 condition_text 也无条件词 → 条件被 LLM 丢失。
        if cls._has_condition(source) and not cls._has_condition(claim):
            return False, True
        return True, False

    @classmethod
    def _time_match(cls, claim: str, source: str) -> bool:
        """claim 的时间词在 source 中需一致；source 无同组时间词时宽松通过。"""

        for group in cls.TIME_GROUPS:
            claim_words = [word for word in group if word in claim]
            if not claim_words:
                continue
            source_words = [word for word in group if word in source]
            if not source_words:
                continue  # 宽松：source 无该组时间词不算冲突
            if not any(word in source_words for word in claim_words):
                return False  # 同组不同词（今年 vs 去年）→ 时间口径冲突
        return True

    @staticmethod
    def _keyword_overlap(claim: str, source: str) -> float:
        # Chinese character bigrams give an explainable approximation for a
        # semantic candidate check without treating LLM confidence as evidence.
        tokens = {claim[index : index + 2] for index in range(len(claim) - 1) if claim[index : index + 2].strip()}
        if not tokens:
            return 1.0
        return len({token for token in tokens if token in source}) / len(tokens)

    @staticmethod
    def _result(status: str, score: float, reasons: list[str], checks: dict[str, Any]) -> dict[str, Any]:
        score = max(0.0, min(1.0, round(score, 4)))
        return {
            "support_status": status,
            "support_score": score,
            # 兼容键：与 support_score 同值；语义是 uncalibrated support score（§17），
            # 在完成 Golden Dataset Calibration 前不得解释为概率。
            "support_probability": score,
            "reason_codes": reasons,
            "checks": checks,
        }

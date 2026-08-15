"""Semantic Entailment Judge（最终收敛设计文档 §4，剩余项一）。

ClaimEvidenceVerifier Stage B 的真实语义裁判，复用现有
``AnalysisModelClient``（JSON mode，temperature=0），不引入新模型体系。

安全降级语义：
- 模型不可用 → NOT_ENOUGH_EVIDENCE + JUDGE_UNAVAILABLE；
- 调用异常 / JSON 解析失败 → NOT_ENOUGH_EVIDENCE + JUDGE_ERROR；
- label 不在三值白名单 → NOT_ENOUGH_EVIDENCE + JUDGE_LABEL_INVALID；
- score 越界 → clamp 到 [0, 1]。

绝不伪报 SUPPORTED。基础设施类失败（JUDGE_UNAVAILABLE / JUDGE_ERROR /
JUDGE_LABEL_INVALID）只代表裁判自身失效，不代表证据不足；verifier 对这类
reason 按弃权处理（不降级），只对真实语义判断的 NOT_ENOUGH_EVIDENCE /
CONTRADICTED 降级。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

JUDGE_VERSION = "judge-v1"

VALID_LABELS = frozenset({"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"})

# 基础设施失败 reason：裁判自身失效，而非 claim-evidence 语义不足。
INFRA_FAILURE_REASONS = frozenset({"JUDGE_UNAVAILABLE", "JUDGE_ERROR", "JUDGE_LABEL_INVALID"})

SYSTEM_PROMPT = """你是金融 Claim-Evidence 语义一致性裁判。

你的任务不是判断事实在现实世界是否为真，
只判断 Evidence 是否在语义上支持 Claim。

必须严格区分：
1. SOURCE SUPPORT：视频证据是否表达了该结论；
2. EXTERNAL TRUTH：现实世界是否为真。

当前只判断第一项。

若 Evidence 与 Claim 语义相反，输出 CONTRADICTED。
若 Evidence 不足以推出 Claim，输出 NOT_ENOUGH_EVIDENCE。
仅当 Evidence 明确支持 Claim 时输出 SUPPORTED。

不得根据常识、金融知识或外部信息补充 Evidence 中不存在的信息。

只输出一个严格的 JSON 对象，不要输出任何其他文字：
{"label": "SUPPORTED|CONTRADICTED|NOT_ENOUGH_EVIDENCE",
 "score": 0.0到1.0之间的小数, "reason_codes": ["可选的原因码"]}"""


class SemanticEntailmentJudge:
    """兼容 ``ClaimEvidenceVerifier`` JudgeFn 的语义裁判（§4）。"""

    VERSION = JUDGE_VERSION

    def __init__(self, model_client) -> None:
        self.model_client = model_client

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider, model = self._client_identity()
        if self.model_client is None or not self._available():
            return self._verdict("NOT_ENOUGH_EVIDENCE", 0.0, ["JUDGE_UNAVAILABLE"], provider, model)
        try:
            response = self.model_client.complete(
                prompt=self._prompt(payload),
                system=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.warning("semantic entailment judge 调用失败", exc_info=True)
            return self._verdict("NOT_ENOUGH_EVIDENCE", 0.0, ["JUDGE_ERROR"], provider, model)
        provider = response.get("provider") or provider
        model = response.get("model") or model
        try:
            parsed = json.loads(str(response.get("content") or ""))
            if not isinstance(parsed, dict):
                raise ValueError("judge output is not a JSON object")
        except (json.JSONDecodeError, ValueError):
            logger.warning("semantic entailment judge 输出不是合法 JSON：%r", str(response.get("content") or "")[:200])
            return self._verdict("NOT_ENOUGH_EVIDENCE", 0.0, ["JUDGE_ERROR"], provider, model)

        label = str(parsed.get("label") or "").strip().upper()
        reason_codes = [str(code) for code in parsed.get("reason_codes") or []]
        if label not in VALID_LABELS:
            return self._verdict("NOT_ENOUGH_EVIDENCE", 0.0, [*reason_codes, "JUDGE_LABEL_INVALID"], provider, model)
        return self._verdict(label, self._clamp_score(parsed.get("score")), reason_codes, provider, model)

    # ------------------------------------------------------------------

    @staticmethod
    def _prompt(payload: dict[str, Any]) -> str:
        structured = json.dumps(payload.get("structured_checks") or {}, ensure_ascii=False, sort_keys=True)
        return (
            f"Claim：{payload.get('claim') or ''}\n\n"
            f"Evidence：{payload.get('evidence') or ''}\n\n"
            f"结构化硬校验结果（仅供参考，不得覆盖）：{structured}\n\n"
            "请判断 Evidence 是否在语义上支持 Claim，输出 JSON。"
        )

    def _available(self) -> bool:
        available = getattr(self.model_client, "available", None)
        try:
            return bool(available()) if callable(available) else True
        except Exception:
            return False

    def _client_identity(self) -> tuple[str | None, str | None]:
        settings = getattr(self.model_client, "settings", None)
        provider = getattr(settings, "provider", None) or getattr(self.model_client, "provider", None)
        model = getattr(settings, "model", None) or getattr(self.model_client, "model", None)
        return provider, model

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    def _verdict(
        self,
        label: str,
        score: float,
        reason_codes: list[str],
        provider: str | None,
        model: str | None,
    ) -> dict[str, Any]:
        return {
            "label": label,
            "score": score,
            "reason_codes": reason_codes,
            "provider": provider,
            "model": model,
            "version": self.VERSION,
        }

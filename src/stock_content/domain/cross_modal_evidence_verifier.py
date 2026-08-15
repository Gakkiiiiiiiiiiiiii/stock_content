"""Cross-Modal Evidence Verifier（P0-15 / §46-47）。

对每条 claim 做 ASR × OCR 双证据交叉印证，产出真实的
CROSS_MODAL_SUPPORTED / CROSS_MODAL_CONFLICT，结果写入
unit.attributes["cross_modal_verification"]（repository 按 CROSS_MODAL 入账）。

规则（§47 OCR Numeric Gate + §12 收敛修复）：
- OCR block score >= 0.95 -> HIGH；0.85~0.95 -> MEDIUM；< 0.85 不足以作 strong support。
- 升级 CROSS_MODAL_SUPPORTED 最低条件（§12.3）：ASR 已 SOURCE_SUPPORTED
  且 OCR HIGH/MEDIUM 且 subject 命中且 claim 有结构化数字
  且数字值/单位匹配且 metric 不冲突；claim 含方向词时 OCR 须含一致方向。
  无数字 claim 保持 SOURCE_SUPPORTED，subject-only 命中不得升级。
- ASR 与 OCR 数字可比较（metric/unit 一致）且双高置信冲突 -> NEEDS_REVIEW + CROSS_MODAL_CONFLICT；
  不可比较（PE 20倍 vs 股价30元）既不一致也不冲突（§12.4）。
- 无 OCR 证据 -> 不动。
"""

from __future__ import annotations

import re
from typing import Any

from stock_content.domain.knowledge_enums import SupportStatus

OCR_SOURCE_TYPES = {"OCR", "VISION", "FRAME"}
HIGH_THRESHOLD = 0.95
MEDIUM_THRESHOLD = 0.85


class CrossModalEvidenceVerifier:
    def __init__(self, high_threshold: float = HIGH_THRESHOLD, medium_threshold: float = MEDIUM_THRESHOLD) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def verify_many(self, units: list[dict], ocr_evidence: list[dict] | None = None) -> list[dict]:
        results: list[dict] = []
        for unit in units:
            results.append(self._verify_one(unit, ocr_evidence or []))
        return results

    def _verify_one(self, unit: dict, extra_ocr: list[dict]) -> dict:
        item = dict(unit)
        evidence = list(item.get("evidence") or [])
        primary = next((e for e in evidence if e.get("is_primary")), evidence[0] if evidence else None)
        ocr_items = self._overlapping_ocr(primary, evidence) + self._overlapping_ocr(primary, extra_ocr)
        if not ocr_items:
            return item

        statement = str(item.get("statement") or "")
        claim_values = self._numbers(statement)
        claim_direction = self._direction(statement)
        subject_tokens = self._subject_tokens(item)

        qualified: list[dict] = []  # (ocr_item, level, text, numbers, subject_hit)
        for ocr in ocr_items:
            level = self._ocr_level(ocr)
            text = self._ocr_text(ocr)
            if not text:
                continue
            numbers = self._numbers(text)
            subject_hit = not subject_tokens or any(token in text for token in subject_tokens)
            qualified.append({"ocr": ocr, "level": level, "text": text, "numbers": numbers, "subject_hit": subject_hit})
        if not qualified:
            return item

        asr_support = str(item.get("support_status") or "")
        asr_score = item.get("support_score")
        if asr_score is None:
            asr_score = item.get("support_probability")

        # 数字冲突门：主体命中 + OCR HIGH + 双侧数字可比较（metric/unit 一致）但不一致
        # -> 双高置信冲突（§12.4：PE 20倍 vs 股价30元 既不一致也不冲突）。
        for entry in qualified:
            if entry["level"] != "HIGH" or not entry["subject_hit"]:
                continue
            if claim_values and entry["numbers"] and self._numbers_conflict(claim_values, entry["numbers"]):
                return item | {
                    "support_status": "NEEDS_REVIEW",
                    "verification_status": "NEEDS_REVIEW",
                    "attributes": (item.get("attributes") or {}) | {
                        "cross_modal_verification": {
                            "status": "CROSS_MODAL_CONFLICT",
                            "asr_support_score": asr_score,
                            "ocr_support_score": self._ocr_score(entry["ocr"]),
                            "matched_blocks": [self._block_ref(entry["ocr"])],
                            "reason_codes": ["CROSS_MODAL_CONFLICT"],
                            "claim_values": [self._number_ref(v) for v in claim_values],
                            "ocr_values": [self._number_ref(v) for v in entry["numbers"]],
                        }
                    },
                }

        # 升级门（§12.3）：ASR SOURCE_SUPPORTED + OCR HIGH/MEDIUM + 主体命中
        # + claim 有结构化数字 + 数字值/单位匹配 + metric 不冲突 + 方向一致。
        # 无数字 claim 保持 SOURCE_SUPPORTED，subject-only 命中不得升级（§11）。
        if asr_support != SupportStatus.SOURCE_SUPPORTED.value or not claim_values:
            return item
        for entry in qualified:
            if entry["level"] not in {"HIGH", "MEDIUM"} or not entry["subject_hit"]:
                continue
            if not entry["numbers"]:
                continue
            if not self._numbers_match(claim_values, entry["numbers"]):
                continue
            # claim 含方向词（增长/下降/上涨/下跌…）时 OCR 也须含一致方向，否则不升级。
            if claim_direction is not None and self._direction(entry["text"]) != claim_direction:
                continue
            return item | {
                "support_status": SupportStatus.CROSS_MODAL_SUPPORTED.value,
                "verification_status": SupportStatus.CROSS_MODAL_SUPPORTED.value,
                "attributes": (item.get("attributes") or {}) | {
                    "cross_modal_verification": {
                        "status": SupportStatus.CROSS_MODAL_SUPPORTED.value,
                        "asr_support_score": asr_score,
                        "ocr_support_score": self._ocr_score(entry["ocr"]),
                        "matched_blocks": [self._block_ref(entry["ocr"])],
                    }
                },
            }
        return item

    def _overlapping_ocr(self, primary: dict | None, candidates: list[dict]) -> list[dict]:
        items: list[dict] = []
        for candidate in candidates:
            source_type = str(candidate.get("source_type") or "OCR").upper()
            if source_type not in OCR_SOURCE_TYPES:
                continue
            if primary is None or not self._overlaps(primary, candidate):
                continue
            items.append(candidate)
        return items

    @staticmethod
    def _overlaps(left: dict, right: dict) -> bool:
        left_start, left_end = left.get("start_ms"), left.get("end_ms")
        right_start, right_end = right.get("start_ms"), right.get("end_ms")
        if None in (left_start, left_end, right_start, right_end):
            return True  # 无时间信息时不排除，交由后续门槛裁决
        return float(left_start) < float(right_end) and float(right_start) < float(left_end)

    def _ocr_level(self, ocr: dict) -> str:
        score = self._ocr_score(ocr)
        if score is None:
            return "LOW"
        if score >= self.high_threshold:
            return "HIGH"
        if score >= self.medium_threshold:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _ocr_score(ocr: dict) -> float | None:
        metrics = ocr.get("ocr_metrics") or {}
        score = metrics.get("mean_confidence")
        if score is None:
            score = ocr.get("score")
        if score is None:
            score = ocr.get("confidence_score")
        try:
            return None if score is None else float(score)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ocr_text(ocr: dict) -> str:
        return " ".join(
            str(ocr.get(key) or "") for key in ("raw_text", "normalized_text", "evidence_text", "text")
        ).strip()

    @staticmethod
    def _subject_tokens(unit: dict) -> list[str]:
        tokens = [str(unit.get(key) or "").strip() for key in ("subject_name", "subject_key")]
        tokens.extend(str(e.get("entity_name") or "").strip() for e in unit.get("entities") or [])
        return [token for token in tokens if len(token) >= 2]

    @staticmethod
    def _block_ref(ocr: dict) -> dict:
        return {
            "source_ref": ocr.get("source_ref"),
            "frame_id": ocr.get("frame_id"),
            "score": CrossModalEvidenceVerifier._ocr_score(ocr),
        }

    @staticmethod
    def _numbers(text: str) -> list:
        """§12.1：返回结构化 FinancialNumericValue（保留 value/unit/metric/comparator）。"""
        from stock_content.domain.financial_numeric import parse_financial_numerics

        return parse_financial_numerics(text)

    @staticmethod
    def _number_ref(value) -> dict:
        return {"value": value.value, "unit": value.unit, "metric": value.metric}

    @staticmethod
    def _comparable(claim, ocr) -> bool:
        """metric 双方都非空且不同、或 unit 双方都非空且不同 -> 不可比较（§12.4）。"""
        if claim.metric and ocr.metric and claim.metric != ocr.metric:
            return False
        if claim.unit and ocr.unit and claim.unit != ocr.unit:
            return False
        return True

    @staticmethod
    def _numbers_match(claim_values: list, ocr_values: list) -> bool:
        """§12.2：metric 不冲突 + numeric_values_match 逐项配对。"""
        from stock_content.domain.financial_numeric import numeric_values_match

        for claim in claim_values:
            matched = False
            for ocr in ocr_values:
                if (
                    claim.metric
                    and ocr.metric
                    and claim.metric != ocr.metric
                ):
                    continue
                if numeric_values_match(claim, ocr):
                    matched = True
                    break
            if not matched:
                return False
        return True

    @classmethod
    def _numbers_conflict(cls, claim_values: list, ocr_values: list) -> bool:
        """存在可比较（metric/unit 一致）的数字对且全部不匹配 -> 冲突；
        完全不可比较（PE 对股价）-> 既不一致也不冲突（§12.4）。"""
        if not any(cls._comparable(claim, ocr) for claim in claim_values for ocr in ocr_values):
            return False
        return not cls._numbers_match(claim_values, ocr_values)

    _UP_WORDS = ("增长", "上涨", "上升", "提升", "提高", "增加", "走高")
    _DOWN_WORDS = ("下降", "下跌", "下滑", "降低", "减少", "回落", "走低")

    @classmethod
    def _direction(cls, text: str) -> str | None:
        """方向词（增长/下降…）或带符号数字（+120% / -5%）推断 UP/DOWN。"""
        if any(word in text for word in cls._UP_WORDS):
            return "UP"
        if any(word in text for word in cls._DOWN_WORDS):
            return "DOWN"
        if re.search(r"[+＋]\s*\d", text):
            return "UP"
        if re.search(r"[-−－]\s*\d", text):
            return "DOWN"
        return None

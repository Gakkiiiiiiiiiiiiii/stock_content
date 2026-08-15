from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from stock_content.domain.claim_evidence_verifier import ClaimEvidenceVerifier
from stock_content.domain.financial_entity_normalizer import FinancialEntityNormalizer
from stock_content.domain.knowledge_enums import (
    LEGACY_TRUTH_ALIASES,
    EvidenceQualityStatus,
    ReviewStatus,
    SupportStatus,
    TruthStatus,
)

# verification_status（旧字段）仅做兼容映射，不再承载新逻辑（§22 三轴）。
_SUPPORT_TO_LEGACY_STATUS: dict[str, str] = {
    SupportStatus.UNSUPPORTED.value: "UNSUPPORTED",
    SupportStatus.SOURCE_LOCATED.value: "SOURCE_LOCATED",
    SupportStatus.SOURCE_SUPPORTED.value: "SOURCE_SUPPORTED",
    SupportStatus.CROSS_MODAL_SUPPORTED.value: "CROSS_MODAL_SUPPORTED",
    "NEEDS_REVIEW": "NEEDS_REVIEW",
}

# 高风险实体类型（股票/公司/指数/基金/机构）：纠错必须有非 LLM 证据（§51）。
_HIGH_RISK_ENTITY_TYPES = {"SECURITY", "STOCK", "COMPANY", "INDEX", "FUND", "INSTITUTION", "ETF"}
# 非 LLM 的实体解析证据来源。
_NON_LLM_RESOLUTION_METHODS = {"entity_dictionary", "ticker", "nearby_ocr"}


class KnowledgeUnitNormalizer:
    def __init__(
        self, entity_normalizer: FinancialEntityNormalizer | None = None, verifier: ClaimEvidenceVerifier | None = None
    ) -> None:
        self.entity_normalizer = entity_normalizer or FinancialEntityNormalizer()
        self.verifier = verifier or ClaimEvidenceVerifier()

    def normalize(self, units: list[dict], metadata: dict) -> list[dict]:
        source_date = self.parse_source_datetime(metadata.get("publish_time"))
        normalized: list[dict] = []
        for index, unit in enumerate(units, start=1):
            statement = self._clean_statement(unit.get("statement") or "")
            if not statement or not unit.get("evidence"):
                continue
            canonical = self._canonicalize(statement)
            entities = self._normalize_entities(unit, metadata)
            subject = self._infer_subject(unit, entities)
            subject_key = unit.get("subject_key") or subject.get("subject_key")
            if not subject_key:
                continue
            subject = {
                "subject_type": unit.get("subject_type") or subject.get("subject_type"),
                "subject_key": subject_key,
                "subject_name": unit.get("subject_name") or subject.get("subject_name"),
            }
            content_basis = "|".join(
                [
                    str(unit.get("chapter_index") or 0),
                    str(unit.get("primary_domain") or "GENERAL"),
                    str(unit.get("knowledge_kind") or "STATE"),
                    str(subject_key),
                    canonical,
                    str(unit.get("condition_text") or ""),
                    str(unit.get("invalidation_text") or ""),
                ]
            )
            content_hash = hashlib.sha256(content_basis.encode("utf-8")).hexdigest()
            semantic_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            uid_prefix = str(
                metadata.get("bvid") or metadata.get("platform_video_id") or metadata.get("platform") or "video"
            )
            item = dict(unit)
            legacy_status = unit.get("verification_status") or "SOURCE_LOCATED"
            # P0-5（§19）：实体规范化与主体推断之后，verifier 必须看到规范化后的候选单元。
            candidate = dict(unit)
            candidate.update(
                {
                    "statement": statement,
                    "entities": entities,
                    "subject_type": subject["subject_type"],
                    "subject_key": subject["subject_key"],
                    "subject_name": subject["subject_name"],
                }
            )
            verification = self.verifier.verify(candidate)
            support_status = str(verification.get("support_status") or SupportStatus.UNSUPPORTED.value)
            support_score = verification.get("support_score")
            if support_score is None:
                support_score = verification.get("support_probability")
            reasons = list(verification.get("reason_codes") or [])

            # P0-3（§9）：证据质量独立于语义支持，Unknown 不等于 Good。
            evidence_quality = self._evidence_quality(unit.get("evidence") or [])

            # P1-2（§51）：实体纠错 trace 入账；高风险实体仅有 LLM 自述时降质量上限。
            entity_resolution = self._entity_resolution_trace(unit, entities)
            if entity_resolution is not None and entity_resolution.get("status") == "UNVERIFIED":
                if evidence_quality == EvidenceQualityStatus.HIGH.value:
                    evidence_quality = EvidenceQualityStatus.MEDIUM.value
                reasons.append("ENTITY_RESOLUTION_UNVERIFIED")

            if support_status == SupportStatus.UNSUPPORTED.value:
                # UNSUPPORTED 原样流出，不被质量门改写（设计文档要求可查询）。
                verification_status = "UNSUPPORTED"
            elif evidence_quality == EvidenceQualityStatus.LOW.value:
                verification_status = "NEEDS_REVIEW"
                support_status = "NEEDS_REVIEW"
                reasons.append("EVIDENCE_QUALITY_LOW")
            elif evidence_quality == EvidenceQualityStatus.UNKNOWN.value:
                if self._ocr_high_confidence(unit.get("evidence") or []):
                    # 例外预留给 CROSS_MODAL：本次只记录 reason，不放行（§9）。
                    reasons.append("UNKNOWN_ASR_QUALITY_HIGH_CONFIDENCE_OCR")
                if support_status in {SupportStatus.SOURCE_SUPPORTED.value, SupportStatus.CROSS_MODAL_SUPPORTED.value}:
                    reasons.append("EVIDENCE_QUALITY_UNKNOWN_CAP")
                    support_status = SupportStatus.SOURCE_LOCATED.value
                verification_status = _SUPPORT_TO_LEGACY_STATUS.get(support_status, legacy_status)
            else:
                verification_status = _SUPPORT_TO_LEGACY_STATUS.get(support_status, legacy_status)

            if set(reasons) != set(verification.get("reason_codes") or []):
                # 质量门新增的 reason 一并写入 verification，随 ledger 入账可审计。
                verification = verification | {"reason_codes": reasons}
            attributes = (unit.get("attributes") or {}) | {"verification": verification}
            if entity_resolution is not None:
                attributes = attributes | {"entity_resolution": entity_resolution}
            truth_status = (
                LEGACY_TRUTH_ALIASES.get(
                    str(unit.get("truth_status") or "").strip().upper(),
                    str(unit.get("truth_status") or "").strip().upper(),
                )
                or TruthStatus.NOT_CHECKED.value
            )
            item.update(
                {
                    "knowledge_uid": f"ku_{uid_prefix}_{index:04d}_{content_hash[:10]}",
                    "statement": statement,
                    "canonical_statement": unit.get("canonical_statement") or canonical,
                    "entities": entities,
                    "subject_type": unit.get("subject_type") or subject.get("subject_type"),
                    "subject_key": unit.get("subject_key") or subject.get("subject_key"),
                    "subject_name": unit.get("subject_name") or subject.get("subject_name"),
                    "predicate_key": unit.get("predicate_key") or self._predicate_key(unit),
                    "content_hash": content_hash,
                    "semantic_hash": semantic_hash,
                    "conflict_key": self._conflict_key(unit, subject),
                    "scope_type": unit.get("scope_type") or subject.get("subject_type"),
                    "scope_key": unit.get("scope_key") or subject.get("subject_key"),
                    "verification_status": verification_status,
                    "support_status": support_status,
                    "support_probability": verification.get("support_probability", support_score),
                    "support_score": support_score,
                    "evidence_quality_status": evidence_quality,
                    "truth_status": truth_status,
                    "review_status": str(unit.get("review_status") or ReviewStatus.UNREVIEWED.value),
                    "external_verification_status": unit.get("external_verification_status") or "NOT_RUN",
                    "speaker_id": unit.get("speaker_id") or self._speaker_id(unit.get("evidence") or []),
                    "speaker_name": unit.get("speaker_name"),
                    "attribution_confidence": unit.get("attribution_confidence"),
                    "attributes": attributes,
                    "extractor_version": unit.get("extractor_version") or "v3.2-k3-json-mode",
                    "schema_version": "v1",
                    "as_of_time": unit.get("as_of_time") or source_date,
                }
            )
            normalized.append(item)
        return normalized

    @staticmethod
    def _speaker_id(evidence: list[dict]) -> str | None:
        return next((str(item.get("speaker_id")) for item in evidence if item.get("speaker_id")), None)

    @staticmethod
    def _evidence_quality(evidence: list[dict]) -> str:
        """P0-3（§9）EvidenceQualityStatus。

        无 evidence / 无时间窗 -> LOW；confidence 未测量 -> UNKNOWN；
        < 0.45 LOW，< 0.7 MEDIUM，否则 HIGH。Unknown 不等于 Good。
        """
        if not evidence:
            return EvidenceQualityStatus.LOW.value
        primary = next((item for item in evidence if item.get("is_primary")), evidence[0])
        has_time_range = primary.get("start_ms") is not None and primary.get("end_ms") is not None
        if not has_time_range:
            return EvidenceQualityStatus.LOW.value
        confidence = primary.get("confidence_score")
        if confidence is None:
            return EvidenceQualityStatus.UNKNOWN.value
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return EvidenceQualityStatus.UNKNOWN.value
        if value < 0.45:
            return EvidenceQualityStatus.LOW.value
        if value < 0.7:
            return EvidenceQualityStatus.MEDIUM.value
        return EvidenceQualityStatus.HIGH.value

    @staticmethod
    def _ocr_high_confidence(evidence: list[dict]) -> bool:
        """OCR 证据 mean_confidence >= 0.95（§9 例外，仅记录 reason 用）。"""
        for item in evidence:
            if str(item.get("source_type") or "").upper() not in {"OCR", "VISION", "FRAME"}:
                continue
            metrics = item.get("ocr_metrics") or {}
            score = metrics.get("mean_confidence")
            if score is None:
                score = item.get("confidence_score")
            try:
                if score is not None and float(score) >= 0.95:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _entity_resolution_trace(unit: dict, entities: list[dict]) -> dict | None:
        """P1-2（§51）：LLM 自述的实体纠错写入 entity_resolution，供 ledger 入账。

        高风险实体（股票/公司/指数/基金/机构）若仅有 LLM 自述、缺少
        entity_dictionary / ticker / nearby_ocr 任一非 LLM 证据，则标 UNVERIFIED。
        """
        corrections = unit.get("entity_corrections")
        if not isinstance(corrections, list) or not corrections:
            return None
        entity_type_by_name = {
            str(entity.get("entity_name") or ""): str(entity.get("entity_type") or "") for entity in entities
        }
        items: list[dict] = []
        unverified = False
        for raw in corrections:
            if not isinstance(raw, dict):
                continue
            canonical_name = str(raw.get("canonical_name") or "").strip()
            methods = [str(method).strip().lower() for method in raw.get("resolution_method") or []]
            entity_type = entity_type_by_name.get(canonical_name, "")
            high_risk = entity_type in _HIGH_RISK_ENTITY_TYPES or bool(raw.get("ticker"))
            verified = bool(set(methods) & _NON_LLM_RESOLUTION_METHODS)
            if high_risk and not verified:
                unverified = True
            items.append(
                {
                    "raw_expression": str(raw.get("raw_expression") or "").strip(),
                    "canonical_name": canonical_name,
                    "ticker": raw.get("ticker"),
                    "resolution_method": methods,
                    "confidence": raw.get("confidence"),
                    "high_risk": high_risk,
                    "non_llm_evidence": verified,
                }
            )
        if not items:
            return None
        return {"status": "UNVERIFIED" if unverified else "RESOLVED", "items": items}

    def _normalize_entities(self, unit: dict, metadata: dict) -> list[dict]:
        text = f"{unit.get('statement') or ''} {unit.get('evidence_text') or ''}"
        raw_entities = list(unit.get("entities") or [])
        extracted = self.entity_normalizer.extract_entities(text, "", metadata.get("title") or "")
        for entity in extracted:
            # Missing confidence 保持 UNKNOWN（None），0.0 原样保留。
            confidence = entity.get("confidence_score")
            if confidence is not None:
                confidence = float(confidence)
            raw_entities.append(
                {
                    "entity_type": entity.get("entity_type") or "UNKNOWN",
                    "entity_key": entity.get("ticker") or entity.get("name"),
                    "entity_name": entity.get("name") or entity.get("ticker") or "UNKNOWN",
                    "ticker": entity.get("ticker"),
                    "relation_role": "SUBJECT",
                    "confidence_score": confidence,
                }
            )
        deduped: dict[tuple[str, str], dict] = {}
        for entity in raw_entities:
            name = str(entity.get("entity_name") or entity.get("name") or entity.get("ticker") or "").strip()
            if not name:
                continue
            entity_type = str(entity.get("entity_type") or "UNKNOWN")
            key = str(entity.get("entity_key") or entity.get("ticker") or name)
            confidence = entity.get("confidence_score")
            if confidence is not None:
                confidence = float(confidence)
            deduped[(entity_type, key)] = {
                "entity_type": entity_type,
                "entity_key": key,
                "entity_name": name,
                "ticker": entity.get("ticker"),
                "relation_role": entity.get("relation_role") or "RELATED",
                # Missing confidence 保持 UNKNOWN（None），0.0 原样保留。
                "confidence_score": confidence,
            }
        return list(deduped.values())[:12]

    @staticmethod
    def parse_source_datetime(raw_value: str | None) -> datetime | None:
        text = str(raw_value or "").strip()
        if len(text) == 8 and text.isdigit():
            try:
                return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
            except ValueError:
                return None
        return None

    @staticmethod
    def _clean_statement(value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:1000]

    @staticmethod
    def _canonicalize(value: str) -> str:
        text = re.sub(r"\s+", "", value.strip())
        return text[:1000]

    @staticmethod
    def _infer_subject(unit: dict, entities: list[dict]) -> dict:
        for entity in entities:
            role = str(entity.get("relation_role") or "")
            if role == "SUBJECT" or entity.get("ticker"):
                return {
                    "subject_type": entity.get("entity_type"),
                    "subject_key": entity.get("entity_key") or entity.get("ticker") or entity.get("entity_name"),
                    "subject_name": entity.get("entity_name"),
                }
        return {}

    @staticmethod
    def _predicate_key(unit: dict) -> str:
        kind = str(unit.get("knowledge_kind") or "STATE").lower()
        statement = str(unit.get("statement") or "")
        if "支撑" in statement:
            return "support_level"
        if "压力" in statement or "阻力" in statement:
            return "resistance_level"
        if "减仓" in statement:
            return "reduce_position"
        if "加仓" in statement:
            return "increase_position"
        return kind

    @staticmethod
    def _conflict_key(unit: dict, subject: dict) -> str:
        return "|".join(
            [
                str(unit.get("primary_domain") or "GENERAL"),
                str(unit.get("knowledge_kind") or "STATE"),
                str(subject.get("subject_key") or "UNKNOWN"),
                KnowledgeUnitNormalizer._predicate_key(unit),
                str(unit.get("timeframe") or ""),
            ]
        )

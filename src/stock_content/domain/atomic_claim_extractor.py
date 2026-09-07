"""Stage 2 atomic claim extraction over one semantic context."""

from __future__ import annotations

import json
import re
from typing import Any

from .claim_draft import ClaimOccurrenceDraft, TemporalExpressionDraft
from .semantic_context_builder import SemanticContext


class AtomicClaimExtractor:
    """One extraction request per semantic segment, with an empty result valid."""

    name = "atomic_claim_extraction"

    def __init__(
        self,
        model_gateway: Any | None = None,
        *,
        model_id: str = "",
        prompt_version: str = "atomic-claims.v1",
        allow_offline_fixture: bool = True,
    ):
        self.model_gateway = model_gateway
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.allow_offline_fixture = allow_offline_fixture
        self.last_metrics: dict[str, float] = {}

    def extract(
        self,
        context: SemanticContext | dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        fixture_drafts: list[ClaimOccurrenceDraft | dict[str, Any]] | None = None,
        offline_fixture: bool = False,
    ) -> list[ClaimOccurrenceDraft]:
        semantic_segment_id = (
            context.semantic_segment_id
            if isinstance(context, SemanticContext)
            else str(context.get("semantic_segment_id") or "")
        )
        if fixture_drafts is not None:
            drafts = [self._validate(item, semantic_segment_id) for item in fixture_drafts]
        elif self.model_gateway is None or not bool(getattr(self.model_gateway, "available", lambda: True)()):
            if not (self.allow_offline_fixture or offline_fixture):
                raise RuntimeError("atomic claim extraction model gateway is unavailable")
            drafts = self._fixture_extract(context, semantic_segment_id)
        else:
            response = self._complete(self._prompt(context, metadata or {}))
            drafts = self._parse(response, semantic_segment_id)
        self.last_metrics = {
            "claim_count": float(len(drafts)),
            "zero_claim_context": 1.0 if not drafts else 0.0,
        }
        return drafts

    @staticmethod
    def _fixture_extract(
        context: SemanticContext | dict[str, Any], semantic_segment_id: str
    ) -> list[ClaimOccurrenceDraft]:
        """Explicit offline fixture adapter; never used for a normal request."""
        payload = context.__dict__ if isinstance(context, SemanticContext) else context
        segments = list(payload.get("transcript_segments") or [])
        drafts: list[ClaimOccurrenceDraft] = []
        for segment in segments:
            text = str(segment.get("raw_text") or segment.get("text") or "").strip()
            if len(text) < 6 or any(term in text for term in ("免责声明", "广告", "仅供参考")):
                continue
            idx = int(segment.get("segment_index", 0))
            segment_ticker = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
            # Retain punctuation boundaries so one transcript segment can
            # still yield multiple atomic claims deterministically.
            statements = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*", text) if part.strip()]
            for statement in statements:
                if not re.search(r"\d{3,6}|营收|收入|利润|业绩|毛利率|增长|风险|估值|订单|利好|利空", statement, re.I):
                    continue
                ticker = re.search(r"(?<!\d)(\d{6})(?!\d)", statement) or segment_ticker
                # ``盈利`` is a common offline-fixture synonym for earnings.
                # Keep this confined to the explicit fixture adapter so the
                # production eligibility policy remains authoritative.
                earnings = bool(ticker) or any(
                    x in statement for x in ("营收", "收入", "利润", "盈利", "业绩", "毛利率")
                )
                bullish = any(x in statement for x in ("增长", "利好", "改善"))
                bearish = any(x in statement for x in ("风险", "利空", "下滑"))
                # The offline adapter supplies a deterministic *candidate*
                # shape which is still validated against the authoritative
                # transcript below.  Its subject/predicate must be textual
                # values from the selected sentence; synthetic "fixture"
                # identities would correctly fail the atomic validator.
                predicate = next(
                    (
                        marker
                        for marker in (
                            "营收增长", "收入增长", "利润增长", "毛利率", "营收", "收入", "利润",
                            "业绩", "基本面", "估值", "订单", "增长", "放缓", "下滑", "风险", "利好", "利空",
                        )
                        if marker in statement
                    ),
                    statement.strip("。！？!?；; "),
                )
                subject_name = None
                if not ticker:
                    chinese_run = re.search(r"[\u4e00-\u9fff]{2,}", statement)
                    subject_name = chinese_run.group(0) if chinese_run else None
                temporal_match = re.search(
                    r"(?:19|20)\d{2}\s*(?:年\s*(?:第?[一二三四1234]季度|Q[1-4])|Q[1-4])",
                    statement,
                    re.IGNORECASE,
                )
                drafts.append(ClaimOccurrenceDraft(
                semantic_segment_id=semantic_segment_id,
                knowledge_kind="EARNINGS" if earnings else "CLAIM",
                claim_type="FINANCIAL_METRIC" if earnings else "INDUSTRY_RELATION",
                subject_type="EQUITY" if ticker else "CONTENT",
                subject_key=ticker.group(1) if ticker else "",
                subject_name=subject_name,
                predicate_key=predicate,
                conclusion=statement,
                value=None,
                sentiment="BULLISH" if bullish else "BEARISH" if bearish else "NEUTRAL",
                evidence_segment_indices=[idx],
                temporal_expressions=(
                    [
                        TemporalExpressionDraft(
                            role="REPORTING_PERIOD",
                            raw_expression=temporal_match.group(0),
                            evidence_segment_indices=[idx],
                            confidence=1.0,
                        )
                    ]
                    if temporal_match
                    else []
                ),
                extraction_model_id="offline-fixture",
                extraction_prompt_version="offline-fixture.v1",
                extraction_confidence=1.0,
                ))
        return drafts

    def _validate(self, value: ClaimOccurrenceDraft | dict[str, Any], semantic_segment_id: str) -> ClaimOccurrenceDraft:
        if isinstance(value, ClaimOccurrenceDraft):
            draft = value
        else:
            # The model is allowed to return the public ``claim.atomic.v1``
            # shape, but extraction still exposes the legacy occurrence DTO
            # to the next pipeline stage.  Retain only candidate facts here;
            # acceptance markers below are deliberately overwritten until the
            # transcript-local atomic validator accepts them.
            candidate = dict(value)
            subject = candidate.pop("subject", None)
            if isinstance(subject, dict):
                candidate.setdefault("subject_type", subject.get("subject_type"))
                candidate.setdefault("subject_key", subject.get("subject_key"))
                candidate.setdefault("subject_name", subject.get("subject_name"))
            if "predicate" in candidate:
                candidate.setdefault("predicate_key", candidate.pop("predicate"))
            object_value = candidate.pop("object", None)
            if isinstance(object_value, dict):
                candidate.setdefault("value", object_value.get("value", object_value.get("text")))
                candidate.setdefault("unit", object_value.get("unit"))
                candidate.setdefault("currency", object_value.get("currency"))
            candidate.setdefault("conclusion", candidate.get("normalized_statement", ""))
            # These keys are extractor/model metadata, not acceptance facts.
            # A model must never make its own output formally grounded.
            for key in (
                "polarity", "assertion_tense", "visual_anchors", "claim_schema_version",
                "grounding_status", "grounding_reason_codes", "contradiction_group_id",
                "legacy_grounding_incomplete",
            ):
                candidate.pop(key, None)
            draft = ClaimOccurrenceDraft.model_validate(candidate)
        if draft.semantic_segment_id and draft.semantic_segment_id != semantic_segment_id:
            raise ValueError("claim draft belongs to another semantic segment")
        return draft.model_copy(update={
            "semantic_segment_id": semantic_segment_id,
            "extraction_model_id": draft.extraction_model_id or self.model_id,
            "extraction_prompt_version": draft.extraction_prompt_version or self.prompt_version,
            # This extractor is an untrusted boundary.  Only
            # AtomicClaimValidationStage may replace these fail-closed values.
            "grounding_status": "LEGACY_UNGROUNDED",
            "grounding_reason_codes": [],
            "contradiction_group_id": None,
            "claim_schema_version": "claim.final.v1",
            "legacy_grounding_incomplete": True,
        })

    def _complete(self, prompt: str) -> Any:
        try:
            return self.model_gateway.complete(
                prompt=prompt,
                system="You are an atomic financial claim extractor. Return JSON only.",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except TypeError:
            return self.model_gateway.complete(
                messages=[
                    {"role": "system", "content": "You are an atomic financial claim extractor. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

    def _prompt(self, context: SemanticContext | dict[str, Any], metadata: dict[str, Any]) -> str:
        payload = context.__dict__ if isinstance(context, SemanticContext) else context
        return (
            "Extract atomic claims from this semantic context. Split independent claims. "
            "Include condition and invalidation text where present. Every evidence coordinate and every temporal "
            "expression must carry its own evidence_segment_indices. Do not invent dates, numbers, tickers, or "
            "evidence. Do not call another temporal model. Empty claims are valid. Return exactly "
            '{"claims":[...]} and no prose.\n'
            + json.dumps(
                {"metadata": metadata, "context": payload},
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
        )

    def _parse(self, response: Any, semantic_segment_id: str) -> list[ClaimOccurrenceDraft]:
        content = response.get("content", response) if isinstance(response, dict) else response
        if isinstance(content, str):
            content = json.loads(content)
        if not isinstance(content, dict) or set(content) != {"claims"} or not isinstance(content["claims"], list):
            raise ValueError("atomic extraction response must contain only claims")
        return [self._validate(item, semantic_segment_id) for item in content["claims"]]


__all__ = ["AtomicClaimExtractor"]

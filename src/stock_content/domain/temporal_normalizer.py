"""Deterministic temporal normalization; deliberately no model gateway."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any

from .temporal_semantics import (
    CalendarType,
    ClaimTemporalBinding,
    MetricTemporalNature,
    RecurrencePattern,
    TemporalAssertionStatus,
    TemporalGranularity,
    TemporalPrecision,
    TemporalRole,
    TemporalScope,
    TemporalValueType,
    expression_key_of,
    temporal_binding_id_of,
)
from .temporal_validator import validate_temporal_binding


class TemporalNormalizer:
    def __init__(self, reference_provider: Any = None, normalization_version: str = "normalization.v1"):
        self.reference_provider = reference_provider
        self.normalization_version = normalization_version

    def normalize(
        self,
        raw_expression: str,
        *,
        role: TemporalRole = TemporalRole.VALID_AT,
        anchor: datetime | date | None = None,
        subject_key: str = "",
        as_of: datetime | None = None,
        scope_hint: TemporalScope | None = None,
        evidence_refs: list[str] | None = None,
        metric_temporal_nature: MetricTemporalNature | None = None,
        assertion_status: TemporalAssertionStatus | None = None,
        timezone: str | None = None,
        calendar_type: CalendarType | None = None,
        calendar_id: str | None = None,
        market_session: str | None = None,
        granularity: TemporalGranularity | None = None,
    ) -> ClaimTemporalBinding:
        text = raw_expression.strip()
        parse_expression = self._strip_assertion_markers(text)
        binding = self._normalize(
            parse_expression,
            role=role,
            anchor=anchor,
            subject_key=subject_key,
            as_of=as_of,
            scope_hint=scope_hint,
            evidence_refs=evidence_refs,
        )
        inferred_assertion = self._infer_assertion_status(text, binding)
        updates = {
            "metric_temporal_nature": (
                metric_temporal_nature
                if metric_temporal_nature is not None
                else self._infer_metric_nature(text, binding)
            ),
            "assertion_status": (
                assertion_status
                if assertion_status is not None
                else inferred_assertion or binding.assertion_status
            ),
            "timezone": timezone,
            "calendar_type": calendar_type,
            "calendar_id": calendar_id,
            "market_session": market_session,
            "granularity": granularity,
        }
        updates = {key: value for key, value in updates.items() if value is not None}
        # Keep the source expression for auditability even when deterministic
        # parsing removed an assertion marker.  Partial identities must use
        # the original expression as well.
        if parse_expression != text:
            updates["raw_expression"] = text
            if binding.normalization_status.upper() in {"PARTIAL", "UNRESOLVED"}:
                updates["expression_key"] = expression_key_of(text, role, binding.scope)
        enriched = binding.model_copy(update=updates)
        enriched = enriched.model_copy(update={"temporal_binding_id": temporal_binding_id_of(enriched)})
        return validate_temporal_binding(enriched)

    @staticmethod
    def _strip_assertion_markers(text: str) -> str:
        """Remove deterministic assertion qualifiers before parsing a date.

        Assertion status remains a separate semantic field.  Keeping these
        qualifiers out of the date parser lets expressions such as
        ``预计2026Q2`` normalize exactly like ``2026Q2`` while preserving the
        original expression on the returned binding.
        """
        stripped = re.sub(
            r"(?:下修|上修|修订|修正|改到|计划|拟|预计|预期|"
            r"planned|expected|estimate|revised)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"^(?:在|于|到|for|in|by)\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+(?:在|于|到|for|in|by)\s*", " ", stripped, flags=re.IGNORECASE)
        return stripped.strip(" ，,：:;；()（）")

    @staticmethod
    def _infer_metric_nature(raw: str, binding: ClaimTemporalBinding) -> MetricTemporalNature | None:
        """Infer only the unambiguous financial horizon labels.

        Metric nature is normally supplied by Stage 2.  These four labels are
        explicitly defined by the temporal contract, so recognizing them here
        is deterministic and does not turn this parser into a semantic judge.
        """
        label = (binding.period_label or raw).strip().upper()
        if re.search(r"(?:^|\b)(?:YTD)\b", label):
            return MetricTemporalNature.CUMULATIVE
        if re.search(r"(?:^|\b)(?:TTM|LTM)\b", label):
            return MetricTemporalNature.TRAILING
        if re.search(r"(?:^|\b)NTM\b", label):
            return MetricTemporalNature.FORWARD
        return None

    @staticmethod
    def _infer_assertion_status(raw: str, binding: ClaimTemporalBinding) -> TemporalAssertionStatus | None:
        normalized = re.sub(r"\s+", "", raw).casefold()
        if any(token in normalized for token in ("下修", "上修", "修订", "修正", "改到", "revised")):
            return TemporalAssertionStatus.REVISED
        if any(token in normalized for token in ("计划", "拟", "planned")):
            return TemporalAssertionStatus.PLANNED
        if any(token in normalized for token in ("预计", "预期", "expected", "estimate")):
            return TemporalAssertionStatus.EXPECTED
        return None

    def _normalize(
        self,
        raw_expression: str,
        *,
        role: TemporalRole = TemporalRole.VALID_AT,
        anchor: datetime | date | None = None,
        subject_key: str = "",
        as_of: datetime | None = None,
        scope_hint: TemporalScope | None = None,
        evidence_refs: list[str] | None = None,
    ) -> ClaimTemporalBinding:
        text = raw_expression.strip()
        normalized_text = re.sub(r"\s+", "", text).casefold()
        base = as_of or (
            anchor
            if isinstance(anchor, datetime)
            else datetime.combine(anchor, datetime.min.time())
            if isinstance(anchor, date)
            else None
        )
        # A timeless proposition has no asserted date.  It still remains
        # unavailable until its occurrence is ingested; TIMELESS is semantic,
        # not an epistemic timestamp.
        if (not normalized_text and scope_hint is TemporalScope.TIMELESS) or normalized_text in {
            "timeless",
            "无时间限定",
            "无时间性",
            "永恒",
            "永久",
        }:
            scope, assertion = self._role_semantics(role, TemporalScope.TIMELESS)
            return ClaimTemporalBinding(
                role=role,
                scope=TemporalScope.TIMELESS,
                value_type=TemporalValueType.NONE,
                precision=TemporalPrecision.UNKNOWN,
                assertion_status=assertion,
                normalization_status="NORMALIZED",
                normalization_version=self.normalization_version,
                raw_expression=text,
                source_evidence_refs=evidence_refs or [],
            )
        open_ended = self._open_ended(text, role, evidence_refs)
        if open_ended is not None:
            return open_ended
        # Explicit market-session expressions are safe to parse, but exchange
        # calendar identity/timezone are attached only when the reference port
        # supplies them.
        session = self._market_session(text)
        market_text = self._strip_market_session(text) if session else text
        if session and market_text and market_text != text:
            nested = self._normalize(
                market_text,
                role=role,
                anchor=anchor,
                subject_key=subject_key,
                as_of=as_of,
                scope_hint=scope_hint,
                evidence_refs=evidence_refs,
            )
            if nested.normalization_status != "UNRESOLVED":
                nested = nested.model_copy(update={"market_session": session})
                nested = self._apply_exchange_reference(nested, subject_key, base)
                return nested.model_copy(update={"temporal_binding_id": temporal_binding_id_of(nested)})
        # Date-only ISO has DATE semantics (datetime.fromisoformat also accepts
        # it, so check it before the timestamp parser).
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return self._date(text, date.fromisoformat(text), role, TemporalPrecision.DAY, evidence_refs)
        # exact ISO timestamp
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return self._timestamp(text, parsed, role, evidence_refs)
        except ValueError:
            pass
        try:
            parsed_date = date.fromisoformat(text)
            return self._date(text, parsed_date, role, TemporalPrecision.DAY, evidence_refs)
        except ValueError:
            pass
        year = base.year if base is not None else None
        m = re.fullmatch(r"(\d{4})[年\-/]?\s*(?:Q|第)?([1-4])(?:季度)?", text, re.I)
        if m:
            return self._period(text, int(m.group(1)), int(m.group(2)), role, evidence_refs)
        m = re.fullmatch(r"(\d{4})\s*[H半]([12])", text, re.I)
        if m:
            y, half = int(m.group(1)), int(m.group(2))
            start = date(y, 1 if half == 1 else 7, 1)
            end = date(y, 6 if half == 1 else 12, 30 if half == 1 else 31)
            return self._range(text, start, end, role, TemporalPrecision.HALF_YEAR, "HALF_YEAR", evidence_refs)
        m = re.fullmatch(r"(?:CY)?(\d{4})(?:年|Y)?", text, re.I)
        if m:
            y = int(m.group(1))
            return self._range(
                text, date(y, 1, 1), date(y, 12, 31), role, TemporalPrecision.YEAR, "ANNUAL", evidence_refs, f"CY{y}"
            )
        m = re.fullmatch(r"(\d{4})[-年](\d{1,2})(?:月)?", text)
        if m:
            y, month = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12:
                return self._range(
                    text,
                    date(y, month, 1),
                    date(y, month, monthrange(y, month)[1]),
                    role,
                    TemporalPrecision.MONTH,
                    "MONTHLY",
                    evidence_refs,
                    f"{y}-{month:02d}",
                )
        m = re.fullmatch(r"FY(\d{4})(?:Q([1-4]))?", text, re.I)
        if m:
            fiscal_ref = None
            if self.reference_provider and base is not None:
                resolver = getattr(self.reference_provider, "resolve_fiscal_calendar", None)
                fiscal_ref = resolver(subject_key, base) if resolver else None
                period_resolver = getattr(self.reference_provider, "resolve_period", None)
                resolved = period_resolver(subject_key, text, as_of or base) if period_resolver else None
                if resolved:
                    return self._resolved_period(text, resolved, role, evidence_refs)
            return self._unresolved(
                text,
                role,
                TemporalScope.INTERVAL if m.group(2) else TemporalScope.UNKNOWN,
                evidence_refs,
                period_label=text,
                normalization_status="PARTIAL",
                reference_snapshot_id=getattr(fiscal_ref, "reference_snapshot_id", None),
                reference_data_version=getattr(fiscal_ref, "data_version", None),
                reference_available_at=getattr(fiscal_ref, "available_at", None),
            )
        if year is not None and re.fullmatch(r"(?:今年|本年|当年)\s*[Q第]?([1-4])(?:季度)?", text, re.I):
            q = int(re.search(r"([1-4])", text).group(1))
            return self._period(text, year, q, role, evidence_refs)
        chinese_q = {"一": 1, "二": 2, "三": 3, "四": 4}
        m = re.fullmatch(r"(?:今年|本年|当年)?\s*第?([一二三四])季度", text)
        if m and year is not None:
            return self._period(text, year, chinese_q[m.group(1)], role, evidence_refs)
        m = re.fullmatch(r"(?:(\d{4}))?(YTD|TTM|LTM|NTM)", text, re.I)
        if m:
            label = m.group(2).upper()
            if base is None and not m.group(1):
                return self._unresolved(text, role, scope_hint or TemporalScope.UNKNOWN, evidence_refs)
            y = int(m.group(1)) if m.group(1) else base.year
            if label == "YTD":
                return self._range(
                    text,
                    date(y, 1, 1),
                    base.date() if base is not None and y == base.year else date(y, 12, 31),
                    role,
                    TemporalPrecision.YEAR,
                    "ANNUAL",
                    evidence_refs,
                    text,
                )
            if label in {"TTM", "LTM"}:
                if base is None:
                    return self._unresolved(text, role, scope_hint or TemporalScope.UNKNOWN, evidence_refs)
                end = base.date()
                return self._range(
                    text, end - timedelta(days=365), end, role, TemporalPrecision.EXACT, "DAILY", evidence_refs, label
                )
            if base is None:
                return self._unresolved(text, role, scope_hint or TemporalScope.UNKNOWN, evidence_refs)
            start = base.date()
            return self._range(
                text, start, start + timedelta(days=365), role, TemporalPrecision.EXACT, "DAILY", evidence_refs, label
            )
        if text in {"今天收盘", "今日收盘"} and base is not None:
            binding = self._date(text, base.date(), role, TemporalPrecision.DAY, evidence_refs)
            binding = binding.model_copy(update={"market_session": "REGULAR"})
            binding = self._apply_exchange_reference(binding, subject_key, base)
            return binding.model_copy(update={"temporal_binding_id": temporal_binding_id_of(binding)})
        if text in {"今天", "今日"} and base is not None:
            return self._date(text, base.date(), role, TemporalPrecision.DAY, evidence_refs)
        if text in {"昨天", "昨日"} and base is not None:
            return self._date(text, (base - timedelta(days=1)).date(), role, TemporalPrecision.DAY, evidence_refs)
        if text in {"明天", "明日"} and base is not None:
            return self._date(text, (base + timedelta(days=1)).date(), role, TemporalPrecision.DAY, evidence_refs)
        m = re.fullmatch(r"未来([一两三四五六七八九十\d]+)(?:到|至|-)?([一两三四五六七八九十\d]+)?个?季度", text)
        if m:
            return self._approximate_horizon(text, role, base, evidence_refs, "QUARTER", m.group(1), m.group(2))
        m = re.fullmatch(r"未来([一两三四五六七八九十\d]+)(?:到|至|-)?([一两三四五六七八九十\d]+)?个?月", text)
        if m:
            return self._approximate_horizon(text, role, base, evidence_refs, "MONTH", m.group(1), m.group(2))
        if text in {"近期", "中期", "年内", "年底左右", "下半年附近"}:
            return self._approximate_named(text, role, base, evidence_refs)
        if text in {"以后", "很久以后"}:
            return self._unresolved(text, role, scope_hint or TemporalScope.UNKNOWN, evidence_refs)
        if text.startswith(("每年", "每季度", "每月", "每周", "每日")):
            frequency = {
                "每年": "YEARLY",
                "每季度": "QUARTERLY",
                "每月": "MONTHLY",
                "每周": "WEEKLY",
                "每日": "DAILY",
            }[next(prefix for prefix in ("每季度", "每月", "每周", "每日", "每年") if text.startswith(prefix))]
            quarter_match = re.search(r"Q([1-4])", text, re.I)
            month_match = re.search(r"(?:第)?([1-9]|1[0-2])月", text)
            weekday_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7}
            weekday_match = re.search(r"周([一二三四五六日天])", text)
            return ClaimTemporalBinding(
                role=role,
                scope=TemporalScope.RECURRING,
                value_type=TemporalValueType.NONE,
                precision=(
                    TemporalPrecision.QUARTER
                    if quarter_match
                    else TemporalPrecision.MONTH
                    if month_match
                    else TemporalPrecision.UNKNOWN
                ),
                assertion_status=TemporalAssertionStatus.ACTUAL,
                recurrence=RecurrencePattern(
                    frequency=frequency,
                    months=[int(month_match.group(1))] if month_match else [],
                    quarters=[int(quarter_match.group(1))] if quarter_match else [],
                    weekdays=[weekday_map[weekday_match.group(1)]] if weekday_match else [],
                    rule_text=text,
                ),
                normalization_status="NORMALIZED",
                normalization_version=self.normalization_version,
                raw_expression=text,
                source_evidence_refs=evidence_refs or [],
            )
        return self._unresolved(text, role, scope_hint or TemporalScope.UNKNOWN, evidence_refs)

    @staticmethod
    def _count(value: str) -> int:
        values = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        try:
            return int(value)
        except ValueError:
            if value == "十":
                return 10
            if value.startswith("十"):
                return 10 + values.get(value[1:], 0)
            if value.endswith("十"):
                return values.get(value[:-1], 1) * 10
            return values.get(value, 0)

    @staticmethod
    def _shift_months(value: date, months: int) -> date:
        ordinal = value.year * 12 + value.month - 1 + months
        year, month0 = divmod(ordinal, 12)
        month = month0 + 1
        return date(year, month, min(value.day, monthrange(year, month)[1]))

    def _open_ended(self, text: str, role: TemporalRole, refs: list[str] | None):
        match = re.fullmatch(
            r"(?:自|从)(\d{4})(?:年)?(?:Q([1-4]))?(?:起|开始|以来|onward|onwards)?",
            text,
            re.I,
        )
        if not match:
            match = re.fullmatch(r"(?:since|from)\s+(\d{4})(?:-([1-4]))?", text, re.I)
        if not match:
            return None
        year, quarter = int(match.group(1)), match.group(2)
        start = date(year, (int(quarter) - 1) * 3 + 1, 1) if quarter else date(year, 1, 1)
        label = f"{year}Q{quarter}" if quarter else f"CY{year}"
        scope, assertion = self._role_semantics(role, TemporalScope.OPEN_ENDED)
        return ClaimTemporalBinding(
            role=role,
            scope=TemporalScope.OPEN_ENDED,
            value_type=TemporalValueType.DATE,
            start_date=start,
            period_label=label,
            precision=TemporalPrecision.QUARTER if quarter else TemporalPrecision.YEAR,
            granularity=TemporalGranularity.QUARTERLY if quarter else TemporalGranularity.ANNUAL,
            assertion_status=assertion,
            normalization_status="NORMALIZED",
            normalization_version=self.normalization_version,
            raw_expression=text,
            source_evidence_refs=refs or [],
        )

    def _approximate_horizon(
        self,
        text: str,
        role: TemporalRole,
        base: datetime | None,
        refs: list[str] | None,
        unit: str,
        first: str,
        second: str | None,
    ) -> ClaimTemporalBinding:
        lower, embedded_upper = self._count_range(first)
        upper = self._count(second) if second else embedded_upper or lower
        if not lower or not upper or upper < lower:
            return self._unresolved(text, role, TemporalScope.FORECAST, refs)
        updates: dict[str, Any] = {}
        if base is not None:
            if unit == "QUARTER":
                earliest = self._shift_months(base.date(), lower * 3)
                latest = self._shift_months(base.date(), upper * 3)
            else:
                earliest = self._shift_months(base.date(), lower)
                latest = self._shift_months(base.date(), upper)
            updates = {
                "value_type": TemporalValueType.DATE,
                "earliest_end_date": earliest,
                "latest_end_date": latest,
            }
        return ClaimTemporalBinding(
            role=role,
            scope=TemporalScope.FORECAST,
            value_type=updates.pop("value_type", TemporalValueType.NONE),
            precision=TemporalPrecision.APPROXIMATE,
            granularity=TemporalGranularity.QUARTERLY if unit == "QUARTER" else TemporalGranularity.MONTHLY,
            assertion_status=TemporalAssertionStatus.EXPECTED,
            normalization_status="PARTIAL",
            normalization_version=self.normalization_version,
            raw_expression=text,
            expression_key=expression_key_of(text, role, TemporalScope.FORECAST),
            source_evidence_refs=refs or [],
            **updates,
        )

    def _count_range(self, value: str) -> tuple[int, int | None]:
        compact = value.replace("个", "")
        if len(compact) == 2 and all(char in "一两二三四五六七八九十" for char in compact):
            return self._count(compact[0]), self._count(compact[1])
        return self._count(compact), None

    def _approximate_named(
        self, text: str, role: TemporalRole, base: datetime | None, refs: list[str] | None
    ) -> ClaimTemporalBinding:
        if base is None:
            return self._unresolved(text, role, TemporalScope.FORECAST, refs, normalization_status="PARTIAL")
        today = base.date()
        if text == "近期":
            start, end = today, today + timedelta(days=90)
        elif text == "中期":
            start, end = today + timedelta(days=90), today + timedelta(days=365)
        elif text == "年内":
            start, end = today, date(today.year, 12, 31)
        elif text == "下半年附近":
            start, end = date(today.year, 7, 1), date(today.year, 12, 31)
        else:  # 年底左右
            start, end = date(today.year, 12, 1), date(today.year, 12, 31)
        return ClaimTemporalBinding(
            role=role,
            scope=TemporalScope.FORECAST,
            value_type=TemporalValueType.DATE,
            start_date=start,
            end_date=end,
            precision=TemporalPrecision.APPROXIMATE,
            granularity=TemporalGranularity.DAILY,
            assertion_status=TemporalAssertionStatus.EXPECTED,
            normalization_status="PARTIAL",
            normalization_version=self.normalization_version,
            raw_expression=text,
            expression_key=expression_key_of(text, role, TemporalScope.FORECAST),
            source_evidence_refs=refs or [],
        )

    @staticmethod
    def _market_session(text: str) -> str | None:
        for markers, session in (
            (("盘前", "pre-market", "premarket"), "PREMARKET"),
            (("盘后", "after-hours", "afterhours"), "AFTERHOURS"),
            (("集合竞价", "auction"), "AUCTION"),
            (("闭市", "休市", "closed"), "CLOSED"),
            (("收盘", "盘中收盘", "regular close"), "REGULAR"),
            (("开盘", "盘中", "regular"), "REGULAR"),
        ):
            if any(marker.casefold() in text.casefold() for marker in markers):
                return session
        return None

    @staticmethod
    def _strip_market_session(text: str) -> str:
        result = re.sub(
            r"(?:盘前|盘后|集合竞价|闭市|休市|收盘|开盘|盘中|pre[- ]?market|"
            r"after[- ]?hours|auction|closed|regular(?:\s+close)?)",
            "",
            text,
            flags=re.I,
        )
        return result.strip(" ，,：:()（）")

    def _apply_exchange_reference(
        self, binding: ClaimTemporalBinding, subject_key: str, base: datetime | None
    ) -> ClaimTemporalBinding:
        if not self.reference_provider or base is None:
            return binding
        reference = self.reference_provider.resolve_exchange_calendar(subject_key, base)
        if not reference:
            return binding
        return binding.model_copy(
            update={
                "calendar_type": CalendarType.EXCHANGE,
                "calendar_id": reference.calendar_id,
                "timezone": reference.timezone,
                "reference_snapshot_id": reference.reference_snapshot_id,
                "reference_data_version": reference.data_version,
                "reference_available_at": reference.available_at,
            }
        )

    def _date(self, raw, value, role, precision, refs):
        scope, assertion = self._role_semantics(role, TemporalScope.POINT)
        return ClaimTemporalBinding(
            role=role,
            scope=scope,
            value_type=TemporalValueType.DATE,
            start_date=value,
            end_date=value,
            precision=precision,
            granularity=TemporalGranularity.DAILY,
            assertion_status=assertion,
            normalization_status="NORMALIZED",
            normalization_version=self.normalization_version,
            raw_expression=raw,
            source_evidence_refs=refs or [],
        )

    def _timestamp(self, raw, value, role, refs):
        scope, assertion = self._role_semantics(role, TemporalScope.POINT)
        return ClaimTemporalBinding(
            role=role,
            scope=scope,
            value_type=TemporalValueType.TIMESTAMP,
            start_time=value,
            end_time=value,
            precision=TemporalPrecision.EXACT,
            granularity=TemporalGranularity.SECOND,
            assertion_status=assertion,
            normalization_status="NORMALIZED",
            normalization_version=self.normalization_version,
            raw_expression=raw,
            source_evidence_refs=refs or [],
        )

    def _period(self, raw, year, quarter, role, refs):
        start = date(year, quarter * 3 - 2, 1)
        end_month = quarter * 3
        end = date(year, end_month, monthrange(year, end_month)[1])
        return self._range(raw, start, end, role, TemporalPrecision.QUARTER, "QUARTERLY", refs, f"{year}Q{quarter}")

    def _range(self, raw, start, end, role, precision, granularity, refs, label=None):
        scope, assertion = self._role_semantics(role, TemporalScope.INTERVAL)
        return ClaimTemporalBinding(
            role=role,
            scope=scope,
            value_type=TemporalValueType.DATE,
            start_date=start,
            end_date=end,
            period_label=label,
            precision=precision,
            granularity=TemporalGranularity.__members__.get(granularity, TemporalGranularity.UNKNOWN),
            assertion_status=assertion,
            normalization_status="NORMALIZED",
            normalization_version=self.normalization_version,
            raw_expression=raw,
            source_evidence_refs=refs or [],
        )

    def _unresolved(
        self,
        raw,
        role,
        scope,
        refs,
        *,
        period_label=None,
        normalization_status="UNRESOLVED",
        reference_snapshot_id=None,
        reference_data_version=None,
        reference_available_at=None,
    ):
        if role is TemporalRole.FORECAST_TARGET or role == TemporalRole.FORECAST_TARGET:
            scope = TemporalScope.FORECAST
        elif role in {TemporalRole.CONDITION_PERIOD, TemporalRole.INVALIDATION_PERIOD}:
            scope = TemporalScope.INTERVAL
        return ClaimTemporalBinding(
            role=role,
            scope=scope,
            value_type=TemporalValueType.NONE,
            precision=TemporalPrecision.UNKNOWN,
            period_label=period_label,
            assertion_status=(
                TemporalAssertionStatus.EXPECTED if scope is TemporalScope.FORECAST else TemporalAssertionStatus.UNKNOWN
            ),
            normalization_status=normalization_status,
            normalization_version=self.normalization_version,
            raw_expression=raw,
            expression_key=expression_key_of(raw, role, scope),
            source_evidence_refs=refs or [],
            reference_snapshot_id=reference_snapshot_id,
            reference_data_version=reference_data_version,
            reference_available_at=reference_available_at,
        )

    @staticmethod
    def _role_semantics(role, default_scope):
        if role is TemporalRole.FORECAST_TARGET or role == TemporalRole.FORECAST_TARGET:
            return TemporalScope.FORECAST, TemporalAssertionStatus.EXPECTED
        if role is TemporalRole.CONDITION_PERIOD or role == TemporalRole.CONDITION_PERIOD:
            return TemporalScope.INTERVAL, TemporalAssertionStatus.ACTUAL
        if role is TemporalRole.INVALIDATION_PERIOD or role == TemporalRole.INVALIDATION_PERIOD:
            return TemporalScope.INTERVAL, TemporalAssertionStatus.ACTUAL
        return default_scope, TemporalAssertionStatus.ACTUAL

    def _resolved_period(self, raw, resolved, role, refs):
        is_quarter = bool(re.search(r"Q[1-4]", raw, re.I))
        raw_calendar = getattr(resolved, "calendar_type", CalendarType.FISCAL)
        try:
            calendar = CalendarType(raw_calendar)
        except ValueError:
            calendar = CalendarType.FISCAL
        binding = self._range(
            raw,
            resolved.start_date,
            resolved.end_date,
            role,
            TemporalPrecision.QUARTER if is_quarter else TemporalPrecision.YEAR,
            "QUARTERLY" if is_quarter else "ANNUAL",
            refs,
            getattr(resolved, "period_label", raw),
        )
        enriched = binding.model_copy(
            update={
                "calendar_type": calendar,
                "calendar_id": getattr(resolved, "calendar_id", None),
                "reference_snapshot_id": getattr(resolved, "reference_snapshot_id", None),
                "reference_data_version": getattr(resolved, "data_version", None),
                "reference_available_at": getattr(resolved, "available_at", None),
            }
        )
        return enriched.model_copy(update={"temporal_binding_id": temporal_binding_id_of(enriched)})


normalize_temporal_expression = TemporalNormalizer().normalize


__all__ = ["TemporalNormalizer", "normalize_temporal_expression"]

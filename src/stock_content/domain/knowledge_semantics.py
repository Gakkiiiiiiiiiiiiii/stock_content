"""Semantic envelope construction for the additive knowledge Bundle v2.

The extractor may provide the envelope directly, but this module is also the
single conservative normalisation point for production records.  In
particular, it never turns an unspecified date into the ingestion date and it
keeps speaker viewpoints distinct from externally checked facts.
"""

from __future__ import annotations

import re
from calendar import monthrange
from collections.abc import Mapping
from datetime import UTC, datetime, time
from typing import Any

_DOMAINS = frozenset(
    {
        "PORTFOLIO_RISK_MANAGEMENT",
        "FINANCIAL_SECTOR_CAPITAL_POLICY",
        "AI_INFERENCE_COMPUTE",
        "INFORMATION_INFRASTRUCTURE_POLICY",
        "CENTRAL_BANK_GOLD_RESERVES",
        "UNKNOWN",
    }
)
_ATTRIBUTED = frozenset({"OPINION", "FORECAST", "CAUSAL_THESIS"})
_COURSE_PREFIX = re.compile(r"^(?:(?:本)?课程|视频)(?:提出|设置|建议|强调|认为|指出)[：:，,、\s]*")
_YEAR = re.compile(r"(?:19|20)\d{2}")
_YEAR_ONLY = re.compile(r"^(?P<year>(?:19|20)\d{2})(?:年)?$")
_YEAR_MONTH = re.compile(r"^(?P<year>(?:19|20)\d{2})(?:-|年\s*)(?P<month>0?[1-9]|1[0-2])(?:月)?$")
_YEAR_MONTH_DAY = re.compile(
    r"^(?P<year>(?:19|20)\d{2})(?:-|年\s*)(?P<month>0?[1-9]|1[0-2])(?:-|月\s*)(?P<day>0?[1-9]|[12]\d|3[01])(?:日)?$"
)


def atomic_statement(value: Any) -> str:
    statement = _COURSE_PREFIX.sub("", str(value or "").strip())
    if not statement:
        raise ValueError("EMPTY_ATOMIC_STATEMENT")
    return statement


def primary_domain_for(statement: str, declared: Any = None) -> str:
    if str(declared or "") in _DOMAINS:
        return str(declared)
    text = statement.casefold()
    if any(token in text for token in ("资本补充", "银行资本", "银行", "保险")):
        return "FINANCIAL_SECTOR_CAPITAL_POLICY"
    if any(token in text for token in ("ai推理", "推理算力", "gpu", "算力", "芯片")):
        return "AI_INFERENCE_COMPUTE"
    if any(token in text for token in ("信息基础设施", "基础设施投资", "信息基建")):
        return "INFORMATION_INFRASTRUCTURE_POLICY"
    if any(token in text for token in ("黄金储备", "黄金", "央行储备")):
        return "CENTRAL_BANK_GOLD_RESERVES"
    if any(token in text for token in ("仓位", "资金池", "止损", "均线", "杠杆", "风险预算", "集中度")):
        return "PORTFOLIO_RISK_MANAGEMENT"
    return "UNKNOWN"


def claim_nature_for(claim_type: str, statement: str, declared: Any = None) -> str:
    if isinstance(declared, str) and declared.strip():
        return declared.strip().upper()
    kind = str(claim_type or "").upper()
    if kind == "FORECAST":
        return "FORECAST"
    if kind == "OPINION":
        return "OPINION"
    if kind == "INFERENCE":
        return "CAUSAL_THESIS" if any(x in statement for x in ("因为", "导致", "推动", "从而", "所以")) else "OPINION"
    if any(x in statement for x in ("应当", "不要", "建议", "我认为", "看好", "看空")):
        return "OPINION"
    factual_types = {
        "PRICE",
        "RETURN",
        "VALUATION",
        "FINANCIAL_METRIC",
        "CORPORATE_EVENT",
        "INDUSTRY_RELATION",
    }
    return "FACT" if kind in factual_types else "METHOD"


def _detail(value: Any, statement: str) -> dict[str, str | None]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    allowed = ("explanation", "mechanism", "procedure", "formula", "example", "scope", "risks")
    result = {key: (str(raw[key]).strip() if raw.get(key) is not None else None) for key in allowed}
    # A production model is instructed to provide rich detail.  This fallback
    # remains source-bound rather than inventing a generic risk disclaimer.
    if not any(result.values()):
        result["explanation"] = statement
    return result


def _utc_rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _calendar_interval(value: Any) -> tuple[str, str, str] | None:
    """Return a UTC interval for an explicitly stated calendar expression.

    The Bundle v2 wire schema deliberately has only RFC3339 instants.  A
    source's calendar-level expression is still useful, but must never leak
    as ``"2030"`` (or be silently anchored to ingestion time).  This helper
    translates only an explicit year/month/day to its calendar interval in
    UTC.  Expressions without a year, such as ``8月末``, intentionally remain
    unresolved because no source business year was supplied.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        instant = _utc_rfc3339(value)
        return instant, instant, "INSTANT"
    if not isinstance(value, str) or not (text := value.strip()):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        instant = _utc_rfc3339(parsed)
        return instant, instant, "INSTANT"
    match = _YEAR_MONTH_DAY.fullmatch(text)
    if match:
        year, month, day = (int(match.group(key)) for key in ("year", "month", "day"))
        try:
            start = datetime.combine(datetime(year, month, day).date(), time.min, tzinfo=UTC)
        except ValueError:
            return None
        return _utc_rfc3339(start), _utc_rfc3339(datetime.combine(start.date(), time.max, tzinfo=UTC)), "DAY"
    match = _YEAR_MONTH.fullmatch(text)
    if match:
        year, month = (int(match.group(key)) for key in ("year", "month"))
        start = datetime(year, month, 1, tzinfo=UTC)
        end = datetime(year, month, monthrange(year, month)[1], 23, 59, 59, 999999, tzinfo=UTC)
        return _utc_rfc3339(start), _utc_rfc3339(end), "MONTH"
    match = _YEAR_ONLY.fullmatch(text)
    if match:
        year = int(match.group("year"))
        return (
            _utc_rfc3339(datetime(year, 1, 1, tzinfo=UTC)),
            _utc_rfc3339(datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)),
            "YEAR",
        )
    # The extraction model may use a complete proposition as the label (for
    # example ``预计到2030年达到…``).  The year token is still explicit evidence,
    # so it can safely carry a forecast-target interval.  Do not apply this to
    # a year-less ``8月末`` expression.
    if (year_match := _YEAR.search(text)) is not None:
        year = int(year_match.group(0))
        return (
            _utc_rfc3339(datetime(year, 1, 1, tzinfo=UTC)),
            _utc_rfc3339(datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)),
            "YEAR",
        )
    return None


def _normalized_temporal(raw: Mapping[str, Any], expression: str) -> dict[str, Any]:
    """Normalize explicit v2 temporal values without inventing a business year."""
    kind = str(raw.get("kind") or "UNKNOWN")
    label = raw.get("label") or expression or next(
        (str(raw[field]).strip() for field in ("as_of", "start", "end") if str(raw.get(field) or "").strip()),
        None,
    )
    precision = str(raw.get("precision") or "UNKNOWN")
    result = {
        "kind": kind,
        "start": None,
        "end": None,
        "as_of": None,
        "rule": raw.get("rule"),
        "label": label,
        "precision": precision,
        "explicitly_unknown": False,
    }
    if kind == "RECURRING_RULE":
        return result
    # A single coarse source value is a whole calendar interval, rather than
    # an invented instant.  This is especially important for forecast targets:
    # ``2030`` means the 2030 target year, not 2030-01-01 alone.
    source_values = {
        field: raw.get(field)
        for field in ("start", "end", "as_of")
        if raw.get(field) is not None
    }
    if not source_values and expression:
        source_values = {"label": expression}
    normalized = {field: _calendar_interval(value) for field, value in source_values.items()}
    first_interval = next((interval for interval in normalized.values() if interval is not None), None)
    if first_interval is not None:
        _, _, detected_precision = first_interval
        if detected_precision != "INSTANT":
            result["precision"] = detected_precision
        if kind == "FORECAST_TARGET" and len(source_values) == 1:
            # A bare target date/year is represented as the evidence-bounded
            # interval, which is valid RFC3339 and preserves its precision.
            result["start"], result["end"] = first_interval[:2]
        else:
            if (interval := normalized.get("start")) is not None:
                result["start"] = interval[0]
            if (interval := normalized.get("end")) is not None:
                result["end"] = interval[1]
            if (interval := normalized.get("as_of")) is not None:
                # AS_OF consumes the latest instant of its explicit calendar
                # period.  A year-less month/day never reaches this branch.
                result["as_of"] = interval[1]
    # ``8月末`` is evidence of month/day semantics but not of a year.  Keep
    # the label and precision, while exposing no instant the consumer could
    # mistake for a real business date.
    if not any((result["start"], result["end"], result["as_of"])) and label and _YEAR.search(str(label)) is None:
        result["precision"] = "YEAR_UNSPECIFIED"
    return result


def _temporal(value: Any, expressions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    expression = next((str(x.get("raw_expression") or "").strip() for x in expressions or () if x), "")
    if raw.get("kind") in {"EVENT", "AS_OF", "FORECAST_TARGET", "RECURRING_RULE"}:
        return _normalized_temporal(raw, expression)
    if not expression:
        return {
            "kind": "UNKNOWN",
            "start": None,
            "end": None,
            "as_of": None,
            "rule": None,
            "label": None,
            "precision": "UNKNOWN",
            "explicitly_unknown": True,
        }
    if any(token in expression for token in ("每月", "每周", "每日", "交易日")):
        return {
            "kind": "RECURRING_RULE",
            "start": None,
            "end": None,
            "as_of": None,
            "rule": expression,
            "label": expression,
            "precision": "RULE",
            "explicitly_unknown": False,
        }
    if "预测" in expression or "预计" in expression or "目标" in expression or "2030" in expression:
        return _normalized_temporal({"kind": "FORECAST_TARGET", "label": expression}, expression)
    if any(token in expression for token in ("截至", "末", "月", "日")):
        return {
            "kind": "AS_OF",
            "start": None,
            "end": None,
            "as_of": None,
            "rule": None,
            "label": expression,
            "precision": "DAY" if _YEAR.search(expression) else "YEAR_UNSPECIFIED",
            "explicitly_unknown": False,
        }
    return {
        "kind": "UNKNOWN",
        "start": None,
        "end": None,
        "as_of": None,
        "rule": None,
        "label": expression,
        "precision": "UNRESOLVED",
        "explicitly_unknown": True,
    }


def bundle_v2_semantics(
    *,
    statement: str,
    claim_type: str,
    supplied: Any = None,
    temporal_expressions: list[dict[str, Any]] | None = None,
    review_reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Return a complete, conservative v2 semantic envelope.

    ``review_reason_codes`` is occurrence-local: a claim may be reused by a
    later source occurrence whose visual evidence does not conflict.
    """
    raw = dict(supplied) if isinstance(supplied, Mapping) else {}
    statement = atomic_statement(statement)
    nature = claim_nature_for(claim_type, statement, raw.get("claim_nature"))
    attribution_raw = dict(raw.get("attribution") or {})
    source_grade = str(raw.get("source_grade") or ("SOURCE_ASSERTION" if nature in _ATTRIBUTED else "UNKNOWN"))
    external_truth_status = str(raw.get("external_truth_status") or "NOT_CHECKED")
    # Secondary/unknown external facts remain a sourced assertion until an
    # independent primary source verifies them.
    attributed = (
        bool(attribution_raw.get("attributed"))
        or nature in _ATTRIBUTED
        or source_grade in {"SECONDARY", "UNKNOWN"}
    )
    attribution = {
        "attributed": attributed,
        "source_label": (
            str(attribution_raw.get("source_label") or "").strip() or ("source_speaker" if attributed else None)
        ),
    }
    if attribution_raw.get("speaker") is not None:
        attribution["speaker"] = str(attribution_raw["speaker"])
    raw_review = dict(raw.get("occurrence_review") or {})
    raw_reasons = review_reason_codes or raw_review.get("reason_codes") or []
    reasons = sorted({str(code) for code in raw_reasons if str(code)})
    review_status = "HUMAN_REVIEW_REQUIRED" if reasons else str(raw_review.get("status") or "NOT_REQUIRED")
    return {
        "primary_domain": primary_domain_for(statement, raw.get("primary_domain")),
        "claim_nature": nature,
        "attribution": attribution,
        "source_grade": source_grade,
        "detail": _detail(raw.get("detail"), statement),
        "temporal": _temporal(raw.get("temporal"), temporal_expressions),
        "occurrence_review": {
            "status": review_status,
            "reason_codes": reasons,
        },
        # This is deliberately separate from the source-grounding state. A
        # presenter forecast/opinion is a located source assertion, not an
        # externally verified investment fact.
        "external_truth_status": external_truth_status,
    }


__all__ = ["atomic_statement", "bundle_v2_semantics", "claim_nature_for", "primary_domain_for"]

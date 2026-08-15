"""金融口播数字结构化解析（P1-1 / 设计文档 §48-49）。

把 ``20%``、``一百五十亿``、``十来倍``、``不到两成`` 这类表达解析成
``FinancialNumericValue``，供 ``ClaimEvidenceVerifier`` 做数字 / 单位绑定校验。

原则：

- 禁止伪精确：约数 / 区间（``十来倍``）保留 ``min_value/max_value``，
  ``value`` 保持 ``None``，绝不取中值 15。
- 中文数字解析自实现（个十百千万亿），不依赖外部库。
- 比较器（``不到``/``超过``/``至少`` …）结构化为 ``comparator``，
  匹配时按区间 / 方向语义判断，不做字符串子串比较。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FinancialNumericValue:
    raw_expression: str
    value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    comparator: str | None = None  # GT/GTE/LT/LTE/EQ/APPROX/None
    approximate: bool = False
    unit: str | None = None  # PERCENT/MULTIPLE/CNY/CNY_YI/CNY_WAN/POINT/None
    metric: str | None = None  # PE/PB/PS/EPS/REVENUE/PROFIT/... 能推断才填


_CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_SMALL_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_BIG_UNIT = {"万": 10_000, "亿": 100_000_000}

_UNIT_BY_SUFFIX = {
    "%": "PERCENT",
    "％": "PERCENT",
    "倍": "MULTIPLE",
    "亿": "CNY_YI",
    "万": "CNY_WAN",
    "点": "POINT",
    "元": "CNY",
}

# 前缀比较器，按长度降序匹配，避免 "不超过" 被 "超过" 抢先命中。
_COMPARATOR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("不超过", "LTE"),
    ("不低于", "GTE"),
    ("不到", "LT"),
    ("不足", "LT"),
    ("低于", "LT"),
    ("少于", "LT"),
    ("超过", "GT"),
    ("高于", "GT"),
    ("多于", "GT"),
    ("大于", "GT"),
    ("至少", "GTE"),
    ("最少", "GTE"),
    ("至多", "LTE"),
    ("最多", "LTE"),
    ("大约", "APPROX"),
    ("约", "APPROX"),
)

# metric 推断关键词：越具体的越靠前。
_METRIC_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("净资产收益率", "ROE"), "ROE"),
    (("市盈率", "PE"), "PE"),
    (("市净率", "PB"), "PB"),
    (("市销率", "PS"), "PS"),
    (("每股收益", "EPS"), "EPS"),
    (("净利润", "利润", "盈利", "净利"), "PROFIT"),
    (("营业收入", "营收", "收入", "销售额"), "REVENUE"),
    (("毛利率",), "GROSS_MARGIN"),
    (("净利率",), "NET_MARGIN"),
    (("GDP",), "GDP"),
)

_CN_NUM = "零一二两三四五六七八九十百千万"
_CN_NUM_NO_BIG = "零一二两三四五六七八九十百千"

# 各模式按优先级依次套用，已占用区间不再重复匹配。
_RE_PERCENT_CN = re.compile(rf"百分之([{_CN_NUM}]+)")
# 十来倍 / 二十来倍 / 十几（个）亿
_RE_RANGE_LAI_JI = re.compile(rf"([零一二两三四五六七八九]{{0,2}}十)[来几](?:个)?(倍|亿|万|点|元|成)?")
# 二十多（个）点 / 一百五十多倍
_RE_RANGE_DUO = re.compile(rf"([零一二两三四五六七八九十百]+?)多(?:个)?(倍|亿|万|点|元|成)?")
# 一百五十亿 / 一万五千亿
_RE_CN_YI = re.compile(rf"([{_CN_NUM}]+)亿")
# 三千四百万（万后不跟亿）
_RE_CN_WAN = re.compile(rf"([{_CN_NUM_NO_BIG}]+)万(?!亿)")
# 两成 / 十五倍 / 三千四百点 / 三十元
_RE_CN_UNIT = re.compile(rf"([{_CN_NUM_NO_BIG}]+)(成|倍|点|元)")
_RE_ARABIC = re.compile(r"(\d+(?:\.\d+)?)\s*(%|％|倍|亿|万|点|元)?")


def _parse_cn_number(text: str) -> float | None:
    """解析中文数字（个十百千万亿），失败返回 None。"""

    if not text:
        return None
    total = 0.0
    section = 0.0
    digit = 0.0
    seen = False
    last_small_unit = 1
    zero_seen = False
    for ch in text:
        if ch in _CN_DIGIT:
            digit = float(_CN_DIGIT[ch])
            if ch == "零":
                zero_seen = True
            seen = True
        elif ch in _CN_SMALL_UNIT:
            unit = _CN_SMALL_UNIT[ch]
            section += (digit or 1.0) * unit
            digit = 0.0
            last_small_unit = unit
            zero_seen = False
            seen = True
        elif ch in _CN_BIG_UNIT:
            section = (section + digit) * _CN_BIG_UNIT[ch]
            total += section
            section = 0.0
            digit = 0.0
            last_small_unit = 1
            zero_seen = False
            seen = True
        else:
            return None
    if not seen:
        return None
    # 口语省略：一百五 = 150、三千四 = 3400（末尾数字按上一级单位的 1/10 折算），
    # 但 "零" 之后的是个位数（一百零五 = 105）。
    if digit and last_small_unit >= 10 and not zero_seen:
        digit *= max(last_small_unit // 10, 1)
    return total + section + digit


def _prefix_comparator(text: str, start: int) -> tuple[str | None, str]:
    """在数字表达式前 4 字内找比较器前缀（不到/超过/约…）。"""

    window = text[max(0, start - 4) : start]
    for word, comparator in _COMPARATOR_PREFIXES:
        if word in window:
            return comparator, word
    return None, ""


def _infer_metric(text: str, start: int, end: int) -> str | None:
    window = text[max(0, start - 8) : min(len(text), end + 4)]
    upper = window.upper()
    for keywords, metric in _METRIC_KEYWORDS:
        if any(keyword in window or keyword in upper for keyword in keywords):
            return metric
    return None


def parse_financial_numerics(text: str) -> list[FinancialNumericValue]:
    """从文本中抽取全部金融数字表达，按出现顺序返回。"""

    if not text:
        return []
    occupied: list[tuple[int, int]] = []
    found: list[tuple[int, FinancialNumericValue]] = []

    def spans_free(start: int, end: int) -> bool:
        return all(end <= occ_start or start >= occ_end for occ_start, occ_end in occupied)

    def register(match: re.Match[str], value: FinancialNumericValue) -> None:
        start, end = match.start(), match.end()
        comparator, prefix = _prefix_comparator(text, start)
        if comparator:
            value.comparator = comparator
            if comparator == "APPROX":
                value.approximate = True
            if prefix and text[start - len(prefix) : start] == prefix:
                start -= len(prefix)
        value.metric = _infer_metric(text, match.start(), match.end())
        value.raw_expression = text[start : match.end()]
        occupied.append((match.start(), match.end()))
        found.append((match.start(), value))

    def scan(pattern: re.Pattern[str], builder) -> None:
        for match in pattern.finditer(text):
            if not spans_free(match.start(), match.end()):
                continue
            value = builder(match)
            if value is not None:
                register(match, value)

    def cn_point(number: float, suffix: str | None) -> FinancialNumericValue:
        if suffix == "成":
            # §8.2：与 "20%" -> 20.0 PERCENT 同尺度（percentage points），一成=10、两成=20。
            return FinancialNumericValue(raw_expression="", value=number * 10.0, unit="PERCENT")
        return FinancialNumericValue(raw_expression="", value=number, unit=_UNIT_BY_SUFFIX.get(suffix or ""))

    def range_value(base: float, step: float, suffix: str | None) -> FinancialNumericValue:
        min_value, max_value = base, base + step - 1
        unit = _UNIT_BY_SUFFIX.get(suffix or "")
        if suffix == "成":
            min_value, max_value, unit = min_value * 10.0, max_value * 10.0, "PERCENT"
        return FinancialNumericValue(
            raw_expression="",
            min_value=min_value,
            max_value=max_value,
            approximate=True,
            unit=unit,
        )

    scan(_RE_PERCENT_CN, lambda m: FinancialNumericValue(raw_expression="", value=_parse_cn_number(m.group(1)), unit="PERCENT"))
    scan(_RE_RANGE_LAI_JI, lambda m: range_value(_parse_cn_number(m.group(1)) or 0.0, 10.0, m.group(2)))
    scan(
        _RE_RANGE_DUO,
        lambda m: range_value(
            _parse_cn_number(m.group(1)) or 0.0,
            10.0 if m.group(1).endswith("十") else 100.0,
            m.group(2),
        ),
    )
    scan(_RE_CN_YI, lambda m: FinancialNumericValue(raw_expression="", value=_parse_cn_number(m.group(1)), unit="CNY_YI"))
    scan(_RE_CN_WAN, lambda m: FinancialNumericValue(raw_expression="", value=_parse_cn_number(m.group(1)), unit="CNY_WAN"))
    scan(_RE_CN_UNIT, lambda m: cn_point(_parse_cn_number(m.group(1)) or 0.0, m.group(2)))
    scan(_RE_ARABIC, lambda m: FinancialNumericValue(raw_expression="", value=float(m.group(1)), unit=_UNIT_BY_SUFFIX.get(m.group(2) or "")))

    found.sort(key=lambda item: item[0])
    return [value for _, value in found]


def _interval(value: FinancialNumericValue) -> tuple[float, float] | None:
    """把数值折算成闭区间；比较器表达（不到/超过）由调用方单独处理。"""

    if value.min_value is not None and value.max_value is not None:
        return value.min_value, value.max_value
    if value.value is None:
        return None
    point = value.value
    if value.comparator == "APPROX" or value.approximate:
        return point * 0.9, point * 1.1
    return point, point


def _satisfies(interval: tuple[float, float], comparator: str, threshold: float) -> bool:
    lo, hi = interval
    if comparator == "LT":
        return hi < threshold
    if comparator == "LTE":
        return hi <= threshold
    if comparator == "GT":
        return lo > threshold
    if comparator == "GTE":
        return lo >= threshold
    return False


def numeric_values_match(claim_val: FinancialNumericValue, source_val: FinancialNumericValue) -> bool:
    """区间 / 比较器语义匹配（非子串比较）。

    - 单位不一致（% 对 倍）直接失败；
    - claim 点值落在 source 区间内（或反之）即匹配；
    - 比较器（不到/超过）要求 source 值满足方向约束，双方都是比较器时
      要求方向一致且阈值相同。
    """

    _DIRECTIONAL = {"LT", "LTE", "GT", "GTE"}
    if claim_val.unit and source_val.unit and claim_val.unit != source_val.unit:
        return False
    claim_directional = claim_val.comparator in _DIRECTIONAL and claim_val.value is not None
    source_directional = source_val.comparator in _DIRECTIONAL and source_val.value is not None
    if claim_directional and source_directional:
        return claim_val.comparator == source_val.comparator and abs(claim_val.value - source_val.value) < 1e-9
    if claim_directional:
        source_interval = _interval(source_val)
        return source_interval is not None and _satisfies(source_interval, claim_val.comparator, claim_val.value)
    if source_directional:
        claim_interval = _interval(claim_val)
        return claim_interval is not None and _satisfies(claim_interval, source_val.comparator, source_val.value)
    claim_interval = _interval(claim_val)
    source_interval = _interval(source_val)
    if claim_interval is None or source_interval is None:
        return False
    return claim_interval[0] <= source_interval[1] and source_interval[0] <= claim_interval[1]


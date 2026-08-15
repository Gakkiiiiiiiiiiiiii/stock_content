from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


logger = logging.getLogger(__name__)

ENTITY_ALIASES_PATH = Path("config") / "entity_aliases.yaml"


COMMON_ENTITY_ALIASES = {
    "上证指数": {"entity_type": "INDEX", "ticker": "000001.SH"},
    "沪指": {"entity_type": "INDEX", "ticker": "000001.SH"},
    "深证成指": {"entity_type": "INDEX", "ticker": "399001.SZ"},
    "创业板": {"entity_type": "INDEX", "ticker": "399006.SZ"},
    "恒生科技": {"entity_type": "INDEX", "ticker": "HSTECH"},
    "黄金": {"entity_type": "COMMODITY", "ticker": "XAUUSD"},
    "原油": {"entity_type": "COMMODITY", "ticker": "CL"},
    "美联储": {"entity_type": "MACRO", "ticker": "FED"},
    "半导体": {"entity_type": "INDUSTRY", "ticker": "SEMI"},
    "AI": {"entity_type": "THEME", "ticker": "AI"},
}

NON_COMPANY_NAME_TERMS = {
    "交易信息",
    "市场行情",
    "行情",
    "资讯",
    "服务",
    "前复权",
    "后复权",
    "不复权",
    "日线",
    "周线",
    "月线",
    "季线",
    "年线",
    "分时",
    "分钟",
    "现价",
    "今开",
    "涨跌",
    "涨幅",
    "总收入",
    "净利润",
    "总量",
    "总额",
    "总笔",
    "市值",
    "流通",
    "换手",
    "主力净额",
    "闭市阶段",
}

COMPANY_NAME_SUFFIXES = (
    "网络",
    "科技",
    "电子",
    "通信",
    "股份",
    "集团",
    "能源",
    "电气",
    "光电",
    "信息",
    "软件",
    "医疗",
    "药业",
    "生物",
    "材料",
    "数通",
    "电力",
    "环境",
    "制造",
    "智控",
    "智能",
    "资本",
)

COMPANY_NAME_LEADING_FILLERS = (
    "这里",
    "这页",
    "这个",
    "那个",
    "主要",
    "关于",
    "在讲",
    "讲的",
    "看看",
    "再看",
    "比如",
    "就是",
    "一下",
    "来看",
    "说到",
    "提到",
)

COMPANY_NAME_PATTERN = re.compile(
    rf"([\u4e00-\u9fff]{{2,4}}(?:{'|'.join(COMPANY_NAME_SUFFIXES)}))"
)

CODE_NAME_PATTERNS = (
    re.compile(r"(?:[A-Z]{0,3})?(\d{6})\s*([\u4e00-\u9fff]{2,8})"),
    re.compile(r"([\u4e00-\u9fff]{2,8})\s*(?:\(|（)?(?:[A-Z]{0,3})?(\d{6})"),
)


class FinancialEntityNormalizer:
    def __init__(self, aliases_path: str | Path | None = None) -> None:
        self.aliases = self._load_entity_aliases(aliases_path)

    def extract_entities(self, *texts: str) -> list[dict]:
        joined = " ".join(str(text or "") for text in texts if str(text or "").strip())
        results: list[dict] = []
        seen: set[str] = set()
        for alias, payload in self.aliases.items():
            if alias not in joined:
                continue
            entity_id = payload["ticker"]
            if entity_id in seen:
                continue
            results.append(
                {
                    "name": alias,
                    "ticker": payload["ticker"],
                    "entity_type": payload["entity_type"],
                }
            )
            seen.add(entity_id)
        for ticker, company_name in self._extract_code_name_pairs(joined):
            if ticker in seen or company_name in seen:
                continue
            results.append(
                {
                    "name": company_name,
                    "ticker": ticker,
                    "entity_type": "EQUITY",
                }
            )
            seen.add(ticker)
            seen.add(company_name)
        for code in re.findall(r"\b\d{6}\b|\b\d{4}\.HK\b|\b[A-Z]{1,5}\b", joined, flags=re.IGNORECASE):
            normalized = code.upper()
            if normalized in seen:
                continue
            if normalized.isalpha() and len(normalized) <= 2:
                continue
            results.append(
                {
                    "name": normalized,
                    "ticker": normalized,
                    "entity_type": self._infer_entity_type(normalized),
                }
            )
            seen.add(normalized)
        for company_name in self._extract_company_names(joined):
            if company_name in seen:
                continue
            results.append(
                {
                    "name": company_name,
                    "ticker": company_name,
                    "entity_type": "EQUITY",
                }
            )
            seen.add(company_name)
        return results

    @staticmethod
    def _load_entity_aliases(aliases_path: str | Path | None = None) -> dict[str, dict]:
        path = Path(aliases_path) if aliases_path else project_root() / ENTITY_ALIASES_PATH
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("实体别名词典加载失败（%s），回退内置别名表", path, exc_info=True)
            return dict(COMMON_ENTITY_ALIASES)
        entries = data.get("aliases")
        if not isinstance(entries, dict):
            logger.warning("实体别名词典为空或格式不正确（%s），回退内置别名表", path)
            return dict(COMMON_ENTITY_ALIASES)
        merged = dict(COMMON_ENTITY_ALIASES)
        for name, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            ticker = str(payload.get("ticker") or "").strip()
            entity_type = str(payload.get("entity_type") or "").strip()
            if not ticker or not entity_type:
                continue
            merged[str(name).strip()] = {"entity_type": entity_type, "ticker": ticker}
        return merged

    def normalize_time_expression(self, text: str, publish_time: str | None) -> dict[str, str | None]:
        normalized = str(text or "").strip()
        if not normalized:
            return {"time_expression": None, "normalized_time_start": None, "normalized_time_end": None}
        anchor = self._parse_publish_date(publish_time)
        if anchor is None:
            return {"time_expression": normalized, "normalized_time_start": None, "normalized_time_end": None}
        if "下周" in normalized:
            start = anchor + timedelta(days=(7 - anchor.weekday()))
            end = start + timedelta(days=6)
            return self._build_time_payload(normalized, start, end)
        if "本周" in normalized or "这周" in normalized:
            start = anchor - timedelta(days=anchor.weekday())
            end = start + timedelta(days=6)
            return self._build_time_payload(normalized, start, end)
        if "明天" in normalized:
            target = anchor + timedelta(days=1)
            return self._build_time_payload(normalized, target, target)
        if "今天" in normalized:
            return self._build_time_payload(normalized, anchor, anchor)
        if "短期" in normalized or "近期" in normalized:
            return {"time_expression": normalized, "normalized_time_start": None, "normalized_time_end": None}
        date_match = re.search(r"(20\d{2})[年/-]?(0?\d{1,2})[月/-]?(0?\d{1,2})日?", normalized)
        if date_match:
            try:
                target = datetime(
                    int(date_match.group(1)),
                    int(date_match.group(2)),
                    int(date_match.group(3)),
                    tzinfo=UTC,
                )
            except ValueError:
                return {"time_expression": normalized, "normalized_time_start": None, "normalized_time_end": None}
            return self._build_time_payload(normalized, target, target)
        return {"time_expression": normalized, "normalized_time_start": None, "normalized_time_end": None}

    @staticmethod
    def normalize_numeric_text(text: str) -> str:
        normalized = str(text or "")
        replacements = {
            "一": "1",
            "二": "2",
            "两": "2",
            "三": "3",
            "四": "4",
            "五": "5",
            "六": "6",
            "七": "7",
            "八": "8",
            "九": "9",
            "零": "0",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        normalized = normalized.replace("百分之", "")
        return normalized

    @staticmethod
    def infer_themes(text: str, known_themes: list[str] | None = None) -> list[str]:
        candidates = []
        merged = str(text or "")
        for theme in known_themes or []:
            if theme and theme in merged:
                candidates.append(theme)
        for fallback in ("黄金", "半导体", "AI", "消费", "券商", "银行", "新能源", "医药"):
            if fallback in merged and fallback not in candidates:
                candidates.append(fallback)
        return candidates

    @staticmethod
    def extract_company_names(text: str) -> list[str]:
        return FinancialEntityNormalizer._extract_company_names(text)

    @staticmethod
    def _infer_entity_type(value: str) -> str:
        if re.fullmatch(r"\d{4}\.HK", value):
            return "EQUITY"
        if re.fullmatch(r"\d{6}", value):
            return "EQUITY"
        if value in {"FED", "SEMI", "AI"}:
            return "THEME"
        return "UNKNOWN"

    @staticmethod
    def _extract_company_names(text: str) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for match in COMPANY_NAME_PATTERN.finditer(str(text or "")):
            name = FinancialEntityNormalizer._strip_company_leading_fillers(match.group(1).strip())
            if not FinancialEntityNormalizer._is_likely_company_name(name) or name in seen:
                continue
            seen.add(name)
            results.append(name)
        return results

    @staticmethod
    def _extract_code_name_pairs(text: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        seen: set[str] = set()
        source = str(text or "")
        for pattern in CODE_NAME_PATTERNS:
            for match in pattern.finditer(source):
                if pattern.pattern.startswith("(?:"):
                    ticker = match.group(1).strip()
                    raw_name = match.group(2).strip()
                else:
                    raw_name = match.group(1).strip()
                    ticker = match.group(2).strip()
                company_name = FinancialEntityNormalizer._strip_company_leading_fillers(raw_name)
                company_name = re.sub(r"[（(].*$", "", company_name).strip()
                if not FinancialEntityNormalizer._is_likely_company_name(company_name):
                    continue
                key = f"{ticker}:{company_name}"
                if key in seen:
                    continue
                seen.add(key)
                results.append((ticker, company_name))
        return results

    @staticmethod
    def _strip_company_leading_fillers(value: str) -> str:
        name = str(value or "").strip()
        changed = True
        while changed and name:
            changed = False
            for filler in COMPANY_NAME_LEADING_FILLERS:
                if name.startswith(filler) and len(name) > len(filler) + 1:
                    name = name[len(filler):].strip()
                    changed = True
        return name

    @staticmethod
    def _is_likely_company_name(value: str) -> bool:
        name = re.sub(r"[^\u4e00-\u9fff]", "", str(value or "").strip())
        if len(name) < 3 or len(name) > 8:
            return False
        if name in NON_COMPANY_NAME_TERMS:
            return False
        if name.endswith(("阶段", "指标", "策略", "市场", "行情", "资讯", "服务")):
            return False
        return True

    @staticmethod
    def _parse_publish_date(raw_value: str | None) -> datetime | None:
        text = str(raw_value or "").strip()
        if len(text) == 8 and text.isdigit():
            try:
                return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
            except ValueError:
                return None
        return None

    @staticmethod
    def _build_time_payload(raw_expression: str, start: datetime, end: datetime) -> dict[str, str]:
        return {
            "time_expression": raw_expression,
            "normalized_time_start": start.date().isoformat(),
            "normalized_time_end": end.date().isoformat(),
        }


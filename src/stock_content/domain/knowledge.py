from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from stock_content.domain.models import KnowledgeUnit, VideoChapter

_TICKER = re.compile(r"(?<!\d)(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)", re.IGNORECASE)
_BULLISH = ("增长", "利好", "上涨", "增持", "突破", "改善", "bullish")
_BEARISH = ("下跌", "利空", "风险", "减持", "恶化", "bearish")
_KIND_TERMS = {
    "CATALYST": ("催化", "订单", "政策利好", "涨价"),
    "RISK": ("风险", "下滑", "减持", "处罚"),
    "VALUATION": ("估值", "市盈率", "pe", "pb"),
    "EARNINGS": ("营收", "利润", "业绩", "毛利率"),
}


class KnowledgeExtractor:
    """Rules-first baseline; LLM extraction can be injected later without changing the pipeline."""

    @staticmethod
    def _kind(statement: str) -> str:
        lowered = statement.lower()
        for kind, terms in _KIND_TERMS.items():
            if any(term in lowered for term in terms):
                return kind
        return "CLAIM"

    @staticmethod
    def _sentiment(statement: str) -> str:
        lowered = statement.lower()
        bullish = sum(term in lowered for term in _BULLISH)
        bearish = sum(term in lowered for term in _BEARISH)
        return "BULLISH" if bullish > bearish else "BEARISH" if bearish > bullish else "NEUTRAL"

    def extract(
        self,
        video_id: str,
        chapters: list[VideoChapter],
        as_of: datetime | None = None,
    ) -> list[KnowledgeUnit]:
        timestamp = as_of or datetime.now(UTC)
        units: list[KnowledgeUnit] = []
        seen: set[str] = set()
        for chapter in chapters:
            chapter_ticker = _TICKER.search(chapter.summary)
            statements = re.split(r"(?<=[。！？!?；;])\s*", chapter.summary)
            for raw in statements:
                statement = re.sub(r"\s+", " ", raw).strip()
                if len(statement) < 6:
                    continue
                canonical = statement.casefold().rstrip("。！？!?；;")
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                match = _TICKER.search(statement) or chapter_ticker
                units.append(
                    KnowledgeUnit(
                        knowledge_uid=digest,
                        video_id=video_id,
                        chapter_id=chapter.chapter_id,
                        statement=statement,
                        kind=self._kind(statement),
                        subject=match.group(1) if match else None,
                        ticker=match.group(1) if match else None,
                        sentiment=self._sentiment(statement),
                        as_of=timestamp,
                        available_from=timestamp,
                    )
                )
        return units

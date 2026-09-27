"""Optional traditional-to-simplified transcript conversion adapter."""

from __future__ import annotations

import re


class OpenCCTranscriptConverter:
    """Lazy adapter; domain processing never imports OpenCC directly."""

    name = "opencc"
    _TRADITIONAL_HINTS = frozenset("體臺灣關藍曉設備彈萬與為於億傳儲兌冊劃務區協醫嚴壞壓壘壟壽夢夠奧")

    def __init__(self) -> None:
        self._converter = None
        self.version = "opencc.t2s"

    def convert(self, text: str) -> str:
        if not re.search(r"[\u3400-\u9fff]", text):
            return text
        try:
            import opencc
            from opencc import OpenCC
        except ImportError as exc:
            if any(char in self._TRADITIONAL_HINTS for char in text):
                raise RuntimeError("TRANSCRIPT_TRADITIONAL_CONVERTER_UNAVAILABLE_HUMAN_REVIEW") from exc
            return text
        if self._converter is None:
            self._converter = OpenCC("t2s")
            self.version = f"opencc.t2s@{getattr(opencc, '__version__', 'unknown')}"
        return str(self._converter.convert(text))

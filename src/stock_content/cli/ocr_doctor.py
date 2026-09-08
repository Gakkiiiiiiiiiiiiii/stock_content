"""Fail-closed isolated OCR GPU health check."""

from __future__ import annotations

import json

from stock_content.adapters.media.ocr import PaddleOcrEngine


def main() -> None:
    engine = PaddleOcrEngine()
    try:
        print(json.dumps(engine.start_and_probe(), sort_keys=True))
    finally:
        engine.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


class PaddleOcrEngine:
    def __init__(
        self,
        *,
        predictor_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._predictor_factory = predictor_factory or self._build_predictor
        self._predictor: Any | None = None

    def recognize(self, frame_path: str) -> dict:
        result = self._predictor_for_request().predict(str(Path(frame_path)))
        blocks = []
        for item in result:
            payload = item.json if hasattr(item, "json") else item
            for text, score, bbox in zip(
                payload.get("rec_texts", []), payload.get("rec_scores", []), payload.get("rec_boxes", []), strict=False
            ):
                blocks.append({"text": str(text), "score": float(score), "bbox": bbox})
        return {
            "text": "\n".join(block["text"] for block in blocks),
            "blocks": blocks,
            "engine": "paddleocr",
            "engine_version": "3",
        }

    def _predictor_for_request(self) -> Any:
        if self._predictor is None:
            self._predictor = self._predictor_factory()
        return self._predictor

    @staticmethod
    def _build_predictor() -> Any:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("install stock-content[multimodal] to enable OCR") from exc
        return PaddleOCR(
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False
        )

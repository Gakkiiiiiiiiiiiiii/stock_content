from __future__ import annotations

from pathlib import Path


class PaddleOcrEngine:
    def recognize(self, frame_path: str) -> dict:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("install stock-content[multimodal] to enable OCR") from exc
        result = PaddleOCR(
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False
        ).predict(str(Path(frame_path)))
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

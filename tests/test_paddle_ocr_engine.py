from __future__ import annotations

from stock_content.adapters.media.ocr import PaddleOcrEngine


def test_paddle_ocr_predictor_is_initialized_once_and_keeps_frame_specific_output():
    initialized = []

    class Predictor:
        def predict(self, frame_path: str):
            return [
                {
                    "rec_texts": [f"recognized:{frame_path}"],
                    "rec_scores": [0.91],
                    "rec_boxes": [[1, 2, 30, 40]],
                }
            ]

    def build_predictor():
        initialized.append("created")
        return Predictor()

    engine = PaddleOcrEngine(predictor_factory=build_predictor)

    first = engine.recognize("first-frame.jpg")
    second = engine.recognize("second-frame.jpg")

    assert initialized == ["created"]
    assert first == {
        "text": "recognized:first-frame.jpg",
        "blocks": [{"text": "recognized:first-frame.jpg", "score": 0.91, "bbox": [1, 2, 30, 40]}],
        "engine": "paddleocr",
        "engine_version": "3",
    }
    assert second == {
        "text": "recognized:second-frame.jpg",
        "blocks": [{"text": "recognized:second-frame.jpg", "score": 0.91, "bbox": [1, 2, 30, 40]}],
        "engine": "paddleocr",
        "engine_version": "3",
    }

from __future__ import annotations

from stock_content.workers import content_worker


def test_video_startup_probe_always_closes_isolated_ocr_process(monkeypatch):
    closed = []

    class Engine:
        def start_and_probe(self):
            raise RuntimeError("GPU probe failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(content_worker, "PaddleOcrEngine", Engine)

    try:
        content_worker._probe_video_ocr_runtime()
    except RuntimeError as exc:
        assert str(exc) == "GPU probe failed"
    else:  # pragma: no cover - keeps the cleanup assertion tied to a failed probe.
        raise AssertionError("expected the GPU probe to fail")

    assert closed == [True]

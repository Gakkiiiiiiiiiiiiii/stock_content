from pathlib import Path

from stock_content.adapters.media.frame import FfmpegFrameExtractor
from stock_content.domain.knowledge_frame_plan import KnowledgeFrameRequest


def _request(timestamp_ms: int, reason: str = "KNOWLEDGE_CENTER") -> KnowledgeFrameRequest:
    return KnowledgeFrameRequest(
        timestamp_ms=timestamp_ms,
        extraction_reason=reason,
        semantic_segment_ids=(f"semantic-{timestamp_ms}",),
        evidence_window_ids=(f"window-{timestamp_ms}",),
    )


def test_targeted_extract_uses_exact_ffmpeg_seek_and_preserves_request_provenance(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"unique-frame-" + command[3].encode())

    monkeypatch.setattr("stock_content.adapters.media.frame.subprocess.run", fake_run)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    frames = FfmpegFrameExtractor().extract_targeted(
        video, tmp_path, [_request(3_000), _request(1_234, "HIGH_SIGNAL")]
    )

    assert [call[2:8] for call in calls] == [
        ["-ss", "1.234", "-i", str(video), "-frames:v", "1"],
        ["-ss", "3.000", "-i", str(video), "-frames:v", "1"],
    ]
    assert [(item["timestamp_ms"], item["extraction_reason"]) for item in frames] == [
        (1_234, "HIGH_SIGNAL"),
        (3_000, "KNOWLEDGE_CENTER"),
    ]
    assert frames[0]["semantic_segment_ids"] == ["semantic-1234"]
    assert frames[0]["evidence_window_ids"] == ["window-1234"]


def test_targeted_extract_deduplicates_image_bytes_against_existing_and_targeted_frames(tmp_path, monkeypatch):
    def fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"duplicate-image")

    monkeypatch.setattr("stock_content.adapters.media.frame.subprocess.run", fake_run)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    existing_hash = __import__("hashlib").sha256(b"duplicate-image").hexdigest()
    assert FfmpegFrameExtractor().extract_targeted(
        video, tmp_path, [_request(1_000), _request(2_000)], existing_image_hashes={existing_hash}
    ) == []

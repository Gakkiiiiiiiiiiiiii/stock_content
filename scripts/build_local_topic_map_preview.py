"""Build a fresh GPT-6 Sol topic map for a hash-verified local preview.

This operator entry point is deliberately non-production: it validates a
previously materialized XiaoE video and its normalized transcript, preserves
the raw ASR rows, and writes new immutable local-review artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}")
    return value


def write_new(path: Path, value: dict) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def unresolved_entity_windows(legacy: dict | None, rows: list[dict], converter) -> list[dict]:
    if not legacy:
        return []
    windows: list[dict] = []
    seen: set[tuple[int, int, str]] = set()
    for mention in legacy.get("security_mentions") or []:
        if mention.get("relation") != "ENTITY_CORRECTION_PENDING":
            continue
        candidates = [
            converter.convert(str(mention.get("raw_asr_text") or "")).strip(),
            converter.convert(str(mention.get("asr_normalized_name") or "")).strip(),
        ]
        candidates = [text for text in candidates if len(text) >= 2]
        match_index = next(
            (
                index
                for index, row in enumerate(rows)
                if any(text in str(row.get("text") or "") for text in candidates)
            ),
            None,
        )
        if match_index is None:
            continue
        start = max(0, match_index - 1)
        end = min(len(rows) - 1, match_index + 1)
        raw_text = str(rows[match_index].get("text") or "")
        key = (start, end, raw_text)
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            {
                "start_segment_index": start,
                "end_segment_index": end,
                "raw_text": raw_text,
                "reason": "FIX01 ASR/visual entity alignment remained pending; do not infer a canonical name or code from this window.",
                "status": "UNRESOLVED",
            }
        )
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lesson-id", required=True)
    parser.add_argument("--source-transcript", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--legacy-knowledge", type=Path)
    parser.add_argument("--codex-cli", type=Path, required=True)
    parser.add_argument("--opencc-package-dir", type=Path, required=True)
    parser.add_argument("--topic-map-output", type=Path, required=True)
    parser.add_argument("--audio-review-output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str((args.repo / "src").resolve(strict=True)))
    sys.path.insert(0, str(args.opencc_package_dir.resolve(strict=True)))
    from opencc import OpenCC  # noqa: PLC0415 - operator-only dependency
    from stock_content.adapters.http.model_client import ContentModelClient  # noqa: PLC0415
    from stock_content.domain.artifacts import TranscriptArtifact, TranscriptSegmentItem  # noqa: PLC0415
    from stock_content.domain.semantic_segmenter import SemanticSegmenter  # noqa: PLC0415

    manifest = read_object(args.manifest)
    matching = [item for item in manifest.get("lessons", []) if item.get("lesson_id") == args.lesson_id]
    if len(matching) != 1:
        raise ValueError("Lesson identity missing or ambiguous in source manifest")
    lesson = matching[0]
    source = read_object(args.transcript)
    legacy = read_object(args.legacy_knowledge) if args.legacy_knowledge else None
    media_hash = sha256_file(args.media)
    source_hash = sha256_file(args.transcript)
    raw_hash = sha256_file(args.source_transcript)
    rows = source.get("segments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Transcript rows are missing")
    if (
        media_hash != lesson.get("video_sha256")
        or media_hash != source.get("media", {}).get("video_sha256")
        or source_hash != lesson.get("normalized_transcript_sha256")
        or raw_hash != lesson.get("source_transcript_sha256")
        or source.get("source_ref") != lesson.get("source_identity")
        or source.get("title") != lesson.get("title")
        or source.get("quality_report", {}).get("coverage_ratio", 0) < 0.90
        or source.get("quality_report", {}).get("timestamp_monotonic") is not True
        or source.get("postprocess", {}).get("normalized_text_is_simplified") is not True
        or any(row.get("segment_index") != index for index, row in enumerate(rows))
    ):
        raise ValueError("Local source provenance, transcript quality, or Simplified Chinese normalization failed")

    converter = OpenCC("t2s")
    for row in rows:
        text = str(row.get("text") or "")
        if text != str(row.get("normalized_text") or "") or converter.convert(text) != text:
            raise ValueError(f"Transcript row {row['segment_index']} is not normalized Simplified Chinese")
        if row.get("raw_text") is None:
            raise ValueError(f"Transcript row {row['segment_index']} does not preserve raw ASR text")
        if row["start_seconds"] < 0 or row["end_seconds"] < row["start_seconds"]:
            raise ValueError(f"Transcript row {row['segment_index']} has invalid coordinates")
        if row["end_seconds"] * 1000 > source["duration_ms"] + 1000:
            raise ValueError(f"Transcript row {row['segment_index']} exceeds media duration")

    cli_version = subprocess.run(
        [str(args.codex_cli), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    os.environ["CONTENT_MODEL_BACKEND"] = "codex_cli"
    os.environ["CONTENT_CODEX_CLI"] = str(args.codex_cli.resolve(strict=True))
    os.environ["CONTENT_MODEL_NAME"] = "gpt-6-sol"
    os.environ.pop("CONTENT_MODEL_URL", None)
    os.environ.pop("CONTENT_VISION_URL", None)
    os.environ.pop("CONTENT_MODEL_API_KEY", None)

    unresolved = unresolved_entity_windows(legacy, rows, converter)
    audio_review = {
        "schema_version": "local-audio-review.v1",
        "status": "REUSED_HASH_VERIFIED_FIX01_TRANSCRIPT_NO_NEW_CORRECTIONS",
        "source_transcript_sha256": source_hash,
        "raw_asr_sha256": raw_hash,
        "media_sha256": media_hash,
        "corrections": [],
        "unresolved_entity_windows": unresolved,
        "raw_asr_preserved": True,
        "normalized_text_is_simplified": True,
        "traditional_to_simplified_segment_count": source["postprocess"].get(
            "traditional_to_simplified_segment_count", 0
        ),
    }
    review_hash = write_new(args.audio_review_output, audio_review)

    transcript = TranscriptArtifact(
        artifact_id=f"local-transcript-{source_hash[:16]}",
        artifact_type="transcript",
        media_artifact_id=f"local-media-{media_hash[:16]}",
        language="zh-CN",
        asr_model=source["model"]["name"],
        asr_model_version=source["model"]["version"],
        segments=[
            TranscriptSegmentItem(
                segment_index=index,
                start_seconds=row["start_seconds"],
                end_seconds=row["end_seconds"],
                text=row["text"],
                raw_text=row["raw_text"],
                normalized_text=row["normalized_text"],
                media_artifact_id=f"local-media-{media_hash[:16]}",
                asr_model=source["model"]["name"],
                asr_model_version=source["model"]["version"],
            )
            for index, row in enumerate(rows)
        ],
    )

    class TracedGateway:
        def __init__(self) -> None:
            self.inner = ContentModelClient()
            self.calls = 0

        def available(self) -> bool:
            return self.inner.available()

        def complete(self, **kwargs):
            self.calls += 1
            print(f"GPT-6 Sol semantic call {self.calls}", flush=True)
            return self.inner.complete(**kwargs)

    gateway = TracedGateway()
    if not gateway.available():
        raise RuntimeError("Codex CLI is unavailable")
    segmenter = SemanticSegmenter(
        gateway,
        model_id="gpt-6-sol",
        prompt_version="semantic-segmentation.prompt.v4.codex-cli",
        allow_offline_fixture=False,
        require_model_identity=True,
        require_initial_topic=True,
        refine_segments=True,
        verify_brief_topic_labels=True,
    )
    result = segmenter.segment(transcript)
    if not result.segments or result.segments[0].start_segment_index != 0:
        raise ValueError("Initial topic or full coverage is missing")
    if result.segments[-1].end_segment_index != len(rows) - 1:
        raise ValueError("Final transcript row is not covered")
    payload = {
        "schema_version": "local-topic-map-codex-sol-v4",
        "status": "LOCAL_REVIEW_ONLY_NOT_SEALED_OR_FRONTEND_PUBLISHED",
        "run_kind": "FRESH_CODEX_LOCAL_TRANSCRIPT_REPARSE",
        "source_ref": source["source_ref"],
        "source_transcript_sha256": source_hash,
        "raw_asr_sha256": raw_hash,
        "media_sha256": media_hash,
        "audio_review_sha256": review_hash,
        "requested_model": "gpt-6-sol",
        "model_identity_source": "accepted_codex_cli_invocation_not_response_metadata",
        "codex_cli_version": cli_version,
        "segmentation_prompt_version": segmenter.prompt_version,
        "segment_count": len(result.segments),
        "metrics": {**result.metrics, "model_invocation_count": gateway.calls},
        "acceptance_checks": {
            "full_contiguous_coverage": True,
            "initial_topic_present": bool(result.segments[0].topic),
            "source_coverage_ratio": source["quality_report"]["coverage_ratio"],
            "main_text_simplified_chinese": True,
            "raw_asr_preserved": True,
        },
        "unresolved_entity_windows": unresolved,
        "external_fact_verification": "NOT_PERFORMED",
        "segments": [
            {
                "start": item.start_segment_index,
                "end": item.end_segment_index,
                "topic": converter.convert(item.topic),
                "subject": converter.convert(item.subject) if item.subject else None,
            }
            for item in result.segments
        ],
    }
    digest = write_new(args.topic_map_output, payload)
    print(
        json.dumps(
            {
                "lesson_id": args.lesson_id,
                "topic_count": len(result.segments),
                "unresolved_entity_windows": len(unresolved),
                "topic_map_sha256": digest,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

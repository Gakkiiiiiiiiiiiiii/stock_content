"""Finalize a completed frame-review draft after a transport-sized audit failure."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from build_local_coherent_knowledge_preview import (
    corrected_transcript,
    read_json,
    validate_topic_map,
    write_new,
)
from review_local_equity_frames import audit_prompt

from stock_content.adapters.codex_cli import CodexCliRunner
from stock_content.adapters.media.ocr import PaddleOcrEngine


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _downgrade(record: dict) -> None:
    focused = str(record.get("evidence_tier") or "").startswith("FOCUSED_CHART")
    record["evidence_tier"] = (
        "FOCUSED_CHART_VISUAL_ONLY" if focused else "SLIDE_ENTITY_VISUAL_ONLY"
    )
    record["identity_status"] = "CONFIRMED_ON_SCREEN_SPEECH_LINK_UNRESOLVED"
    record["spoken_connection"] = "UNSURE"
    record["transcript_evidence"] = []
    record["asr_correction_supported"] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--topic-map", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--audio-review", type=Path, required=True)
    parser.add_argument("--ocr-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.draft.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Frame-review draft must be an array")
    request, request_hash = read_json(args.request)
    topic_map, map_hash = read_json(args.topic_map)
    source, source_hash = read_json(args.transcript)
    audio, audio_hash = read_json(args.audio_review)
    media_hash = topic_map["media_sha256"]
    transcript = corrected_transcript(source, source_hash, media_hash, audio)
    validate_topic_map(topic_map, transcript, source_hash, media_hash)
    requested = {frame["image_sha256"]: frame for frame in request.get("frames") or []}
    root = args.request.resolve(strict=True).parent
    for record in records:
        for evidence in record.get("transcript_evidence") or []:
            index = evidence.get("segment_index")
            if not isinstance(index, int) or evidence.get("text") != transcript["segments"][index]["text"]:
                raise ValueError("Draft transcript evidence mismatch")
        for frame in record.get("visual_evidence") or []:
            expected = requested.get(frame.get("image_sha256"))
            if expected is None or expected.get("relative_path") != frame.get("relative_path"):
                raise ValueError("Draft frame is outside the reviewed request")
            path = (root / frame["relative_path"]).resolve(strict=True)
            if not path.is_relative_to(root) or _hash(path) != frame["image_sha256"]:
                raise ValueError("Draft frame path/hash mismatch")

    runner = CodexCliRunner(timeout_seconds=420)
    working = copy.deepcopy(records)
    audit = {"pass": False, "issues": ["not yet audited"]}
    for _ in range(3):
        audit = runner.run(
            system=(
                "You independently audit source-grounded Chinese visual entity records. "
                "Return JSON only; use no tools."
            ),
            prompt=audit_prompt(working),
        )["raw_response"] if working else {"pass": True, "issues": []}
        if audit.get("pass") is True and audit.get("issues") == []:
            break
        issue_ids = {
            issue.get("entity_id") for issue in audit.get("issues") or [] if isinstance(issue, dict)
        }
        if not issue_ids:
            break
        for record in working:
            if record.get("entity_id") not in issue_ids:
                continue
            message = " ".join(
                str(issue.get("issue") or "") for issue in audit.get("issues") or []
                if isinstance(issue, dict) and issue.get("entity_id") == record.get("entity_id")
            ).lower()
            if "market" in message or "市场" in message:
                record["market"] = None
            if "code" in message and any(
                marker in message for marker in ("unsupported", "invent", "unconfirmed")
            ):
                record["code"] = None
                record["code_status"] = "NOT_VISIBLE_IN_VIDEO"
            if any(marker in message for marker in (
                "spoken", "口述", "visual_only", "visual only", "speech link"
            )):
                _downgrade(record)
            record["context"] = (
                "画面确认同期实体的规范名称；代码和市场仅在实体专属画面文字明确出现时保存，"
                "口播关联按结构化状态单独表示。"
            )
    if audit.get("pass") is not True or audit.get("issues") != []:
        for record in working:
            _downgrade(record)
            record["market"] = None
        audit = runner.run(
            system=(
                "You independently audit source-grounded Chinese visual entity records. "
                "Return JSON only; use no tools."
            ),
            prompt=audit_prompt(working),
        )["raw_response"] if working else {"pass": True, "issues": []}
    if audit.get("pass") is not True or audit.get("issues") != []:
        raise RuntimeError("Draft entity records failed conservative independent audit")

    engine = PaddleOcrEngine(python_path=str(args.ocr_python), device="gpu:0", require_gpu=True)
    try:
        identity = engine.start_and_probe()
    finally:
        engine.close()
    if identity.get("actual_device", "").lower() != "gpu:0":
        raise RuntimeError("OCR runtime identity is not GPU")
    payload = {
        "schema_version": "local-equity-frame-review.v1",
        "status": "LOCAL_REVIEW_ONLY_NOT_SEALED_OR_PERSISTED",
        "requested_model": "gpt-6-sol",
        "model_identity_source": "accepted_codex_cli_invocation_not_response_metadata",
        "request_sha256": request_hash,
        "topic_map_sha256": map_hash,
        "source_transcript_sha256": source_hash,
        "audio_review_sha256": audio_hash,
        "media_sha256": media_hash,
        "ocr_runtime_identity": identity,
        "external_fact_verification": "NOT_PERFORMED",
        "candidate_coverage": request.get("unresolved_entity_coverage") or [],
        "skipped_frame_reviews": [],
        "mentions": working,
        "audit": {
            "passed": True,
            "method": "fresh compact GPT-6 Sol audit of completed per-frame GPU-OCR review draft",
            "issues": [],
        },
    }
    output_hash = write_new(args.output, payload)
    print(json.dumps({
        "status": "PASS_LOCAL_REVIEW_ONLY",
        "mentions": len(working),
        "output_sha256": output_hash,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

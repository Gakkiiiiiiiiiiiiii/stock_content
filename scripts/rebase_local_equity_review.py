"""Rebase hash-verified visual evidence onto a newly reviewed audio transcript."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

from build_local_coherent_knowledge_preview import (
    corrected_transcript,
    read_json,
    validate_topic_map,
    write_new,
)
from review_local_equity_frames import audit_prompt

from stock_content.adapters.codex_cli import CodexCliRunner

SPOKEN_TIERS = {"FOCUSED_CHART_SPOKEN", "SLIDE_ENTITY_SPOKEN"}
AUDIT_SYSTEM = (
    "You independently audit source-grounded Chinese visual entity records. "
    "Return JSON only; use no tools."
)


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
    parser.add_argument("--base-review", type=Path, required=True)
    parser.add_argument("--topic-map", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--audio-review", type=Path, required=True)
    parser.add_argument("--opencc-package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.opencc_package_dir))
    from opencc import OpenCC  # noqa: PLC0415

    base, base_hash = read_json(args.base_review)
    topic_map, map_hash = read_json(args.topic_map)
    source, source_hash = read_json(args.transcript)
    audio, audio_hash = read_json(args.audio_review)
    media_hash = topic_map["media_sha256"]
    transcript = corrected_transcript(source, source_hash, media_hash, audio)
    validate_topic_map(topic_map, transcript, source_hash, media_hash)
    if (
        base.get("audit", {}).get("passed") is not True
        or base.get("media_sha256") != media_hash
        or base.get("source_transcript_sha256") != source_hash
    ):
        raise ValueError("Base equity review provenance mismatch")

    converter = OpenCC("t2s")
    root = args.base_review.resolve(strict=True).parent
    destination = args.output.parent / "rebased-equity-frames"
    destination.mkdir(parents=True, exist_ok=True)
    mentions = copy.deepcopy(base.get("mentions") or [])
    copied: dict[str, str] = {}
    for mention in mentions:
        for field in ("name", "raw_entity_text", "context"):
            if isinstance(mention.get(field), str):
                mention[field] = converter.convert(mention[field])
        for evidence in mention.get("transcript_evidence") or []:
            index = evidence.get("segment_index")
            if not isinstance(index, int) or not 0 <= index < len(transcript["segments"]):
                raise ValueError("Base equity transcript coordinate is invalid")
            evidence["text"] = transcript["segments"][index]["text"]
        mention["raw_spoken_mentions"] = copy.deepcopy(mention.get("transcript_evidence") or [])
        for frame in mention.get("visual_evidence") or []:
            relative = frame.get("relative_path")
            image_hash = frame.get("image_sha256")
            if not isinstance(relative, str) or not isinstance(image_hash, str):
                raise ValueError("Base visual evidence is missing path/hash")
            source_path = (root / relative).resolve(strict=True)
            if not source_path.is_relative_to(root) or _hash(source_path) != image_hash:
                raise ValueError("Base visual evidence path/hash mismatch")
            if image_hash not in copied:
                target = destination / f"{image_hash[:24]}{source_path.suffix.lower() or '.jpg'}"
                shutil.copy2(source_path, target)
                copied[image_hash] = target.relative_to(args.output.parent).as_posix()
            frame["relative_path"] = copied[image_hash]

    runner = CodexCliRunner(timeout_seconds=420)
    audit = runner.run(
        system=AUDIT_SYSTEM,
        prompt=audit_prompt(mentions),
    )["raw_response"] if mentions else {"pass": True, "issues": []}
    if audit.get("pass") is not True or audit.get("issues") != []:
        issue_ids = {
            issue.get("entity_id") for issue in audit.get("issues") or [] if isinstance(issue, dict)
        }
        changed = False
        for mention in mentions:
            if mention.get("entity_id") in issue_ids and mention.get("evidence_tier") in SPOKEN_TIERS:
                _downgrade(mention)
                changed = True
        if changed:
            audit = runner.run(
                system=AUDIT_SYSTEM,
                prompt=audit_prompt(mentions),
            )["raw_response"]
    if audit.get("pass") is not True or audit.get("issues") != []:
        for mention in mentions:
            if mention.get("evidence_tier") in SPOKEN_TIERS:
                _downgrade(mention)
        audit = runner.run(
            system=AUDIT_SYSTEM,
            prompt=audit_prompt(mentions),
        )["raw_response"] if mentions else {"pass": True, "issues": []}
    if audit.get("pass") is not True or audit.get("issues") != []:
        raise RuntimeError("Rebased equity review did not pass independent audit")

    payload = {
        **base,
        "run_kind": "HASH_VERIFIED_VISUAL_EVIDENCE_REBASED_TO_FRESH_AUDIO_REVIEW",
        "base_review_sha256": base_hash,
        "topic_map_sha256": map_hash,
        "source_transcript_sha256": source_hash,
        "audio_review_sha256": audio_hash,
        "media_sha256": media_hash,
        "mentions": mentions,
        "audit": {
            "passed": True,
            "method": "fresh GPT-6 Sol audit after hash-verified frame and transcript reprojection",
            "issues": [],
        },
    }
    output_hash = write_new(args.output, payload)
    print(json.dumps({
        "status": "PASS_LOCAL_REVIEW_ONLY",
        "mentions": len(mentions),
        "frames": len(copied),
        "output_sha256": output_hash,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

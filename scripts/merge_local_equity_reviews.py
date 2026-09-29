"""Merge fresh targeted entity review with unaffected reviewed mentions, then audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from review_local_equity_frames import audit_prompt

from stock_content.adapters.codex_cli import CodexCliRunner


def read_json(path: Path) -> tuple[dict, str]:
    return json.loads(path.read_text(encoding="utf-8")), hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-review", type=Path, required=True)
    parser.add_argument("--overlay-review", type=Path, required=True)
    parser.add_argument("--overlay-name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base, base_hash = read_json(args.base_review)
    overlay, overlay_hash = read_json(args.overlay_review)
    for payload in (base, overlay):
        if payload.get("audit", {}).get("passed") is not True:
            raise ValueError("Source entity review has not passed audit")
    for key in ("source_transcript_sha256", "audio_review_sha256", "media_sha256"):
        if base.get(key) != overlay.get(key):
            raise ValueError(f"Entity review source mismatch: {key}")

    selected_names = set(args.overlay_name)
    mentions = [
        deepcopy(mention) for mention in base.get("mentions") or []
        if mention.get("name") not in selected_names
    ]
    overlays = [
        deepcopy(mention) for mention in overlay.get("mentions") or []
        if mention.get("name") in selected_names
    ]
    if {mention.get("name") for mention in overlays} != selected_names:
        raise ValueError("Requested overlay entity is missing")
    mentions.extend(overlays)
    mentions.sort(key=lambda mention: (mention.get("stage_ids") or ["T999"])[0])
    for index, mention in enumerate(mentions, start=1):
        mention["entity_id"] = f"local-equity-{index:02d}"
        mention.setdefault("market", None)
        mention.setdefault("code_status", "CONFIRMED_IN_VIDEO" if mention.get("code") else "NOT_VISIBLE_IN_VIDEO")
        mention.setdefault("raw_entity_text", "")
        mention.setdefault("raw_spoken_mentions", mention.get("transcript_evidence") or [])

    audit = CodexCliRunner(timeout_seconds=420).run(
        system="You independently audit source-grounded Chinese visual entity records. Return JSON only; use no tools.",
        prompt=audit_prompt(mentions),
    )["raw_response"]
    if audit.get("pass") is not True or audit.get("issues") != []:
        raise RuntimeError(f"Merged equity review audit failed: {len(audit.get('issues') or [])} issues")
    payload = {
        **overlay,
        "run_kind": "MERGED_REUSED_UNAFFECTED_AND_FRESH_TARGETED_ENTITY_REVIEW",
        "base_review_sha256": base_hash,
        "overlay_review_sha256": overlay_hash,
        "mentions": mentions,
        "audit": {
            "passed": True,
            "method": "fresh GPT-6 Sol merged entity record audit",
            "issues": [],
        },
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(encoded)
    print(json.dumps({
        "status": "PASS_LOCAL_REVIEW_ONLY",
        "mentions": len(mentions),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
    }))


if __name__ == "__main__":
    main()

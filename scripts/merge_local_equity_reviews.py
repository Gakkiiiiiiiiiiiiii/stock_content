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
    parser.add_argument("--overlay-name", action="append")
    parser.add_argument("--overlay-all", action="store_true")
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

    if not args.overlay_all and not args.overlay_name:
        raise ValueError("Either --overlay-all or --overlay-name is required")
    selected_names = set(args.overlay_name or [])
    if args.overlay_all:
        overlay_keys = {
            (mention.get("name"), tuple(mention.get("stage_ids") or []))
            for mention in overlay.get("mentions") or []
        }
        mentions = [
            deepcopy(mention) for mention in base.get("mentions") or []
            if (mention.get("name"), tuple(mention.get("stage_ids") or [])) not in overlay_keys
        ]
        overlays = [deepcopy(mention) for mention in overlay.get("mentions") or []]
    else:
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

    runner = CodexCliRunner(timeout_seconds=420)
    audit = {"pass": False, "issues": ["not yet audited"]}
    for _ in range(3):
        audit = runner.run(
            system=(
                "You independently audit source-grounded Chinese visual entity records. "
                "Return JSON only; use no tools."
            ),
            prompt=audit_prompt(mentions),
        )["raw_response"]
        if audit.get("pass") is True and audit.get("issues") == []:
            break
        print(
            "Merged entity review repair issues: "
            + json.dumps(audit.get("issues") or [], ensure_ascii=False),
            flush=True,
        )
        issue_ids = {
            issue.get("entity_id") for issue in audit.get("issues") or []
            if isinstance(issue, dict)
        }
        if not issue_ids:
            break
        for mention in mentions:
            if mention.get("entity_id") not in issue_ids:
                continue
            message = " ".join(
                str(issue.get("issue") or "") for issue in audit.get("issues") or []
                if isinstance(issue, dict) and issue.get("entity_id") == mention.get("entity_id")
            ).lower()
            if "market" in message or "市场" in message:
                mention["market"] = None
            if any(marker in message for marker in (
                "spoken", "口述", "visual_only", "visual only", "speech link", "phonetic"
            )):
                focused = str(mention.get("evidence_tier") or "").startswith("FOCUSED_CHART")
                mention["evidence_tier"] = (
                    "FOCUSED_CHART_VISUAL_ONLY" if focused else "SLIDE_ENTITY_VISUAL_ONLY"
                )
                mention["identity_status"] = "CONFIRMED_ON_SCREEN_SPEECH_LINK_UNRESOLVED"
                mention["spoken_connection"] = "UNSURE"
                mention["transcript_evidence"] = []
                mention["asr_correction_supported"] = False
            mention["context"] = (
                "画面确认同期实体的规范名称；代码和市场仅在实体专属画面文字明确出现时保存，"
                "口播关联按结构化状态单独表示。"
            )
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

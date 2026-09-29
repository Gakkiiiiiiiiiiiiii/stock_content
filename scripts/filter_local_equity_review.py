"""Remove visually reviewed occurrences whose frame is outside their declared topic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from review_local_equity_frames import audit_prompt

from stock_content.adapters.codex_cli import CodexCliRunner


def read_json(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--topic-map", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source, source_hash = read_json(args.source_review)
    topic_map, _ = read_json(args.topic_map)
    transcript, _ = read_json(args.transcript)
    if source.get("audit", {}).get("passed") is not True:
        raise ValueError("Source equity review did not pass")
    rows = transcript["segments"]
    stages = {
        f"T{index + 1:02d}": {
            "start_segment_index": topic["start"],
            "end_segment_index": topic["end"],
            "start_ms": round(rows[topic["start"]]["start_seconds"] * 1000),
            "end_ms": round(rows[topic["end"]]["end_seconds"] * 1000),
        }
        for index, topic in enumerate(topic_map["segments"])
    }
    kept = []
    excluded = []
    for mention in source["mentions"]:
        declared = [stages[stage_id] for stage_id in mention["stage_ids"]]
        if mention["evidence_tier"] in {"FOCUSED_CHART_VISUAL_ONLY", "SLIDE_ENTITY_VISUAL_ONLY"}:
            in_scope = any(
                stage["start_ms"] <= frame["timestamp_ms"] <= stage["end_ms"]
                for stage in declared
                for frame in mention["visual_evidence"]
            )
        else:
            in_scope = any(
                stage["start_segment_index"] <= item["segment_index"] <= stage["end_segment_index"]
                for stage in declared
                for item in mention["transcript_evidence"]
            )
        if in_scope:
            kept.append(mention)
        else:
            excluded.append(
                {
                    "entity_id": mention["entity_id"],
                    "reason": "VISUAL_ONLY_FRAME_OUTSIDE_DECLARED_TOPIC",
                }
            )
    runner = CodexCliRunner(timeout_seconds=420)
    audit = runner.run(
        system="You independently audit source-grounded Chinese visual entity records. Return JSON only; use no tools.",
        prompt=audit_prompt(kept),
    )["raw_response"]
    if audit.get("pass") is not True or audit.get("issues") != []:
        raise RuntimeError(f"Filtered equity review failed audit: {len(audit.get('issues', []))}")
    payload = {
        **source,
        "run_kind": "FILTERED_TO_FRAME_SCOPED_VISUAL_OCCURRENCES",
        "source_review_sha256": source_hash,
        "mentions": kept,
        "excluded_mentions": excluded,
        "audit": {
            "passed": True,
            "method": "fresh GPT-6 Sol frame-scope record audit",
            "issues": [],
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mentions": len(kept), "excluded": len(excluded)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

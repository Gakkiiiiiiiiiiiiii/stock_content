"""Reproject reviewed spoken entities into topic labels without altering evidence bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

SPOKEN_TIERS = {"FOCUSED_CHART_SPOKEN", "SLIDE_ENTITY_SPOKEN"}


def read_json(path: Path) -> tuple[dict, str]:
    return json.loads(path.read_text(encoding="utf-8")), hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-map", type=Path, required=True)
    parser.add_argument("--equity-review", type=Path, required=True)
    parser.add_argument("--output-topic-map", type=Path, required=True)
    parser.add_argument("--output-equity-review", type=Path, required=True)
    args = parser.parse_args()

    topic_map, map_hash = read_json(args.topic_map)
    review, review_hash = read_json(args.equity_review)
    if review.get("topic_map_sha256") != map_hash or review.get("audit", {}).get("passed") is not True:
        raise ValueError("Reviewed entities do not match the source topic map")
    projected_map = deepcopy(topic_map)
    projections = []
    resolved_window_indices: set[int] = set()
    for mention in review.get("mentions") or []:
        if mention.get("evidence_tier") not in SPOKEN_TIERS:
            continue
        raw = str(mention.get("raw_entity_text") or "").strip()
        canonical = str(mention.get("name") or "").strip()
        if not raw or not canonical or raw == canonical:
            continue
        for stage_id in mention.get("stage_ids") or []:
            if not isinstance(stage_id, str) or not stage_id.startswith("T") or not stage_id[1:].isdigit():
                raise ValueError("Invalid stage ID in reviewed entity")
            topic_index = int(stage_id[1:]) - 1
            if not 0 <= topic_index < len(projected_map.get("segments") or []):
                raise ValueError("Reviewed entity stage is outside topic map")
            topic = projected_map["segments"][topic_index]
            before = {"topic": topic.get("topic"), "subject": topic.get("subject")}
            for field in ("topic", "subject"):
                if isinstance(topic.get(field), str):
                    topic[field] = topic[field].replace(raw, canonical)
            projections.append({
                "entity_id": mention["entity_id"],
                "topic_index": topic_index,
                "raw_spoken_text": raw,
                "canonical_name": canonical,
                "before": before,
                "after": {"topic": topic.get("topic"), "subject": topic.get("subject")},
                "status": "REPROJECTED_FROM_REVIEWED_VISUAL_SPOKEN_LINK",
            })
        resolved_window_indices.update(
            evidence["segment_index"]
            for evidence in mention.get("transcript_evidence") or []
            if isinstance(evidence.get("segment_index"), int)
        )
    retained_windows = []
    resolved_windows = []
    for window in projected_map.get("unresolved_entity_windows") or []:
        covered = set(range(window["start_segment_index"], window["end_segment_index"] + 1))
        if covered.intersection(resolved_window_indices):
            resolved_windows.append({
                **window,
                "status": "RESOLVED_BY_REVIEWED_VISUAL_SPOKEN_LINK",
            })
        else:
            retained_windows.append(window)
    projected_map["unresolved_entity_windows"] = retained_windows
    projected_map["resolved_entity_windows"] = resolved_windows
    projected_map["entity_reprojections"] = projections
    projected_map["source_topic_map_sha256"] = map_hash
    projected_map["equity_review_sha256"] = review_hash
    projected_map["audio_review_sha256"] = review.get("audio_review_sha256")
    projected_hash = write_new(args.output_topic_map, projected_map)

    rebased_review = deepcopy(review)
    rebased_review["source_topic_map_sha256"] = map_hash
    rebased_review["topic_map_sha256"] = projected_hash
    rebased_review["topic_map_reprojection_status"] = "BOUNDARIES_UNCHANGED_LABELS_REPROJECTED"
    rebased_review["topic_map_reprojection_count"] = len(projections)
    rebased_review["resolved_entity_window_count"] = len(resolved_windows)
    rebased_hash = write_new(args.output_equity_review, rebased_review)
    print(json.dumps({
        "status": "PASS_LOCAL_REVIEW_ONLY",
        "projections": len(projections),
        "topic_map_sha256": projected_hash,
        "equity_review_sha256": rebased_hash,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

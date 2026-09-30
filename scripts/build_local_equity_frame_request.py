"""Build a strict visual-review request for local knowledge previews.

Legacy chart inventories remain hypotheses, but they are no longer the only
candidate source.  A previous knowledge draft may contribute unresolved entity
windows; those windows are sampled directly from the verified local media so a
garbled ASR name can be checked against slides, documents, subtitles, or charts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

from stock_content.adapters.media.frame import FfmpegFrameExtractor
from stock_content.domain.ambiguity_resolution import ambiguity_item_id, granular_unresolved_items

DEICTIC_ENTITY_REFERENCES = {
    "这个", "那个", "这两个", "那两个", "这几个", "那几个", "它", "它们", "他们", "她们",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_reviewed_frames(
    *, review_path: Path, review: dict, output: Path, rows: list[dict], topics: list[dict]
) -> tuple[list[dict], set[str]]:
    """Copy previously accepted evidence into a fresh, hash-checked request."""

    root = review_path.resolve(strict=True).parent
    destination = output.parent / "reviewed-equity-frames"
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    copied_hashes: set[str] = set()
    for mention_index, mention in enumerate(review.get("mentions") or [], start=1):
        evidence_indices = sorted({
            item["segment_index"] for item in mention.get("transcript_evidence") or []
            if isinstance(item.get("segment_index"), int)
        })
        stage_topic_indices = sorted({
            int(stage_id[1:]) - 1 for stage_id in mention.get("stage_ids") or []
            if isinstance(stage_id, str) and re.fullmatch(r"T\d+", stage_id)
        })
        fallback_indices = [
            index
            for topic_index in stage_topic_indices
            if 0 <= topic_index < len(topics)
            for index in range(topics[topic_index]["start"], topics[topic_index]["end"] + 1)
        ]
        source_indices = evidence_indices or fallback_indices
        for frame_index, frame in enumerate(mention.get("visual_evidence") or [], start=1):
            relative_path = frame.get("relative_path")
            image_hash = frame.get("image_sha256")
            if not isinstance(relative_path, str) or not isinstance(image_hash, str):
                raise ValueError("Reviewed frame is missing its relative path or hash")
            source_path = (root / relative_path).resolve(strict=True)
            if not source_path.is_relative_to(root) or sha256(source_path) != image_hash:
                raise ValueError(f"Reviewed frame path or hash mismatch: {source_path}")
            if image_hash in copied_hashes:
                continue
            target_path = destination / (
                f"{mention_index:02d}_{frame_index:02d}_{int(frame['timestamp_ms'])}_"
                f"{image_hash[:12]}{source_path.suffix.lower() or '.jpg'}"
            )
            shutil.copy2(source_path, target_path)
            timestamp_seconds = int(frame["timestamp_ms"]) / 1000
            nearby = source_indices or [
                index for index, row in enumerate(rows)
                if float(row["end_seconds"]) >= timestamp_seconds - 20
                and float(row["start_seconds"]) <= timestamp_seconds + 20
            ]
            if not nearby:
                raise ValueError("Reviewed frame has no transcript scope")
            focused = str(mention.get("evidence_tier") or "").startswith("FOCUSED_CHART")
            frames.append({
                "relative_path": target_path.relative_to(output.parent).as_posix(),
                "image_sha256": image_hash,
                "timestamp_ms": int(frame["timestamp_ms"]),
                "transcript_segment_range": [max(0, min(nearby) - 3), min(len(rows) - 1, max(nearby) + 3)],
                "candidate_name": mention.get("name"),
                "candidate_code": mention.get("code"),
                "candidate_market": mention.get("market"),
                "review_target_kind": "FOCUSED_CHART_SECURITY" if focused else "UNRESOLVED_ENTITY_WINDOW",
                "candidate_source": "PRIOR_HASH_VERIFIED_VISUAL_EVIDENCE_REVIEWED_FRESH",
                "knowledge_id": None,
                "raw_entity_text": mention.get("raw_entity_text"),
                "source_segment_indices": source_indices,
                "topic_indices": stage_topic_indices,
                "evidence_window_id": f"reviewed-{mention.get('entity_id') or mention_index}",
            })
            copied_hashes.add(image_hash)
    return frames, copied_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-knowledge", type=Path)
    parser.add_argument("--targeted-frames", type=Path)
    parser.add_argument("--ocr", type=Path)
    parser.add_argument("--base-request", type=Path)
    parser.add_argument("--base-review", type=Path)
    parser.add_argument("--ambiguity-triage", type=Path)
    parser.add_argument("--audio-review", type=Path)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--knowledge", type=Path)
    parser.add_argument("--topic-map", type=Path)
    parser.add_argument("--media", type=Path)
    parser.add_argument("--target-knowledge-id", action="append")
    parser.add_argument("--entity-frame-offset-ms", type=int, action="append")
    parser.add_argument("--skip-legacy-candidates", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entity_offsets = args.entity_frame_offset_ms or [-1000, 0, 1000]
    if not entity_offsets or len(entity_offsets) > 3 or len(set(entity_offsets)) != len(entity_offsets):
        raise ValueError("Entity frame offsets must contain one to three unique values")

    transcript = read_json(args.transcript)
    rows = transcript["segments"]
    legacy = read_json(args.legacy_knowledge) if args.legacy_knowledge else {}
    targeted = read_json(args.targeted_frames) if args.targeted_frames else {"frames": []}
    ocr = read_json(args.ocr) if args.ocr else {"frames": []}
    base_request = read_json(args.base_request) if args.base_request else {}
    base_review = read_json(args.base_review) if args.base_review else {}
    triage = read_json(args.ambiguity_triage) if args.ambiguity_triage else {}
    audio_review = read_json(args.audio_review) if args.audio_review else {}
    if not args.skip_legacy_candidates and not all(
        (args.legacy_knowledge, args.targeted_frames, args.ocr)
    ):
        raise ValueError(
            "--legacy-knowledge, --targeted-frames, and --ocr are required unless "
            "--skip-legacy-candidates is used"
        )
    frame_inventory = {item["image_hash"]: item for item in targeted["frames"]}
    raw_candidates = [] if args.skip_legacy_candidates else (
        list(legacy.get("security_mentions") or [])
        + list(legacy.get("displayed_target_candidates") or [])
    )
    candidates: dict[tuple[str, str], dict] = {}
    for item in raw_candidates:
        name = str(item.get("ocr_name") or "").strip()
        code = str(item.get("ticker") or "").strip()
        if name and re.fullmatch(r"\d{6}", code):
            candidates.setdefault((name, code), item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    copied_hashes: set[str] = set()
    if args.base_request:
        base_root = args.base_request.resolve(strict=True).parent
        base_destination = args.output.parent / "base-equity-frames"
        base_destination.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(base_request.get("frames") or [], start=1):
            relative_path = frame.get("relative_path")
            image_hash = frame.get("image_sha256")
            if not isinstance(relative_path, str) or not isinstance(image_hash, str):
                raise ValueError("Base request frame is missing its relative path or hash")
            source_path = (base_root / relative_path).resolve(strict=True)
            if sha256(source_path) != image_hash:
                raise ValueError(f"Base request frame hash mismatch: {source_path}")
            suffix = source_path.suffix.lower() or ".jpg"
            target_path = base_destination / f"{index:02d}_{int(frame['timestamp_ms'])}_{image_hash[:12]}{suffix}"
            shutil.copy2(source_path, target_path)
            copied = dict(frame)
            copied["relative_path"] = target_path.relative_to(args.output.parent).as_posix()
            frames.append(copied)
            copied_hashes.add(image_hash)

    if args.base_review:
        if not args.topic_map:
            raise ValueError("--topic-map is required with --base-review")
        reviewed_frames, reviewed_hashes = _copy_reviewed_frames(
            review_path=args.base_review,
            review=base_review,
            output=args.output,
            rows=rows,
            topics=read_json(args.topic_map).get("segments") or [],
        )
        frames.extend(frame for frame in reviewed_frames if frame["image_sha256"] not in copied_hashes)
        copied_hashes.update(reviewed_hashes)

    destination = args.output.parent / "equity-candidate-frames"
    destination.mkdir(parents=True, exist_ok=True)
    skipped = []
    for (name, code), candidate in candidates.items():
        matches = []
        for frame in ocr.get("frames") or []:
            top_text = " ".join(
                str(block.get("text") or "")
                for block in frame.get("blocks") or []
                if float(block.get("score") or 0) >= 0.8
                and (block.get("bbox") or [0, 999])[1] <= 60
            )
            if name in top_text and code in top_text:
                matches.append(
                    (
                        abs(int(frame["timestamp_ms"]) - int(candidate.get("timestamp_ms") or 0)),
                        frame,
                    )
                )
        if not matches:
            skipped.append({"name": name, "code": code, "reason": "NO_EXACT_TOP_HEADER_OCR"})
            continue
        _, ocr_frame = min(matches, key=lambda value: value[0])
        source = frame_inventory.get(ocr_frame["image_hash"])
        if not source:
            skipped.append({"name": name, "code": code, "reason": "FRAME_NOT_IN_HASHED_INVENTORY"})
            continue
        source_path = Path(source["frame_path"]).resolve(strict=True)
        if sha256(source_path) != source["image_hash"]:
            raise ValueError(f"Targeted frame hash mismatch: {source_path}")
        filename = f"{len(frames) + 1:02d}_{ocr_frame['timestamp_ms']}_{code}.jpg"
        target_path = destination / filename
        shutil.copy2(source_path, target_path)
        timestamp_seconds = int(ocr_frame["timestamp_ms"]) / 1000
        nearby = [
            index
            for index, row in enumerate(rows)
            if float(row["end_seconds"]) >= timestamp_seconds - 20
            and float(row["start_seconds"]) <= timestamp_seconds + 20
        ]
        if not nearby:
            raise ValueError(f"No transcript window near frame {ocr_frame['timestamp_ms']}")
        target_hash = sha256(target_path)
        if target_hash in copied_hashes:
            skipped.append({"name": name, "code": code, "reason": "DUPLICATE_BASE_FRAME"})
            continue
        frames.append(
            {
                "relative_path": target_path.relative_to(args.output.parent).as_posix(),
                "image_sha256": target_hash,
                "timestamp_ms": int(ocr_frame["timestamp_ms"]),
                "transcript_segment_range": [min(nearby), max(nearby)],
                "candidate_name": name,
                "candidate_code": code,
                "candidate_market": None,
                "review_target_kind": "FOCUSED_CHART_SECURITY",
                "candidate_source": "LEGACY_INVENTORY_HYPOTHESIS_NOT_ACCEPTED_WITHOUT_PIXEL_REVIEW",
            }
        )

    unresolved_coverage = []
    optional_entity_inputs = (args.knowledge, args.topic_map, args.media)
    if any(optional_entity_inputs) and not all(optional_entity_inputs):
        raise ValueError("--knowledge, --topic-map, and --media must be supplied together")
    if all(optional_entity_inputs):
        knowledge = read_json(args.knowledge)
        topic_map = read_json(args.topic_map)
        if args.ambiguity_triage:
            if (
                triage.get("schema_version") != "local-ambiguity-triage.v1"
                or triage.get("audit", {}).get("passed") is not True
                or triage.get("source_knowledge_sha256") != sha256(args.knowledge)
            ):
                raise ValueError("Ambiguity triage provenance mismatch")
        triage_by_id = {
            item["item_id"]: item for item in triage.get("decisions") or []
        }
        audio_by_id = {
            item["item_id"]: item for item in audio_review.get("ambiguity_item_decisions") or []
        }
        topics = topic_map.get("segments") or []
        cards = knowledge.get("knowledge") or []
        requests = []
        request_meta = []
        for card in cards:
            knowledge_id = str(card.get("knowledge_id") or "")
            if args.target_knowledge_id and knowledge_id not in set(args.target_knowledge_id):
                continue
            topic_indices = card.get("topic_indices") or []
            if any(not isinstance(index, int) or not 0 <= index < len(topics) for index in topic_indices):
                raise ValueError(f"Invalid topic indices for {knowledge_id}")
            entity_items = granular_unresolved_items(
                card.get("unresolved_items") or [], kinds={"ENTITY"}
            )
            for unresolved_index, item in enumerate(entity_items):
                raw_entity_text = str(item.get("raw_text") or "").strip()
                item_id = ambiguity_item_id(knowledge_id, item)
                decision = triage_by_id.get(item_id)
                audio_decision = audio_by_id.get(item_id)
                if decision and decision.get("review_text"):
                    raw_entity_text = decision["review_text"]
                if args.ambiguity_triage:
                    if decision is None:
                        unresolved_coverage.append({
                            "knowledge_id": knowledge_id,
                            "item_id": item_id,
                            "raw_text": raw_entity_text,
                            "status": "SKIPPED_NO_TRIAGE_DECISION",
                        })
                        continue
                    effective_kind = decision.get("corrected_kind") or item.get("kind")
                    entity_type = decision.get("entity_type")
                    should_extract = (
                        effective_kind == "ENTITY"
                        and entity_type == "EQUITY"
                        and (
                            decision.get("action") == "VISUAL_REVIEW_REQUIRED"
                            or decision.get("action") == "AUDIO_REVIEW_REQUIRED"
                            and audio_decision is not None
                            and audio_decision.get("decision") == "AUDIO_CANDIDATE_VISUAL_REQUIRED"
                        )
                    )
                    if not should_extract:
                        unresolved_coverage.append({
                            "knowledge_id": knowledge_id,
                            "item_id": item_id,
                            "raw_text": raw_entity_text,
                            "entity_type": entity_type,
                            "triage_action": decision.get("action"),
                            "audio_decision": (
                                audio_decision.get("decision") if audio_decision else None
                            ),
                            "status": "CLASSIFIED_NO_FRAME_REQUIRED",
                        })
                        continue
                if raw_entity_text in DEICTIC_ENTITY_REFERENCES:
                    unresolved_coverage.append({
                        "knowledge_id": knowledge_id,
                        "raw_text": raw_entity_text,
                        "status": "SKIPPED_DEICTIC_REFERENCE",
                    })
                    continue
                indices = sorted(set(item.get("segment_indices") or []))
                if not indices or any(not isinstance(index, int) or not 0 <= index < len(rows) for index in indices):
                    unresolved_coverage.append({
                        "knowledge_id": knowledge_id,
                        "raw_text": item.get("raw_text"),
                        "status": "SKIPPED_INVALID_SEGMENT_COORDINATES",
                    })
                    continue
                start_ms = round(float(rows[indices[0]]["start_seconds"]) * 1000)
                end_ms = round(float(rows[indices[-1]]["end_seconds"]) * 1000)
                center_ms = (start_ms + end_ms) // 2
                window_id = f"entity-{knowledge_id}-{unresolved_index + 1}"
                for offset_ms in entity_offsets:
                    timestamp_ms = max(0, center_ms + offset_ms)
                    requests.append(SimpleNamespace(
                        timestamp_ms=timestamp_ms,
                        extraction_reason="UNRESOLVED_ENTITY_HIGH_SIGNAL",
                        semantic_segment_ids=(knowledge_id,),
                        evidence_window_ids=(window_id,),
                        planner_version="local-entity-frame-plan.v3",
                    ))
                    request_meta.append({
                        "item_id": item_id,
                        "knowledge_id": knowledge_id,
                        "raw_entity_text": raw_entity_text,
                        "source_segment_indices": indices,
                        "transcript_segment_range": [max(0, indices[0] - 3), min(len(rows) - 1, indices[-1] + 3)],
                        "topic_indices": topic_indices,
                        "evidence_window_id": window_id,
                    })
                unresolved_coverage.append({
                    "knowledge_id": knowledge_id,
                    "item_id": item_id,
                    "raw_text": item.get("raw_text"),
                    "segment_indices": indices,
                    "status": "TARGETED_FRAME_REQUESTED",
                })
        if requests:
            extracted = FfmpegFrameExtractor().extract_targeted(
                args.media.resolve(strict=True),
                args.output.parent / "entity-frame-extraction",
                requests,
            )
            if len(extracted) != len(request_meta):
                raise ValueError("Targeted entity frame extraction count mismatch")
            meta_by_key = {
                (request.timestamp_ms, request.semantic_segment_ids[0], request.evidence_window_ids[0]): meta
                for request, meta in zip(requests, request_meta, strict=True)
            }
            for frame in extracted:
                meta_key = (
                    int(frame["timestamp_ms"]),
                    frame["semantic_segment_ids"][0],
                    frame["evidence_window_ids"][0],
                )
                meta = meta_by_key.get(meta_key)
                if meta is None:
                    raise ValueError(f"Missing request metadata for extracted frame: {meta_key}")
                source_path = Path(frame["image_path"]).resolve(strict=True)
                if sha256(source_path) != frame["image_hash"]:
                    raise ValueError(f"Targeted entity frame hash mismatch: {source_path}")
                frames.append({
                    "relative_path": source_path.relative_to(args.output.parent.resolve()).as_posix(),
                    "image_sha256": frame["image_hash"],
                    "timestamp_ms": int(frame["timestamp_ms"]),
                    "transcript_segment_range": meta["transcript_segment_range"],
                    "candidate_name": None,
                    "candidate_code": None,
                    "candidate_market": None,
                    "review_target_kind": "UNRESOLVED_ENTITY_WINDOW",
                    "candidate_source": "CURRENT_KNOWLEDGE_UNRESOLVED_ENTITY",
                    "knowledge_id": meta["knowledge_id"],
                    "ambiguity_item_id": meta["item_id"],
                    "raw_entity_text": meta["raw_entity_text"],
                    "source_segment_indices": meta["source_segment_indices"],
                    "topic_indices": meta["topic_indices"],
                    "evidence_window_id": meta["evidence_window_id"],
                    "planner_version": frame["planner_version"],
                    "extraction_reason": frame["extraction_reason"],
                })

    payload = {
        "schema_version": "local-equity-frame-request.v1",
        "lesson_id": legacy.get("lesson_id") or base_request.get("lesson_id"),
        "candidate_policy": (
            "PINNED_BASE_FRAMES_PLUS_LEGACY_EXACT_TOP_HEADER_PLUS_CURRENT_UNRESOLVED_"
            "ENTITY_WINDOWS_THEN_FRESH_PIXEL_REVIEW"
        ),
        "frames": frames,
        "skipped_candidates": skipped,
        "unresolved_entity_coverage": unresolved_coverage,
        "ambiguity_triage_sha256": sha256(args.ambiguity_triage) if args.ambiguity_triage else None,
        "audio_review_sha256": sha256(args.audio_review) if args.audio_review else None,
        "base_review_sha256": sha256(args.base_review) if args.base_review else None,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(frames), "skipped": len(skipped)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

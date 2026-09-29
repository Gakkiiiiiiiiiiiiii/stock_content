"""Build a hash-checked private frame index for one pinned local preview."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review_bytes = args.equity_review.read_bytes()
    review = json.loads(review_bytes)
    if review.get("audit", {}).get("passed") is not True:
        raise ValueError("Equity review has not passed")
    root = args.equity_review.parent.resolve()
    frames_by_id: dict[str, dict] = {}
    for mention in review.get("mentions") or []:
        for frame in mention.get("visual_evidence") or []:
            digest = frame.get("image_sha256")
            relative_path = frame.get("relative_path")
            if not isinstance(digest, str) or not isinstance(relative_path, str):
                raise ValueError("Reviewed frame identity is incomplete")
            path = (root / relative_path).resolve(strict=True)
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Reviewed frame path or hash mismatch")
            frame_id = frame.get("frame_id") or f"frame_{digest[:24]}"
            record = {
                "frame_id": frame_id,
                "image_hash": digest,
                "path": str(path),
                "timestamp_ms": frame["timestamp_ms"],
                "entity_ids": [mention["entity_id"]],
            }
            existing = frames_by_id.get(frame_id)
            if existing and (existing["image_hash"] != digest or existing["path"] != str(path)):
                raise ValueError("Frame ID maps to multiple identities")
            if existing:
                existing["entity_ids"] = sorted(set(existing["entity_ids"] + record["entity_ids"]))
            else:
                frames_by_id[frame_id] = record
    payload = {
        "schema_version": "local-frame-index.v2",
        "status": "LOCAL_REVIEW_ONLY_PRIVATE_PATH_INDEX",
        "equity_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "frames": sorted(frames_by_id.values(), key=lambda frame: (frame["timestamp_ms"], frame["frame_id"])),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(encoded)
    print(json.dumps({
        "status": "PASS_LOCAL_REVIEW_ONLY",
        "frames": len(payload["frames"]),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
    }))


if __name__ == "__main__":
    main()

"""Create a Simplified-Chinese display projection without changing source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opencc-package-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output.name}")
    sys.path.insert(0, str(args.opencc_package_dir))
    from opencc import OpenCC  # noqa: PLC0415 - local operator dependency only

    raw = args.topic_map.read_bytes()
    source = json.loads(raw)
    if source.get("status") != "LOCAL_REVIEW_ONLY_NOT_SEALED_OR_FRONTEND_PUBLISHED" or source.get(
        "segment_count"
    ) != len(source.get("segments", [])):
        raise ValueError("Topic map is not a valid local review source")
    converter = OpenCC("t2s")
    payload = {
        "schema_version": "local-topic-display-zh-hans.v1",
        "status": "LOCAL_REVIEW_ONLY",
        "source_topic_map_sha256": hashlib.sha256(raw).hexdigest(),
        "stages": [
            {
                "topic_index": index,
                "title": converter.convert(segment["topic"]),
                "subject": converter.convert(segment["subject"]) if segment.get("subject") else None,
            }
            for index, segment in enumerate(source["segments"])
        ],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage_count": len(payload["stages"]), "status": payload["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

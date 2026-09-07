"""Offline CLI for the version-pinned frozen-OOS knowledge quality gate."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permit direct invocation from a source checkout without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

def main(argv: list[str] | None = None) -> int:
    from stock_content.domain.knowledge_quality import KnowledgeQualityError, evaluate, load_json, load_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmarks/knowledge_golden/manifest.json")
    parser.add_argument("--candidate", help="Offline pipeline-result JSON")
    parser.add_argument("--baseline", help="Optional offline baseline-result JSON")
    parser.add_argument("--output", help="Write deterministic report JSON here; otherwise stdout")
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            load_manifest(args.manifest),
            load_json(args.candidate) if args.candidate else None,
            load_json(args.baseline) if args.baseline else None,
        )
    except (OSError, json.JSONDecodeError, KnowledgeQualityError) as error:
        report = {
            "contract": "knowledge-quality-report.v1",
            "schema_version": "1.0.0",
            "annotation_set_version": None,
            "annotation_set_hash": None,
            "partition": "frozen_oos",
            "status": "BLOCKED",
            "release_ready": False,
            "metrics": {},
            "strata": {},
            "blockers": ["INVALID_EVALUATOR_INPUT:" + str(error)],
        }
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report.get("release_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())

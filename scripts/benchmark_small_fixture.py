"""Run the fixed replay fixture and emit an auditable performance observation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class BenchmarkError(ValueError):
    """The deterministic fixture did not satisfy its release limits."""


def _max_rss_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    # Linux reports KiB, which is the Nightly runner platform for this gate.
    return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 3)


def _fixture_digest(root: Path, targets: list[str]) -> str:
    digest = hashlib.sha256()
    for target in targets:
        digest.update(target.encode("utf-8"))
        digest.update((root / target).read_bytes())
    return digest.hexdigest()


def run_benchmark(root: Path, limits: dict[str, Any]) -> dict[str, Any]:
    targets = limits.get("pytest_targets")
    if not isinstance(targets, list) or not all(isinstance(target, str) for target in targets):
        raise BenchmarkError("small fixture pytest targets are required")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".pytest-benchmark-", dir=root) as base_temp:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *targets, "-q", "--basetemp", base_temp],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    elapsed_seconds = round(time.perf_counter() - started, 3)
    report = {
        "schema_version": "small-fixture-benchmark.v1",
        "fixture_digest": _fixture_digest(root, targets),
        "pytest_targets": targets,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "max_rss_mib": _max_rss_mib(),
    }
    failures: list[str] = []
    if completed.returncode:
        failures.append("fixture_failed")
    if elapsed_seconds > limits["max_elapsed_seconds"]:
        failures.append("elapsed_seconds_exceeded")
    if report["max_rss_mib"] is None:
        failures.append("rss_metric_unavailable")
    elif report["max_rss_mib"] > limits["max_rss_mib"]:
        failures.append("max_rss_mib_exceeded")
    report["gate_result"] = "PASS" if not failures else "BLOCKED"
    report["blocking_reasons"] = failures
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-gates", default="config/quality-gates.json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        limits = json.loads(args.quality_gates.read_text(encoding="utf-8"))["small_fixture_benchmark"]
        report = run_benchmark(Path.cwd(), limits)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        if report["gate_result"] != "PASS":
            raise BenchmarkError(", ".join(report["blocking_reasons"]))
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

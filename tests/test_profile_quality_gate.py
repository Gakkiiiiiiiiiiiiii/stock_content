import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_resource_limits_match_audited_policy():
    verifier = _load_script("verify_profile_resources")
    verifier.validate_profile_resources(ROOT / "docker-compose.yml", ROOT / "config" / "quality-gates.json")


def test_profile_resource_limits_fail_closed_on_drift(tmp_path):
    verifier = _load_script("verify_profile_resources")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8").replace('memory: 1024M', 'memory: 2048M', 1)
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(compose, encoding="utf-8")
    with pytest.raises(verifier.ProfileResourceError, match="resource limits"):
        verifier.validate_profile_resources(compose_file, ROOT / "config" / "quality-gates.json")


def test_benchmark_fixture_is_fixed_and_blocks_resource_threshold_drift(monkeypatch):
    benchmark = _load_script("benchmark_small_fixture")

    class Completed:
        returncode = 0

    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: Completed())
    ticks = iter((1.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(benchmark, "_max_rss_mib", lambda: 2.0)
    limits = json.loads((ROOT / "config" / "quality-gates.json").read_text(encoding="utf-8"))["small_fixture_benchmark"]
    report = benchmark.run_benchmark(ROOT, limits)
    assert report["gate_result"] == "PASS"
    assert report["pytest_targets"] == ["tests/test_pipeline_integration.py", "tests/test_pipeline_replay.py"]

    report = benchmark.run_benchmark(ROOT, {**limits, "max_elapsed_seconds": 0})
    assert report["gate_result"] == "BLOCKED"
    assert "elapsed_seconds_exceeded" in report["blocking_reasons"]

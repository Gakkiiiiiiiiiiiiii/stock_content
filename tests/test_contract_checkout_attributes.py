"""Checkout policy regressions for manifest-referenced contract schemas."""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def test_manifest_schema_paths_are_forced_to_lf_on_checkout():
    root = Path(__file__).parents[1]
    manifest = yaml.safe_load((root / "contracts" / "platform-manifest.yaml").read_text(encoding="utf-8"))
    schema_paths = sorted({contract["schema"] for contract in manifest["contracts"]})

    result = subprocess.run(
        ["git", "check-attr", "eol", "--", *schema_paths],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.splitlines() == [f"{path}: eol: lf" for path in schema_paths]

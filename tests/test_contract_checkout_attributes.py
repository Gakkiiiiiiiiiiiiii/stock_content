"""Checkout policy regressions for manifest-referenced contract schemas."""
from __future__ import annotations

import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
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


@pytest.mark.parametrize(
    "contract_id",
    ["content-knowledge-bundle.v1", "content-ingestion.v1"],
)
def test_critical_contract_checkout_bytes_match_manifest_checksum(contract_id):
    """Locked contract schemas must retain their LF-byte digests after checkout."""
    root = Path(__file__).parents[1]
    manifest = yaml.safe_load((root / "contracts" / "platform-manifest.yaml").read_text(encoding="utf-8"))
    contract = next(contract for contract in manifest["contracts"] if contract["id"] == contract_id)
    schema_path = root / contract["schema"]
    expected_checksum = contract["checksum"].removeprefix("sha256:")

    assert sha256(schema_path.read_bytes()).hexdigest().upper() == expected_checksum

    result = subprocess.run(
        ["git", "-c", "core.autocrlf=true", "checkout-index", "--temp", "--", contract["schema"]],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    temporary_name, checkout_path = result.stdout.rstrip("\n").split("\t", maxsplit=1)
    temporary_schema = root / temporary_name
    try:
        assert checkout_path == contract["schema"]
        assert sha256(temporary_schema.read_bytes()).hexdigest().upper() == expected_checksum
    finally:
        temporary_schema.unlink(missing_ok=True)

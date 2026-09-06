"""Security regression tests for platform-manifest file references."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest
import yaml


def _contract_verifier():
    path = Path(__file__).parents[1] / "scripts" / "contracts" / "verify_manifest.py"
    spec = importlib.util.spec_from_file_location("verify_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(
    tmp_path: Path,
    *,
    schema_reference: str = "contracts/schema.json",
    fixture_reference: str | None = None,
) -> Path:
    contracts = tmp_path / "contracts"
    contracts.mkdir(exist_ok=True)
    schema = contracts / "schema.json"
    schema.write_bytes(b'{"type":"object"}\n')
    fixture = contracts / "fixtures" / "sample.json"
    fixture.parent.mkdir(exist_ok=True)
    fixture.write_bytes(b'{"fixture":true}\n')
    contract = {
        "id": "test.contract.v1",
        "schema": schema_reference,
        "checksum": "sha256:" + hashlib.sha256(schema.read_bytes()).hexdigest(),
        "producer": "stock_content",
        "consumers": ["stock_factor"],
        "compatibility": "compatibility",
        "deprecated": False,
        "sunset": None,
        "owner": "content-platform",
    }
    if fixture_reference is not None:
        contract["fixture"] = fixture_reference
    manifest = contracts / "platform-manifest.yaml"
    manifest.write_text(yaml.safe_dump({"contracts": [contract]}, sort_keys=False), encoding="utf-8")
    return manifest


def _link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"symlink creation unavailable: {error}")
        junction = subprocess.run(
            ["cmd", "/d", "/s", "/c", f'mklink /J "{link}" "{target}"'],
            check=False,
            capture_output=True,
            text=True,
        )
        if junction.returncode:
            pytest.skip(f"symlink/junction creation unavailable: {error}")


def test_manifest_verifier_accepts_contract_root_paths_and_raw_hashes(tmp_path):
    verifier = _contract_verifier()
    manifest = _write_manifest(tmp_path, fixture_reference="contracts/fixtures/sample.json")

    assert verifier.verify_manifest(manifest, today=date(2026, 1, 1)) == []


def test_manifest_verifier_keeps_raw_byte_checksum_validation(tmp_path):
    verifier = _contract_verifier()
    manifest = _write_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["contracts"][0]["checksum"] = "sha256:" + "0" * 64
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    errors = verifier.verify_manifest(manifest, today=date(2026, 1, 1))

    assert any("checksum mismatch" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "reference", "reason"),
    [
        ("schema", "/outside/schema.json", "relative"),
        ("schema", "C:\\outside\\schema.json", "relative"),
        ("schema", "contracts/../outside/schema.json", "traversal"),
        ("fixture", "../outside/fixture.json", "traversal"),
    ],
)
def test_manifest_verifier_rejects_absolute_and_traversal_paths(tmp_path, field, reference, reason):
    verifier = _contract_verifier()
    manifest = _write_manifest(tmp_path, **{f"{field}_reference": reference})

    errors = verifier.verify_manifest(manifest, today=date(2026, 1, 1))

    assert any(f"{field} path" in error and reason in error for error in errors)


def test_manifest_verifier_rejects_symlink_or_junction_escape(tmp_path):
    verifier = _contract_verifier()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "schema.json").write_bytes(b'{"outside":true}\n')
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    _link_or_skip(contracts / "escape", outside)
    manifest = _write_manifest(tmp_path, schema_reference="contracts/escape/schema.json")

    errors = verifier.verify_manifest(manifest, today=date(2026, 1, 1))

    assert any("schema path must resolve inside contracts/" in error for error in errors)


def test_manifest_verifier_rejects_manifest_symlink_or_junction_escape(tmp_path):
    verifier = _contract_verifier()
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "platform-manifest.yaml").write_text("[not a manifest]", encoding="utf-8")
    _link_or_skip(contracts / "escape", outside)

    escaped_manifest = contracts / "escape" / "platform-manifest.yaml"
    with pytest.raises(verifier.ManifestError, match="manifest path must resolve inside contracts/"):
        verifier.load_manifest(escaped_manifest)
    errors = verifier.verify_manifest(escaped_manifest, today=date(2026, 1, 1))

    assert errors == ["manifest path must resolve inside contracts/"]


def test_manifest_verifier_rejects_contracts_prefix_collision(tmp_path):
    verifier = _contract_verifier()
    collision = tmp_path / "contracts_evil"
    collision.mkdir()
    (collision / "schema.json").write_bytes(b'{"outside":true}\n')
    manifest = _write_manifest(tmp_path, schema_reference="contracts_evil/schema.json")

    errors = verifier.verify_manifest(manifest, today=date(2026, 1, 1))

    assert any("schema path must be rooted at contracts/" in error for error in errors)

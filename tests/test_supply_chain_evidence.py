import importlib.util
import shutil
from pathlib import Path

import pytest


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sbom_verifier_accepts_each_locked_profile_and_rejects_dependency_tampering(monkeypatch):
    generator = _load_script("generate_sbom")
    monkeypatch.setitem(__import__("sys").modules, "generate_sbom", generator)
    verifier = _load_script("verify_sbom")
    project = Path(__file__).parents[1] / "pyproject.toml"
    core_manifest = None
    for profile in ("core", "media", "multimodal"):
        manifest = generator.build_manifest(project, profile=profile)
        verifier.verify_manifest(manifest, project, profile=profile)
        if profile == "core":
            core_manifest = manifest

    assert core_manifest is not None
    tampered = {
        **core_manifest,
        "components": [*core_manifest["components"], {"type": "library", "requirement": "evil"}],
    }
    with pytest.raises(verifier.SbomVerificationError, match="components"):
        verifier.verify_manifest(tampered, project, profile="core")


def test_sbom_verifier_rejects_lock_drift(monkeypatch, tmp_path):
    generator = _load_script("generate_sbom")
    monkeypatch.setitem(__import__("sys").modules, "generate_sbom", generator)
    verifier = _load_script("verify_sbom")
    root = Path(__file__).parents[1]
    project = tmp_path / "pyproject.toml"
    shutil.copy(root / "pyproject.toml", project)
    shutil.copytree(root / "locks", tmp_path / "locks")
    manifest = generator.build_manifest(project, profile="core")
    (tmp_path / "locks" / "core.lock").write_text("-e .[core]\n", encoding="utf-8")

    with pytest.raises(verifier.SbomVerificationError, match="lock_sha256"):
        verifier.verify_manifest(manifest, project, profile="core")

"""Verify that a generated SBOM exactly describes a selected dependency profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_sbom import build_manifest


class SbomVerificationError(ValueError):
    """The supplied SBOM is not reproducible from the declared project profile."""


def verify_manifest(
    manifest: dict[str, Any], project_file: str | Path = "pyproject.toml", *, profile: str = "core"
) -> None:
    expected = build_manifest(project_file, profile=profile)
    for key in ("bomFormat", "specVersion", "profile", "project", "version", "lock_sha256", "profile_extras"):
        if manifest.get(key) != expected[key]:
            raise SbomVerificationError(f"SBOM {key} does not match the declared project profile")
    actual_components = manifest.get("components")
    if not isinstance(actual_components, list):
        raise SbomVerificationError("SBOM components must be a list")
    if actual_components != expected["components"]:
        raise SbomVerificationError("SBOM components do not exactly match declared dependencies")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sbom", type=Path)
    parser.add_argument("--project", default="pyproject.toml")
    parser.add_argument("--profile", default="core")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.sbom.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise SbomVerificationError("SBOM document must be an object")
        verify_manifest(manifest, args.project, profile=args.profile)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

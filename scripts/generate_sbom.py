"""Generate a small SPDX-like dependency manifest without shell dependencies."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


def _read_project(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle).get("project", {})


def build_manifest(project_file: str | Path = "pyproject.toml", *, profile: str = "core") -> dict[str, Any]:
    project_path = Path(project_file)
    project = _read_project(project_path)
    optional = project.get("optional-dependencies", {})
    dependencies = list(project.get("dependencies", []))
    if profile in optional:
        dependencies.extend(optional[profile])
    return {
        "bomFormat": "stock-content-sbom",
        "specVersion": "1.0",
        "profile": profile,
        "project": project.get("name", "unknown"),
        "version": project.get("version", "unknown"),
        "components": [{"type": "library", "requirement": value} for value in sorted(dependencies)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="pyproject.toml")
    parser.add_argument("--profile", default="core")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = build_manifest(args.project, profile=args.profile)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a small SPDX-like dependency manifest without shell dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

SUPPORTED_PROFILES = frozenset({"core", "media", "multimodal"})


def _read_project(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle).get("project", {})


def _profile_lock(project_path: Path, profile: str) -> Path:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported dependency profile: {profile}")
    lock_path = project_path.parent / "locks" / f"{profile}.lock"
    if not lock_path.is_file():
        raise ValueError(f"missing dependency lock for profile {profile}")
    return lock_path


def profile_extras(project_file: str | Path = "pyproject.toml", *, profile: str = "core") -> tuple[str, ...]:
    """Return the extras selected by the reviewed profile lock."""
    lock_path = _profile_lock(Path(project_file), profile)
    matches = re.findall(r"^-e\s+\.\[([^\]]+)\]\s*$", lock_path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"profile lock {lock_path} must declare exactly one editable extra set")
    extras = tuple(extra.strip() for extra in matches[0].split(",") if extra.strip())
    if not extras:
        raise ValueError(f"profile lock {lock_path} has no selected extras")
    return extras


def build_manifest(project_file: str | Path = "pyproject.toml", *, profile: str = "core") -> dict[str, Any]:
    project_path = Path(project_file)
    project = _read_project(project_path)
    optional = project.get("optional-dependencies", {})
    dependencies = list(project.get("dependencies", []))
    extras = profile_extras(project_path, profile=profile)
    for extra in extras:
        if extra not in optional:
            raise ValueError(f"profile lock selects unknown project extra: {extra}")
        dependencies.extend(optional[extra])
    lock_path = _profile_lock(project_path, profile)
    return {
        "bomFormat": "stock-content-sbom",
        "specVersion": "1.0",
        "profile": profile,
        "project": project.get("name", "unknown"),
        "version": project.get("version", "unknown"),
        "profile_extras": list(extras),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
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

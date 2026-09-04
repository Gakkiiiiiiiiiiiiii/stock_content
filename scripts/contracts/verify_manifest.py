"""Validate the platform contract inventory and schema checksums."""
from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("contracts"), list):
        raise ManifestError("manifest must contain a contracts list")
    return payload


def verify_manifest(path: str | Path = "contracts/platform-manifest.yaml", *, today: date | None = None) -> list[str]:
    manifest_path = Path(path)
    payload = load_manifest(manifest_path)
    root = manifest_path.parent.parent if manifest_path.parent.name == "contracts" else manifest_path.parent
    errors: list[str] = []
    seen: set[str] = set()
    now = today or date.today()
    for index, item in enumerate(payload["contracts"]):
        prefix = f"contracts[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        required = (
            "id", "schema", "checksum", "producer", "consumers", "compatibility",
            "deprecated", "sunset", "owner",
        )
        missing = [key for key in required if key not in item]
        if missing:
            errors.append(f"{prefix} missing: {', '.join(missing)}")
            continue
        identifier = str(item["id"])
        if identifier in seen:
            errors.append(f"duplicate contract id: {identifier}")
        seen.add(identifier)
        if not str(item["producer"]).strip() or not str(item["owner"]).strip():
            errors.append(f"{prefix} producer and owner are required")
        if not isinstance(item["consumers"], list) or not item["consumers"]:
            errors.append(f"{prefix} consumers must be a non-empty list")
        schema = root / str(item["schema"])
        # A schema path is repository-relative in the manifest.
        if not schema.exists():
            schema = manifest_path.parent.parent / str(item["schema"])
        if not schema.is_file():
            errors.append(f"{prefix} schema does not exist: {item['schema']}")
        checksum = str(item["checksum"])
        if not checksum.startswith("sha256:") or len(checksum) != 71:
            errors.append(f"{prefix} checksum must be sha256:<64 hex characters>")
        elif schema.is_file() and checksum[7:].lower() != _sha256(schema):
            errors.append(f"{prefix} checksum mismatch: {identifier}")
        sunset = item["sunset"]
        if sunset is not None:
            try:
                sunset_date = date.fromisoformat(str(sunset))
                if sunset_date <= now and not bool(item["deprecated"]):
                    errors.append(f"{prefix} active contract sunset must be in the future")
            except ValueError:
                errors.append(f"{prefix} sunset must be ISO date or null")
        if identifier.endswith("v5.1"):
            if item.get("compatibility") != "formal" or item.get("formal_expected") is not True:
                errors.append("content-factor-signal.v5.1 must be formal and formal_expected=true")
        elif item.get("compatibility") == "formal":
            errors.append(f"{prefix} only content-factor-signal.v5.1 may be formal")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="contracts/platform-manifest.yaml")
    args = parser.parse_args(argv)
    errors = verify_manifest(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("platform contract manifest: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

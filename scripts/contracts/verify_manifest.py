"""Validate the platform contract inventory and schema checksums."""
from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


class ManifestError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("contracts"), list):
        raise ManifestError("manifest must contain a contracts list")
    return payload


def _contract_root(manifest_path: Path) -> tuple[Path, Path]:
    """Return the declared and resolved repository ``contracts`` roots.

    Manifest entries are repository-relative, rather than relative to an
    arbitrary manifest location.  Keep the lexical root while resolving an
    entry so a symlink or Windows junction cannot make a path appear safe.
    """
    contract_root = next(
        (parent for parent in (manifest_path.parent, *manifest_path.parents) if parent.name == "contracts"),
        None,
    )
    if contract_root is None:
        raise ManifestError("manifest must be located under a contracts directory")
    repository_root = contract_root.parent
    resolved_contract_root = contract_root.resolve()
    if not resolved_contract_root.is_relative_to(repository_root.resolve()):
        raise ManifestError("contracts root must resolve inside its repository")
    return repository_root, resolved_contract_root


def _resolve_contract_path(
    reference: Any,
    *,
    repository_root: Path,
    contracts_root: Path,
    field: str,
) -> Path:
    """Resolve one manifest path, rejecting every route outside contracts/."""
    if not isinstance(reference, str) or not reference.strip():
        raise ManifestError(f"{field} path must be a non-empty string")

    # Check both path grammars so a manifest remains fail-closed when it is
    # authored on a different platform from the verifier.
    windows_path = PureWindowsPath(reference)
    posix_path = PurePosixPath(reference)
    if windows_path.is_absolute() or posix_path.is_absolute():
        raise ManifestError(f"{field} path must be relative to contracts/")
    parts = tuple(part for part in reference.replace("\\", "/").split("/") if part)
    if ".." in parts:
        raise ManifestError(f"{field} path must not contain traversal")
    if not parts or parts[0] != "contracts":
        raise ManifestError(f"{field} path must be rooted at contracts/")

    try:
        resolved = (repository_root / Path(reference)).resolve()
    except OSError as error:
        raise ManifestError(f"{field} path could not be resolved") from error
    if not resolved.is_relative_to(contracts_root):
        raise ManifestError(f"{field} path must resolve inside contracts/")
    return resolved


def _resolve_manifest_path(path: str | Path) -> tuple[Path, Path, Path]:
    """Resolve the manifest through the same containment gate as its entries."""
    manifest_path = Path(path)
    repository_root, contracts_root = _contract_root(manifest_path)
    try:
        reference = (
            manifest_path.relative_to(repository_root)
            if manifest_path.is_absolute()
            else manifest_path
        )
    except ValueError as error:
        raise ManifestError("manifest path must be located under contracts/") from error
    resolved = _resolve_contract_path(
        str(reference),
        repository_root=repository_root,
        contracts_root=contracts_root,
        field="manifest",
    )
    if not resolved.is_file():
        raise ManifestError(f"manifest does not exist: {path}")
    return resolved, repository_root, contracts_root


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path, _, _ = _resolve_manifest_path(path)
    return _read_manifest(manifest_path)


def verify_manifest(path: str | Path = "contracts/platform-manifest.yaml", *, today: date | None = None) -> list[str]:
    try:
        manifest_path, repository_root, contracts_root = _resolve_manifest_path(path)
        payload = _read_manifest(manifest_path)
    except ManifestError as error:
        return [str(error)]
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
        try:
            schema = _resolve_contract_path(
                item["schema"],
                repository_root=repository_root,
                contracts_root=contracts_root,
                field=f"{prefix} schema",
            )
        except ManifestError as error:
            errors.append(str(error))
            schema = None
        if schema is not None and not schema.is_file():
            errors.append(f"{prefix} schema does not exist: {item['schema']}")
        checksum = str(item["checksum"])
        if not checksum.startswith("sha256:") or len(checksum) != 71:
            errors.append(f"{prefix} checksum must be sha256:<64 hex characters>")
        elif schema is not None and schema.is_file() and checksum[7:].lower() != _sha256(schema):
            errors.append(f"{prefix} checksum mismatch: {identifier}")
        if "fixture" in item:
            try:
                fixture = _resolve_contract_path(
                    item["fixture"],
                    repository_root=repository_root,
                    contracts_root=contracts_root,
                    field=f"{prefix} fixture",
                )
            except ManifestError as error:
                errors.append(str(error))
            else:
                if not fixture.is_file():
                    errors.append(f"{prefix} fixture does not exist: {item['fixture']}")
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

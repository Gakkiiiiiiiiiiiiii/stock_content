"""Fail closed when worker profile resource limits drift from quality-gate policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PROFILE_SERVICES = {
    "core": "stock-content-core-worker",
    "media": "media-worker",
    "multimodal": "multimodal-worker",
}


class ProfileResourceError(ValueError):
    """The compose configuration cannot prove a worker resource boundary."""


def validate_profile_resources(compose_file: str | Path, quality_gates_file: str | Path) -> None:
    compose = yaml.safe_load(Path(compose_file).read_text(encoding="utf-8"))
    gates = json.loads(Path(quality_gates_file).read_text(encoding="utf-8"))
    if gates.get("schema_version") != "quality-gates.v1":
        raise ProfileResourceError("unsupported quality-gates schema")
    services = compose.get("services") if isinstance(compose, dict) else None
    limits_by_profile = gates.get("worker_resource_limits") if isinstance(gates, dict) else None
    if not isinstance(services, dict) or not isinstance(limits_by_profile, dict):
        raise ProfileResourceError("compose services and worker resource limits are required")

    for profile, service_name in PROFILE_SERVICES.items():
        service = services.get(service_name)
        expected = limits_by_profile.get(profile)
        if not isinstance(service, dict) or not isinstance(expected, dict):
            raise ProfileResourceError(f"missing {profile} worker resource policy")
        environment = service.get("environment", {})
        if environment.get("CONTENT_WORKER_PROFILE") != profile:
            raise ProfileResourceError(f"{service_name} must declare CONTENT_WORKER_PROFILE={profile}")
        actual = service.get("deploy", {}).get("resources", {}).get("limits", {})
        if actual != expected:
            raise ProfileResourceError(f"{service_name} resource limits do not match the audited {profile} policy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", default="docker-compose.yml", type=Path)
    parser.add_argument("--quality-gates", default="config/quality-gates.json", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_profile_resources(args.compose, args.quality_gates)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

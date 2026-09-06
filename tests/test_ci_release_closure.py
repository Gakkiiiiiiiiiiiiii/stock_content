"""Guard the release-blocking CI closure against hand-maintained test lists."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _workflow() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))


def _run(step: dict) -> str:
    return str(step.get("run", ""))


def test_release_gate_requires_the_complete_same_sha_release_closure() -> None:
    jobs = _workflow()["jobs"]

    assert set(jobs["quality-gate"]["needs"]) == {"deterministic-closure", "postgres-closure"}
    assert set(jobs["release-gate"]["needs"]) == {
        "lint",
        "contract",
        "deterministic-closure",
        "postgres-closure",
        "quality-gate",
        "golden-content",
        "docker-build",
        "profile-smoke",
        "integration",
    }
    assert jobs["release-gate"]["if"] == "always()"
    assert jobs["release-gate"]["env"]["RELEASE_SHA"] == "${{ github.sha }}"
    closure_step = next(
        step
        for step in jobs["release-gate"]["steps"]
        if step.get("name") == "Require every release prerequisite to succeed"
    )
    closure_command = _run(closure_step)
    for job_name in jobs["release-gate"]["needs"]:
        assert f"needs.{job_name}.result" in closure_command
    assert closure_command.count('= "success"') == len(jobs["release-gate"]["needs"])
    release_artifact_download = next(
        step for step in jobs["release-gate"]["steps"] if step.get("uses") == "actions/download-artifact@v4"
    )
    assert release_artifact_download["with"]["pattern"] == "content-*-${{ github.sha }}"


def test_ci_selects_full_test_roots_and_runs_postgres_tests_against_a_real_service() -> None:
    jobs = _workflow()["jobs"]
    deterministic_steps = jobs["deterministic-closure"]["steps"]
    postgres_steps = jobs["postgres-closure"]["steps"]

    assert any("pytest tests -q --ignore=tests/postgres" in _run(step) for step in deterministic_steps)
    assert any("pytest tests/postgres -q" in _run(step) for step in postgres_steps)
    assert jobs["postgres-closure"]["services"]["postgres"]["image"] == "postgres:16-alpine"
    assert jobs["postgres-closure"]["env"]["CONTENT_TEST_POSTGRES_URL"].startswith(
        "postgresql+psycopg://"
    )


def test_release_evidence_artifacts_are_named_for_the_same_checkout_sha() -> None:
    jobs = _workflow()["jobs"]
    for job_name, prefix in (
        ("contract", "content-contract-"),
        ("deterministic-closure", "content-deterministic-"),
        ("postgres-closure", "content-postgres-"),
        ("quality-gate", "content-quality-"),
        ("golden-content", "content-golden-"),
        ("release-gate", "content-release-"),
    ):
        artifact_steps = [step for step in jobs[job_name]["steps"] if step.get("uses") == "actions/upload-artifact@v4"]
        assert any(step["with"]["name"] == f"{prefix}${{{{ github.sha }}}}" for step in artifact_steps)

    integration_artifacts = [
        step["with"]["name"]
        for step in jobs["integration"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    ]
    assert integration_artifacts == [
        "content-integration-logs-${{ github.sha }}",
        "content-integration-rebuild-${{ github.sha }}",
        "content-integration-tests-${{ github.sha }}",
    ]


def test_contract_manifest_and_bounded_docker_profile_jobs_are_release_blocking() -> None:
    jobs = _workflow()["jobs"]

    contract_steps = jobs["contract"]["steps"]
    assert any("scripts/contracts/verify_manifest.py" in _run(step) for step in contract_steps)
    assert any("test_artifact_contracts.py" in _run(step) for step in contract_steps)

    assert set(jobs["integration"]["needs"]) == {
        "docker-build",
        "profile-smoke",
        "deterministic-closure",
        "postgres-closure",
    }
    assert "if" not in jobs["integration"]
    assert any("docker compose --profile" in _run(step) for step in jobs["profile-smoke"]["steps"])
    assert any("verify_profile_resources.py" in _run(step) for step in jobs["profile-smoke"]["steps"])


def test_docker_build_evidence_covers_every_shipped_workload_image() -> None:
    jobs = _workflow()["jobs"]
    docker_build = jobs["docker-build"]
    matrix = docker_build["strategy"]["matrix"]["include"]
    expected_images = {
        "api": ("docker/Dockerfile.api", "stock-content-api:${{ github.sha }}"),
        "core-worker": ("docker/Dockerfile.core-worker", "stock-content-core-worker:${{ github.sha }}"),
        "media-worker": ("docker/Dockerfile.media", "stock-content-media-worker:${{ github.sha }}"),
        "multimodal-worker": (
            "docker/Dockerfile.multimodal",
            "stock-content-multimodal-worker:${{ github.sha }}",
        ),
    }

    assert {entry["target"] for entry in matrix} == set(expected_images)
    for entry in matrix:
        dockerfile, tag = expected_images[entry["target"]]
        assert entry == {
            "target": entry["target"],
            "context": ".",
            "dockerfile": dockerfile,
            "tag": tag,
        }

    # Profile rendering validates the same three capability profiles, but only
    # docker-build supplies release evidence that their images were built.
    assert set(jobs["profile-smoke"]["strategy"]["matrix"]["profile"]) == {
        "core",
        "media",
        "multimodal",
    }

    build_step = next(step for step in docker_build["steps"] if step.get("uses") == "docker/build-push-action@v6")
    assert build_step["with"] == {
        "context": "${{ matrix.context }}",
        "file": "${{ matrix.dockerfile }}",
        "push": False,
        "load": True,
        "tags": "${{ matrix.tag }}",
    }
    image_evidence = next(
        step for step in docker_build["steps"] if step.get("name") == "Record tagged workload image evidence"
    )
    assert 'docker image inspect "${{ matrix.tag }}"' in _run(image_evidence)
    artifact_step = next(
        step
        for step in docker_build["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert artifact_step["with"]["name"] == "content-image-${{ matrix.target }}-${{ github.sha }}"

    release_evidence_step = next(
        step for step in jobs["release-gate"]["steps"] if step.get("name") == "Bind release evidence to this checkout"
    )
    release_evidence_command = _run(release_evidence_step)
    assert "for workload in api core-worker media-worker multimodal-worker; do" in release_evidence_command
    assert "image-$workload.json" in release_evidence_command
    assert "stock-content-$workload:$RELEASE_SHA" in release_evidence_command

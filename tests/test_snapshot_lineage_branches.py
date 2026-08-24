from __future__ import annotations

from fastapi.testclient import TestClient

from stock_content.api.main import create_app
from stock_content.application.service import ContentApplication
from stock_content.domain.artifacts import SourceArtifact
from stock_content.domain.lineage import ContentSnapshot


class _SnapshotLookup:
    def __init__(self, *snapshots: ContentSnapshot) -> None:
        self._snapshots = {snapshot.content_snapshot_id: snapshot for snapshot in snapshots}

    def get(self, content_snapshot_id: str) -> ContentSnapshot | None:
        return self._snapshots.get(content_snapshot_id)


class _ArtifactLookup:
    def __init__(self, *artifacts: SourceArtifact) -> None:
        self._artifacts = {artifact.artifact_id: artifact for artifact in artifacts}

    def get(self, artifact_id: str) -> SourceArtifact | None:
        return self._artifacts.get(artifact_id)


def _snapshot(
    content_snapshot_id: str,
    *,
    parent_snapshot_id: str | None = None,
    supersedes_snapshot_id: str | None = None,
) -> ContentSnapshot:
    return ContentSnapshot(
        content_snapshot_id=content_snapshot_id,
        source_type="fixture",
        source_ref="lineage",
        source_content_hash="raw",
        snapshot_kind="REFRESH" if parent_snapshot_id or supersedes_snapshot_id else "INITIAL",
        parent_snapshot_id=parent_snapshot_id,
        supersedes_snapshot_id=supersedes_snapshot_id,
    )


def _application(*snapshots: ContentSnapshot) -> ContentApplication:
    application = object.__new__(ContentApplication)
    application._snapshots = _SnapshotLookup(*snapshots)  # noqa: SLF001 - focused lineage fixture
    application._artifact_repository = None  # noqa: SLF001 - no artifact graph in this test
    return application


def _artifact(artifact_id: str, *parent_artifact_ids: str) -> SourceArtifact:
    return SourceArtifact(
        artifact_id=artifact_id,
        artifact_type="source",
        source_type="fixture",
        source_ref=artifact_id,
        raw_content_hash=artifact_id,
        parent_artifact_ids=parent_artifact_ids,
    )


def _application_with_artifacts(
    snapshots: tuple[ContentSnapshot, ...], artifacts: tuple[SourceArtifact, ...]
) -> ContentApplication:
    application = _application(*snapshots)
    application._artifact_repository = _ArtifactLookup(*artifacts)  # noqa: SLF001 - focused fixture
    return application


def test_snapshot_lineage_walks_both_parent_edges_in_stable_order() -> None:
    parent = _snapshot("parent")
    superseded = _snapshot("superseded")
    root = _snapshot(
        "root",
        parent_snapshot_id=parent.content_snapshot_id,
        supersedes_snapshot_id=superseded.content_snapshot_id,
    )

    payload = _application(parent, superseded, root).get_snapshot_lineage(root.content_snapshot_id)

    assert payload is not None
    assert payload["lineage_complete"] is True
    assert payload["lineage_errors"] == []
    assert [item["content_snapshot_id"] for item in payload["snapshot_lineage"]["parents"]] == [
        "parent",
        "superseded",
    ]


def test_snapshot_lineage_deduplicates_same_parent_in_both_fields() -> None:
    parent = _snapshot("parent")
    root = _snapshot(
        "root",
        parent_snapshot_id=parent.content_snapshot_id,
        supersedes_snapshot_id=parent.content_snapshot_id,
    )

    payload = _application(parent, root).get_snapshot_lineage(root.content_snapshot_id)

    assert payload is not None
    assert payload["lineage_complete"] is True
    assert [item["content_snapshot_id"] for item in payload["snapshot_lineage"]["parents"]] == ["parent"]


def test_snapshot_lineage_missing_parent_fails_closed_without_partial_graph() -> None:
    root = _snapshot("root", parent_snapshot_id="missing")

    payload = _application(root).get_snapshot_lineage(root.content_snapshot_id)

    assert payload is not None
    assert payload["lineage_complete"] is False
    assert payload["snapshot_lineage"] is None
    assert "missing" in payload["lineage_errors"][0]


def test_snapshot_lineage_cycle_fails_closed_without_partial_graph() -> None:
    first = _snapshot("first", parent_snapshot_id="second")
    second = _snapshot("second", parent_snapshot_id="first")

    payload = _application(first, second).get_snapshot_lineage(first.content_snapshot_id)

    assert payload is not None
    assert payload["lineage_complete"] is False
    assert payload["snapshot_lineage"] is None
    assert "cycle" in payload["lineage_errors"][0]


def test_snapshot_lineage_artifact_dag_is_complete_with_shared_ancestor() -> None:
    shared = _artifact("shared")
    left = _artifact("left", "shared")
    right = _artifact("right", "shared")
    root = _snapshot("root")
    root = ContentSnapshot(**{**root.__dict__, "artifact_ids": {"left": "left", "right": "right"}})

    payload = _application_with_artifacts((root,), (shared, left, right)).get_snapshot_lineage(
        root.content_snapshot_id
    )

    assert payload is not None
    assert payload["lineage_complete"] is True
    assert [item["slot"] for item in payload["artifacts"]] == ["left", "right"]
    assert payload["lineage_errors"] == []


def test_snapshot_lineage_missing_artifact_root_fails_closed() -> None:
    root = ContentSnapshot(**{**_snapshot("root").__dict__, "artifact_ids": {"source": "missing-root"}})

    payload = _application_with_artifacts((root,), ()).get_snapshot_lineage(root.content_snapshot_id)

    assert payload is not None
    assert payload["lineage_complete"] is False
    assert payload["artifacts"] == []
    assert payload["snapshot_lineage"] is None
    assert "missing-root" in payload["lineage_errors"][0]


def test_snapshot_lineage_missing_artifact_parent_fails_closed() -> None:
    root_artifact = _artifact("root-artifact", "missing-parent")
    root = ContentSnapshot(**{**_snapshot("root").__dict__, "artifact_ids": {"source": "root-artifact"}})

    payload = _application_with_artifacts((root,), (root_artifact,)).get_snapshot_lineage(
        root.content_snapshot_id
    )

    assert payload is not None
    assert payload["lineage_complete"] is False
    assert payload["artifacts"] == []
    assert "missing-parent" in payload["lineage_errors"][0]


def test_snapshot_lineage_artifact_cycle_fails_closed() -> None:
    first = _artifact("artifact-a", "artifact-b")
    second = _artifact("artifact-b", "artifact-a")
    root = ContentSnapshot(**{**_snapshot("root").__dict__, "artifact_ids": {"source": "artifact-a"}})

    payload = _application_with_artifacts((root,), (first, second)).get_snapshot_lineage(
        root.content_snapshot_id
    )

    assert payload is not None
    assert payload["lineage_complete"] is False
    assert payload["artifacts"] == []
    assert payload["snapshot_lineage"] is None
    assert "cycle" in payload["lineage_errors"][0]


def test_snapshot_lineage_artifact_repository_unavailable_fails_closed() -> None:
    root = ContentSnapshot(**{**_snapshot("root").__dict__, "artifact_ids": {"source": "root-artifact"}})

    payload = _application(root).get_snapshot_lineage(root.content_snapshot_id)

    assert payload is not None
    assert payload["lineage_complete"] is False
    assert payload["artifacts"] == []
    assert payload["snapshot_lineage"] is None
    assert "repository unavailable" in payload["lineage_errors"][0]


def test_snapshot_lineage_api_exposes_both_branches_and_completeness() -> None:
    parent = _snapshot("parent")
    superseded = _snapshot("superseded")
    root = _snapshot(
        "root",
        parent_snapshot_id=parent.content_snapshot_id,
        supersedes_snapshot_id=superseded.content_snapshot_id,
    )
    client = TestClient(create_app(_application(parent, superseded, root)))

    response = client.get(f"/api/v1/content-snapshots/{root.content_snapshot_id}/lineage")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["lineage_complete"] is True
    assert [item["content_snapshot_id"] for item in data["snapshot_lineage"]["parents"]] == [
        "parent",
        "superseded",
    ]


def test_snapshot_lineage_api_fails_closed_for_incomplete_artifact_graph() -> None:
    artifact = _artifact("root-artifact", "missing-parent")
    root = ContentSnapshot(**{**_snapshot("root").__dict__, "artifact_ids": {"source": "root-artifact"}})
    client = TestClient(create_app(_application_with_artifacts((root,), (artifact,))))

    response = client.get(f"/api/v1/content-snapshots/{root.content_snapshot_id}/lineage")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["lineage_complete"] is False
    assert data["artifacts"] == []
    assert data["snapshot_lineage"] is None

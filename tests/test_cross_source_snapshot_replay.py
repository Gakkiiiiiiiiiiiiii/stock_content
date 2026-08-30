from types import MethodType

from stock_content.api.dependencies import build_application


def test_two_real_sources_publish_source_scoped_snapshots_and_replay_a(tmp_path, monkeypatch):
    """A/B are real application runs; replaying A cannot absorb B's rows."""
    application = build_application(
        f"sqlite:///{tmp_path / 'cross-source-snapshots.db'}", enable_qdrant=False
    )
    knowledge_stage = next(
        item._stage for item in application._pipeline._stages if item.name == "knowledge"
    )
    original_fixture_records = knowledge_stage._fixture_records

    # Keep the authoritative semantic claim shared while allowing the
    # compatibility knowledge projection to coexist under its global UID.
    def source_local_projection(self, context, timestamp):
        records = original_fixture_records(context, timestamp)
        if context.source["ref"] == "BV-source-b":
            return [
                {**record, "knowledge_uid": f"{record['knowledge_uid']}-source-b"}
                for record in records
            ]
        return records

    monkeypatch.setattr(
        knowledge_stage,
        "_fixture_records",
        MethodType(source_local_projection, knowledge_stage),
    )
    options = {
        "metadata": {"title": "same canonical proposition"},
        "transcript": "股票600000营收增长10%。",
        "offline_fixture": True,
        "as_of": "2026-01-01T00:00:00+00:00",
    }
    application.enqueue("bilibili", "BV-source-a", options)
    first_result = application.process_next("source-a-worker")
    assert first_result["status"] == "SUCCEEDED"
    application.enqueue("bilibili", "BV-source-b", options)
    second_result = application.process_next("source-b-worker")
    assert second_result["status"] == "SUCCEEDED"

    snapshot_a = application._snapshots.get(first_result["content_snapshot_id"])
    snapshot_b = application._snapshots.get(second_result["content_snapshot_id"])
    assert snapshot_a is not None and snapshot_b is not None
    assert snapshot_a.content_snapshot_id != snapshot_b.content_snapshot_id
    artifact_repository = application._artifact_repository
    claims_a = artifact_repository.get(snapshot_a.artifact_ids["claims"]).claims
    claims_b = artifact_repository.get(snapshot_b.artifact_ids["claims"]).claims
    assert claims_a == claims_b

    evidence_a = artifact_repository.get(snapshot_a.artifact_ids["evidence"])
    evidence_b = artifact_repository.get(snapshot_b.artifact_ids["evidence"])
    assert evidence_a is not None and evidence_b is not None
    evidence_ids_a = {item.evidence_id for item in evidence_a.evidences}
    evidence_ids_b = {item.evidence_id for item in evidence_b.evidences}
    assert evidence_ids_a and evidence_ids_b and evidence_ids_a.isdisjoint(evidence_ids_b)

    occurrences_a = artifact_repository.get(snapshot_a.artifact_ids["occurrences"])
    occurrences_b = artifact_repository.get(snapshot_b.artifact_ids["occurrences"])
    assert occurrences_a is not None and occurrences_b is not None
    occurrence_ids_a = set(occurrences_a.occurrence_ids)
    occurrence_ids_b = set(occurrences_b.occurrence_ids)
    assert occurrence_ids_a and occurrence_ids_b and occurrence_ids_a.isdisjoint(occurrence_ids_b)
    for occurrence_id in occurrence_ids_a:
        occurrence = application._occurrence_repository.get(occurrence_id)
        assert occurrence is not None
        assert occurrence.claim_id == claims_a[0]
        assert occurrence.source_artifact_id == snapshot_a.source_artifact_id
        assert set(occurrence.evidence_refs) <= evidence_ids_a
    for occurrence_id in occurrence_ids_b:
        occurrence = application._occurrence_repository.get(occurrence_id)
        assert occurrence is not None
        assert occurrence.claim_id == claims_b[0]
        assert occurrence.source_artifact_id == snapshot_b.source_artifact_id
        assert set(occurrence.evidence_refs) <= evidence_ids_b

    replay = application.replay_content_snapshot(snapshot_a.content_snapshot_id)
    assert replay["identity_match"] is True

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

from stock_content.adapters.postgres.database import Database
from stock_content.adapters.postgres.repositories.knowledge_bundle_repository import PostgresKnowledgeBundleRepository
from stock_content.api.dependencies import build_application
from stock_content.api.knowledge_bundles import create_knowledge_bundles_router
from stock_content.application.knowledge_bundle_service import BundleProducerMetadata, KnowledgeBundleService
from stock_content.domain.knowledge_bundle import (
    V2_CONTRACT,
    CanonicalizationError,
    KnowledgeBundleRequest,
    canonical_json,
)
from stock_content.domain.temporal_normalizer import TemporalNormalizer
from stock_content.ports.knowledge_bundle_repository import InMemoryKnowledgeBundleRepository

NOW = datetime(2026, 9, 6, tzinfo=UTC)
CHECKSUM = "sha256:EBFD13B78622C3846890438A4FB3CB858278F571FDAB247CDD72EF18CA211621"
V2_CHECKSUM = "sha256:23C1D9C6BE131CBA8F270F01F7F45EB5D3148EE219EDF43D689F1C5707115800"


def _request(**changes):
    values = {
        "content_snapshot_id": "cs_1",
        "query": "核心逻辑",
        "symbol": "600000",
        "business_as_of": NOW,
        "knowledge_as_of": NOW,
        "availability_as_of": NOW,
        "minimum_support_status": "SOURCE_SUPPORTED",
        "max_items": 30,
    }
    values.update(changes)
    return KnowledgeBundleRequest(**values)


def _item(identifier="co_1"):
    return {
        "knowledge_id": identifier,
        "claim_id": "cl_" + identifier,
        "occurrence_id": identifier,
        "statement": "收入增长约20%",
        "subject": {"type": "EQUITY", "key": "600000"},
        "predicate": "revenue_growth",
        "object": {"value": 20.0, "unit": "percent"},
        "support_status": "SOURCE_SUPPORTED",
        "lifecycle_status": "ACTIVE",
        "temporal": {
            "target_start": "2026-10-01T08:00:00+08:00",
            "target_end": "2026-12-31T00:00:00Z",
            "precision": "EXACT",
        },
        "evidence": [
            {
                "evidence_id": "ev_" + identifier,
                "ownership": "PRIMARY",
                "start_ms": 1,
                "end_ms": 2,
                "quote": "增长",
                "artifact_id": "tr_1",
                "segment_id": "seg_1",
                "quote_hash": "sha256:" + hashlib.sha256(canonical_json("增长").encode("utf-8")).hexdigest(),
                "modality": "transcript",
            }
        ],
        "verification": {"status": "SOURCE_VERIFIED", "reason_codes": ["B", "A"]},
        "known_from": "2026-09-05T00:00:00Z",
        "available_from": "2026-09-05T00:00:00Z",
        "claim_schema_version": "claim.atomic.v1",
        "grounding_status": "GROUNDED",
        "legacy_grounding_incomplete": False,
    }


class Authority:
    def __init__(self, items=None, source_available_from=None):
        self.items = items or [_item()]
        self.source_available_from = source_available_from

    def read_bundle_source(self, request):
        return {
            "snapshot_available": True,
            "source": {
                "source_type": "bilibili",
                "source_identity_hash": "identity-1",
                "source_version_id": "version-1",
                "canonical_url": "https://x.test/a?signature=secret#private",
                "source_content_hash": "h",
            },
            "source_available_from": self.source_available_from,
            "items": deepcopy(self.items),
            "warnings": ["z", "a"],
        }


def _service(authority=None):
    return KnowledgeBundleService(
        authority or Authority(),
        InMemoryKnowledgeBundleRepository(),
        BundleProducerMetadata("stock_content", "1.0.0", "abc123", "pipeline.v3", CHECKSUM),
    )


def _v2_service(authority=None):
    return KnowledgeBundleService(
        authority or Authority(),
        InMemoryKnowledgeBundleRepository(),
        BundleProducerMetadata("stock_content", "1.0.0", "abc123", "pipeline.v3", CHECKSUM),
        v2_contract_checksum=V2_CHECKSUM,
    )


def _v2_item(identifier="co_1", *, review=False, nature="METHOD"):
    item = _item(identifier)
    item.update({
        "statement": "课程提出三层资金池分别承担压舱、核心和卫星风险",
        "subject": {"type": "TOPIC", "key": "capital_allocation"},
        "primary_domain": "PORTFOLIO_RISK_MANAGEMENT",
        "claim_nature": nature,
        "source_grade": "PRIMARY",
        "external_truth_status": "EXTERNALLY_VERIFIED",
        "attribution": {
            "attributed": nature in {"FORECAST", "CAUSAL_THESIS", "OPINION"},
            "source_label": "视频讲者" if nature in {"FORECAST", "CAUSAL_THESIS", "OPINION"} else None,
        },
        "detail": {
            "explanation": "以不同资金桶隔离波动风险。",
            "mechanism": "压舱层降低组合波动，卫星层承载高波动机会。",
            "procedure": "先确定各层比例，再按层设置标的和集中度限制。",
            "formula": "压舱30%-40%，核心40%-50%，卫星10%-20%。",
            "example": None,
            "scope": "适用于课程所述的长期权益配置框架。",
            "risks": "比例不是个体化投资建议，需要结合风险承受能力。",
        },
        "temporal": {
            "kind": "RECURRING_RULE", "start": None, "end": None, "as_of": None,
            "rule": "按月复核", "label": None, "precision": "RULE", "explicitly_unknown": False,
        },
        "occurrence_review": {
            "status": "HUMAN_REVIEW_REQUIRED" if review else "NOT_REQUIRED",
            "reason_codes": ["ASR_OCR_NUMERIC_CONFLICT"] if review else [],
        },
        "numeric_claim": True,
        "evidence": [{
            "evidence_id": "ev_" + identifier,
            "ownership": "PRIMARY",
            "modality": "transcript",
            "artifact_id": "tr_1",
            "artifact_hash": "sha256:" + "a" * 64,
            "locator": {"segment_id": "seg_1", "frame_id": None, "start_ms": 1, "end_ms": 2, "bbox": None},
            "content": "压舱层三到四成，核心层四到五成，卫星层一到二成。",
        }],
    })
    return item


def test_c14n_vectors_and_rejection():
    fixture = json.loads(
        (Path(__file__).parents[1] / "contracts" / "fixtures" / "content-knowledge-bundle.c14n-v1.json").read_text(
            encoding="utf-8"
        )
    )
    for vector in fixture["vectors"][:-1]:
        assert canonical_json(vector["left"]) == canonical_json(vector["right"])
    assert canonical_json(fixture["vectors"][-1]["left"]) != canonical_json(fixture["vectors"][-1]["right"])
    assert canonical_json({"b": 1.0, "a": "e\u0301"}) == canonical_json({"a": "é", "b": 1})
    assert canonical_json({"at": "2026-09-06T08:00:00+08:00"}) == '{"at":"2026-09-06T00:00:00Z"}'
    assert canonical_json({"evidence_refs": ["b", "a"]}) == canonical_json({"evidence_refs": ["a", "b"]})
    with pytest.raises(CanonicalizationError):
        canonical_json({"x": float("nan")})
    with pytest.raises(CanonicalizationError):
        canonical_json({"x": float("inf")})


def test_public_bundle_schema_rejects_incomplete_or_unknown_nested_fields():
    root = Path(__file__).parents[1]
    fixture = json.loads((root / "contracts" / "fixtures" / "content-knowledge-bundle.c14n-v1.json").read_text())
    schema = json.loads((root / "contracts" / "content-knowledge-bundle.v1.json").read_text())
    bundle = _service().create(_request())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert not list(validator.iter_errors(bundle))
    for vector in fixture["schema_rejections"]:
        candidate = deepcopy(bundle)
        parent = candidate
        for part in vector["path"][:-1]:
            parent = parent[part]
        leaf = vector["path"][-1]
        if vector["operation"] == "remove":
            del parent[leaf]
        else:
            parent[leaf] = vector["value"]
        assert list(validator.iter_errors(candidate)), vector["name"]


def test_public_bundle_wire_validates_in_a_fresh_python_process(tmp_path):
    root = Path(__file__).parents[1]
    schema_path = root / "contracts" / "content-knowledge-bundle.v1.json"
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(canonical_json(_service().create(_request())), encoding="utf-8")
    command = (
        "import json,sys; from jsonschema import Draft202012Validator,FormatChecker; "
        "schema=json.load(open(sys.argv[1], encoding='utf-8')); "
        "payload=json.load(open(sys.argv[2], encoding='utf-8')); "
        "errors=list(Draft202012Validator(schema,format_checker=FormatChecker()).iter_errors(payload)); "
        "raise SystemExit(1 if errors else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(schema_path), str(bundle_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_request_binding_sort_hash_and_immutable_get():
    authority = Authority([_item("co_z"), _item("co_a")])
    service = _service(authority)
    first = service.create(_request())
    second = service.create(_request())
    assert first == second == service.get(first["bundle_id"])
    assert [item["occurrence_id"] for item in first["items"]] == ["co_a", "co_z"]
    # ``grounding_status`` is part of the public Bundle item wire contract.
    # Do not strip the proof after validating it locally: the independent
    # stock_agent consumer fails closed when it is absent.
    assert {item["grounding_status"] for item in first["items"]} == {"GROUNDED"}
    assert first["source"]["canonical_url"] == "https://x.test/a"
    assert first["bundle_hash"] != service.create(_request(query="changed"))["bundle_hash"]
    assert first["bundle_hash"] != service.create(_request(symbol="000001"))["bundle_hash"]
    assert first["bundle_hash"] != service.create(_request(policy_version="policy.v2"))["bundle_hash"]
    authority.items[0]["statement"] = "tampered"
    assert first["bundle_hash"] != service.create(_request())["bundle_hash"]


def test_strict_bundle_accepts_utc_boundaries_from_explicit_chinese_quarter():
    """A normalized DATE period is serialized as an aware Bundle instant, not prose."""
    period = TemporalNormalizer().normalize("2025年第三季度")
    item = _item()
    target_start = datetime.combine(
        period.start_date, datetime.min.time(), tzinfo=UTC
    ).isoformat().replace("+00:00", "Z")
    target_end = datetime.combine(
        period.end_date, datetime.min.time(), tzinfo=UTC
    ).isoformat().replace("+00:00", "Z")
    item["temporal"] = {
        "target_start": target_start,
        "target_end": target_end,
        "precision": period.precision.value,
    }
    bundle = _service(Authority([item])).create(_request())
    assert bundle["items"][0]["temporal"] == {
        "target_start": "2025-07-01T00:00:00Z",
        "target_end": "2025-09-30T00:00:00Z",
        "precision": "QUARTER",
    }


def test_raw_public_payload_is_deterministic_for_equivalent_authority_ordering():
    first_item, second_item = _item("co_a"), _item("co_z")
    first_item["evidence"].append({
        "evidence_id": "ev_secondary", "ownership": "SECONDARY", "start_ms": 3, "end_ms": 4,
        "quote": "补充", "artifact_id": "tr_2", "segment_id": "seg_2", "quote_hash": "quote-secondary",
        "modality": "transcript",
    })
    first_item["verification"]["reason_codes"] = ["Z", "A"]
    left = _service(Authority([first_item, second_item])).create(_request())
    right_item = deepcopy(first_item)
    right_item["evidence"].reverse()
    right_item["verification"]["reason_codes"].reverse()
    right = _service(Authority([second_item, right_item])).create(_request())
    assert left == right
    assert canonical_json(left) == canonical_json(right)


def test_sqlite_bundle_insert_is_idempotent_and_rejects_immutable_collision(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'bundle.db'}")
    database.create_schema()
    repository = PostgresKnowledgeBundleRepository(database.session_factory)
    bundle = _service().create(_request())
    assert repository.insert(bundle) == bundle
    assert repository.insert(deepcopy(bundle)) == bundle
    conflicting = deepcopy(bundle)
    conflicting["bundle_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="immutable bundle id collision"):
        repository.insert(conflicting)


def test_qdrant_sentinel_is_never_selected_for_formal_bundle():
    class SqlOnlyAuthority(Authority):
        def qdrant_search(self, *_):
            raise AssertionError("formal bundle must not consult Qdrant")

    assert _service(SqlOnlyAuthority()).create(_request())["items"]


def test_pit_legacy_evidence_and_secret_rejections():
    for change, error in (
        ({"available_from": "2026-09-07T00:00:00Z"}, "AVAILABILITY_AS_OF_EXCEEDED"),
        ({"known_from": "2026-09-07T00:00:00Z"}, "KNOWLEDGE_AS_OF_EXCEEDED"),
        ({"claim_schema_version": "claim.v2"}, "LEGACY_OR_UNGROUNDED_CLAIM"),
        ({"evidence": []}, "PRIMARY_EVIDENCE_REQUIRED"),
    ):
        item = _item()
        item.update(change)
        with pytest.raises(ValueError, match=error):
            _service(Authority([item])).create(_request())
    item = _item()
    item["statement"] = "signed URL"
    with pytest.raises(ValueError, match="UNSAFE_BUNDLE_CONTENT"):
        _service(Authority([item])).create(_request())
    with pytest.raises(ValueError, match="concrete snapshot"):
        _request(content_snapshot_id="latest")


def test_bundle_rejects_non_scalar_object_value_before_schema_serialization():
    item = _item()
    item["object"]["value"] = {"unreviewed": True}
    with pytest.raises(ValueError, match="INVALID_BUNDLE_OBJECT_VALUE"):
        _service(Authority([item])).create(_request())


@pytest.mark.parametrize(
    ("field", "event_time", "request_time", "error"),
    (
        ("source_available_from", "2026-09-06T00:00:00Z", "2026-09-06T00:00:00+00:00", None),
        ("source_available_from", "2026-09-06T08:00:00+08:00", "2026-09-06T00:00:00Z", None),
        ("source_available_from", "2026-09-06T00:00:00Z", "2026-09-05T23:59:59Z", "SOURCE_NOT_AVAILABLE"),
        ("source_available_from", "2026-09-06T00:00:00Z", "2026-09-06T00:00:01Z", None),
        ("known_from", "2026-09-06T00:00:00Z", "2026-09-06T00:00:00+00:00", None),
        ("known_from", "2026-09-06T08:00:00+08:00", "2026-09-06T00:00:00Z", None),
        ("known_from", "2026-09-06T00:00:00Z", "2026-09-05T23:59:59Z", "KNOWLEDGE_AS_OF_EXCEEDED"),
        ("known_from", "2026-09-06T00:00:00Z", "2026-09-06T00:00:01Z", None),
        ("available_from", "2026-09-06T00:00:00Z", "2026-09-06T00:00:00+00:00", None),
        ("available_from", "2026-09-06T08:00:00+08:00", "2026-09-06T00:00:00Z", None),
        ("available_from", "2026-09-06T00:00:00Z", "2026-09-05T23:59:59Z", "AVAILABILITY_AS_OF_EXCEEDED"),
        ("available_from", "2026-09-06T00:00:00Z", "2026-09-06T00:00:01Z", None),
    ),
)
def test_bundle_pit_guards_compare_aware_instants_inclusively(field, event_time, request_time, error):
    request_field = {
        "source_available_from": "availability_as_of",
        "known_from": "knowledge_as_of",
        "available_from": "availability_as_of",
    }[field]
    request = _request(**{request_field: datetime.fromisoformat(request_time.replace("Z", "+00:00"))})
    if field == "source_available_from":
        service = _service(Authority(source_available_from=event_time))
    else:
        item = _item()
        item[field] = event_time
        service = _service(Authority([item]))
    if error:
        with pytest.raises(ValueError, match=error):
            service.create(request)
    else:
        assert service.create(request)["items"]


@pytest.mark.parametrize("field", ("source_available_from", "known_from", "available_from"))
@pytest.mark.parametrize("invalid", ("not-an-instant", "2026-09-06T00:00:00"))
def test_bundle_pit_guards_reject_malformed_or_naive_authority_instants(field, invalid):
    if field == "source_available_from":
        service = _service(Authority(source_available_from=invalid))
    else:
        item = _item()
        item[field] = invalid
        service = _service(Authority([item]))
    with pytest.raises(ValueError, match=f"INVALID_{field.upper()}"):
        service.create(_request())


def test_schema_manifest_checksum_and_post_get_adapter():
    path = Path(__file__).parents[1] / "contracts" / "content-knowledge-bundle.v1.json"
    json.loads(path.read_text(encoding="utf-8"))
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest().upper() == CHECKSUM
    service = _service()
    application = type("App", (), {"create_knowledge_bundle": service.create, "get_knowledge_bundle": service.get})()
    app = FastAPI()
    app.include_router(create_knowledge_bundles_router(lambda: application))
    with TestClient(app) as client:
        response = client.post(
            "/v1/content/knowledge-bundles",
            json={
                key: value.isoformat().replace("+00:00", "Z") if isinstance(value, datetime) else value
                for key, value in _request().canonical_request().items()
            },
        )
        assert response.status_code == 200
        assert client.get("/v1/content/knowledge-bundles/" + response.json()["bundle_id"]).json() == response.json()


def test_production_application_injects_uppercase_locked_bundle_checksum(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'configured-content.db'}"
    Database(database_url).create_schema()
    monkeypatch.setenv("CONTENT_SERVICE_VERSION", "test")
    monkeypatch.setenv("CONTENT_GIT_COMMIT", "content-test-sha")
    monkeypatch.setenv("CONTENT_PIPELINE_VERSION", "pipeline-test")
    application = build_application(database_url, enable_qdrant=False)
    producer = application._knowledge_bundle_service._producer  # noqa: SLF001 - composition contract probe
    assert producer.contract_checksum == CHECKSUM


def test_immutable_migration_declares_update_rejection():
    migration = (Path(__file__).parents[1] / "migrations" / "033_content_knowledge_bundle.sql").read_text(
        encoding="utf-8"
    )
    assert "content_knowledge_bundle" in migration
    assert "BEFORE UPDATE" in migration
    assert "immutable" in migration


def test_v2_bundle_keeps_v1_replay_locked_and_exposes_reviewed_multitopic_semantics():
    first, conflict = _v2_item("co_1"), _v2_item("co_2", review=True)
    authority = Authority([first, conflict])
    request = _request(symbol="UNSPECIFIED", contract_version=V2_CONTRACT)
    bundle = _v2_service(authority).create(request)
    schema = json.loads((Path(__file__).parents[1] / "contracts" / "content-knowledge-bundle.v2.json").read_text())
    schema_path = Path(__file__).parents[1] / "contracts" / "content-knowledge-bundle.v2.json"
    assert "sha256:" + hashlib.sha256(schema_path.read_bytes()).hexdigest().upper() == V2_CHECKSUM
    assert not list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(bundle))
    assert bundle["request"]["subject_scope"] == "ALL_SUBJECTS"
    assert bundle["scope"] == {"subject_scope": "ALL_SUBJECTS", "requested_subject": None}
    assert bundle["items"][0]["statement"] == "三层资金池分别承担压舱、核心和卫星风险"
    assert bundle["quality"]["candidate_count"] == 2
    assert bundle["quality"]["knowledge_count"] == 1
    assert bundle["quality"]["eligible_candidate_count"] == 1
    assert bundle["quality"]["excluded_candidate_count"] == 1
    assert bundle["quality"]["truncated_candidate_count"] == 0
    assert bundle["quality"]["grounded_count"] == 1
    assert bundle["quality"]["grounded_ratio"] == 1
    assert bundle["quality"]["numeric_grounded_ratio"] == 1
    assert bundle["quality"]["human_review_required_count"] == 1
    assert "HUMAN_REVIEW_REQUIRED_ITEMS_EXCLUDED" in bundle["quality"]["warnings"]
    # Contract selection never mutates v1's canonical request/hash path.
    assert _service(Authority([_item()])).create(_request())["contract"] == "content-knowledge-bundle.v1"


def test_v2_requires_attribution_for_source_forecasts_and_explicit_unknown_time():
    item = _v2_item(nature="FORECAST")
    item["attribution"] = {"attributed": False, "source_label": None}
    with pytest.raises(ValueError, match="ATTRIBUTION_REQUIRED"):
        _v2_service(Authority([item])).create(_request(contract_version=V2_CONTRACT))
    item = _v2_item()
    item["temporal"] = {
        "kind": "UNKNOWN", "start": "2026-09-06T00:00:00Z", "end": None, "as_of": None,
        "rule": None, "label": None, "precision": "UNKNOWN", "explicitly_unknown": True,
    }
    with pytest.raises(ValueError, match="UNKNOWN_TEMPORAL_MUST_NOT_INVENT_DATE"):
        _v2_service(Authority([item])).create(_request(contract_version=V2_CONTRACT))


def test_v2_request_body_accepts_consistent_subject_scope_and_rejects_a_mismatch():
    service = _v2_service(Authority([_v2_item()]))
    application = type("App", (), {"create_knowledge_bundle": service.create, "get_knowledge_bundle": service.get})()
    app = FastAPI()
    app.include_router(create_knowledge_bundles_router(lambda: application))
    request = _request(symbol="UNSPECIFIED", contract_version=V2_CONTRACT).canonical_request()
    request = {
        key: value.isoformat().replace("+00:00", "Z") if isinstance(value, datetime) else value
        for key, value in request.items()
    }
    request["subject_scope"] = "ALL_SUBJECTS"
    with TestClient(app) as client:
        response = client.post("/v1/content/knowledge-bundles", json=request)
        assert response.status_code == 200
        assert response.json()["request"]["subject_scope"] == "ALL_SUBJECTS"
        request["subject_scope"] = "SUBJECT_ONLY"
        assert client.post("/v1/content/knowledge-bundles", json=request).status_code == 422


def test_v2_quality_reconciles_published_items_after_truncation_and_excludes_conflicts():
    eligible = [_v2_item(f"co_{letter}") for letter in ("a", "b", "c")]
    for index, item in enumerate(eligible):
        item["primary_domain"] = (
            "AI_INFERENCE_COMPUTE",
            "INFORMATION_INFRASTRUCTURE_POLICY",
            "CENTRAL_BANK_GOLD_RESERVES",
        )[index]
        item["subject"] = {"type": "TOPIC", "key": f"topic_{index}"}
        item["predicate"] = f"predicate_{index}"
    review = _v2_item("co_review", review=True)
    contradictory = _v2_item("co_conflict")
    contradictory["contradiction_group_id"] = "cg_1"
    bundle = _v2_service(Authority([review, eligible[2], contradictory, eligible[1], eligible[0]])).create(
        _request(symbol="UNSPECIFIED", contract_version=V2_CONTRACT, max_items=2)
    )
    quality = bundle["quality"]
    assert len(bundle["items"]) == quality["knowledge_count"] == 2
    assert {item["occurrence_id"] for item in bundle["items"]}.isdisjoint({"co_review", "co_conflict"})
    assert quality["candidate_count"] == 5
    assert quality["eligible_candidate_count"] == 3
    assert quality["excluded_candidate_count"] == 2
    assert quality["truncated_candidate_count"] == 1
    assert quality["grounded_count"] == 2
    assert quality["numeric_candidate_count"] == quality["numeric_grounded_count"] == 2
    assert quality["grounded_ratio"] == quality["numeric_grounded_ratio"] == 1
    assert "PUBLIC_STRICT_ELIGIBLE_ITEMS_TRUNCATED" in quality["warnings"]
    assert "CONFLICT_OR_REVIEW_EVIDENCE_PRESENT" in quality["warnings"]


def test_v2_secondary_unchecked_fact_is_attributed_and_never_in_grounded_quality():
    primary = _v2_item("co_primary")
    secondary = _v2_item("co_secondary")
    secondary.update({
        "primary_domain": "CENTRAL_BANK_GOLD_RESERVES",
        "claim_nature": "POLICY_FACT",
        "source_grade": "SECONDARY",
        "external_truth_status": "NOT_CHECKED",
        "attribution": {"attributed": True, "source_label": "财经媒体报道"},
    })
    bundle = _v2_service(Authority([primary, secondary])).create(
        _request(symbol="UNSPECIFIED", contract_version=V2_CONTRACT)
    )
    quality = bundle["quality"]
    assert quality["knowledge_count"] == quality["numeric_candidate_count"] == 2
    assert quality["grounded_count"] == quality["numeric_grounded_count"] == 1
    assert quality["grounded_ratio"] == quality["numeric_grounded_ratio"] == 0.5
    assert quality["secondary_only_count"] == quality["external_truth_not_checked_count"] == 1
    assert "SECONDARY_EXTERNAL_FACTS_PRESENT" in quality["warnings"]
    assert "EXTERNAL_TRUTH_NOT_CHECKED_ITEMS_PRESENT" in quality["warnings"]
    item = next(value for value in bundle["items"] if value["occurrence_id"] == "co_secondary")
    assert item["source_grade"] == "SECONDARY"
    assert item["external_truth_status"] == "NOT_CHECKED"


def test_v2_detail_that_only_repeats_the_statement_fails_closed():
    item = _v2_item()
    item["detail"] = {"explanation": item["statement"]}
    with pytest.raises(ValueError, match="NONTRIVIAL_KNOWLEDGE_DETAIL_REQUIRED"):
        _v2_service(Authority([item])).create(_request(contract_version=V2_CONTRACT))

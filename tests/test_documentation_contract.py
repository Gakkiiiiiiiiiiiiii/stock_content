"""Evidence-boundary documentation and OpenAPI regression checks."""
from __future__ import annotations

from pathlib import Path

from stock_content.api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_public_docs_describe_only_the_v5_1_evidence_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    authority = (ROOT / "docs" / "slo" / "fact-authority.md").read_text(encoding="utf-8")
    governance = (ROOT / "docs" / "security" / "source-governance.md").read_text(encoding="utf-8")
    adr = (ROOT / "docs" / "adr" / "0005-source-governance-evidence.md").read_text(encoding="utf-8")

    for text in (readme, authority):
        assert "content-factor-signal.v5.1" in text
    assert all(clock in readme for clock in ("business_as_of", "knowledge_as_of", "availability_as_of"))
    assert "read_only_facts" in readme
    assert "formal_publish" in readme
    assert "derived_search" in readme
    assert "BLOCKED" in readme
    assert "source-policy.v1" in governance
    assert "source-governance-evidence.v1" in governance
    assert "pii-redaction.v1" in governance
    assert "metadata-drifting" in adr

    for target in (
        "docs/security/source-governance.md",
        "docs/slo/fact-authority.md",
        "docs/runbooks/README.md",
        "docs/migrations/signal-v3-to-v5_1.md",
        "docs/pending-work-2026-09-04.md",
    ):
        assert f"]({target})" in readme
        assert (ROOT / target).is_file()


def test_openapi_descriptions_preserve_the_readiness_and_governance_boundary() -> None:
    schema = create_app(object()).openapi()

    description = schema["info"]["description"]
    assert "content-factor-signal.v5.1" in description
    assert all(clock in description for clock in ("business_as_of", "knowledge_as_of", "availability_as_of"))
    assert "source-policy.v1" in description

    readiness = schema["paths"]["/readiness"]["get"]
    assert "SQL-backed" in readiness["description"]
    assert "derived_search" in readiness["description"]

    formal = schema["paths"]["/internal/v2/factor-signals/query"]["post"]
    assert "content-factor-signal.v5.1" in formal["description"]
    assert "content_snapshot_id" in formal["description"]

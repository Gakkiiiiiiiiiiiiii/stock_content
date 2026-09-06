# Source governance

Ingestion is permitted only when `source-policy.v1` (`SourcePolicy`) allows
the intended use and rate limit. The immutable source artifact carries a
`source-governance-evidence.v1` record: source type, policy version, license,
robots/terms reference, retention class, access and privacy classifications,
and the required `pii-redaction.v1` policy. Formal release re-reads and
validates that artifact; missing, unsupported, or metadata-drifting evidence
is fail-closed. Artifact removal is represented by an append-only tombstone
carrying actor, reason and policy version; derived records follow the explicit
derived retention period.

PII direct identifiers detected in transcript material are replaced before a
transcript artifact is created. The original identifier is not retained in the
artifact. Dependency evidence is generated with `generate_sbom.py` and may be
checked against the selected project profile with `verify_sbom.py`.

The governed-source and redaction checks are deterministic repository evidence;
a production retention/deletion schedule, approved source inventory, and its
external operating drill remain **BLOCKED** pending the required storage and
governance approvals.

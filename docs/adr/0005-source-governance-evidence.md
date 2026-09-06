# ADR 0005: Versioned source-governance evidence

Status: accepted.

Each governed source is evaluated against `source-policy.v1`. Its immutable
artifact metadata contains `source-governance-evidence.v1` and
`pii-redaction.v1`, including the source type, policy version, license,
robots-or-terms reference, retention, and access/privacy classifications.
Validation rejects missing, unsupported, or metadata-drifting records; replay
and formal release therefore do not read a newer policy into an older artifact.

This decision does not claim a completed production retention/deletion drill.
That external exercise remains **BLOCKED** pending an approved source
inventory, retention schedule, storage/KMS controls, and scheduler access.

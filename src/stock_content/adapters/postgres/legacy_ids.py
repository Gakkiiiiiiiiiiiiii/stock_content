"""Stable identifiers used while upgrading the legacy content schema."""
from __future__ import annotations

import hashlib


def legacy_evidence_member_id(claim_id: str, evidence_id: str) -> str:
    """Return the migration-compatible claim/evidence membership key."""
    return hashlib.md5(  # noqa: S324 - compatibility identity, not security
        f"{claim_id}:{evidence_id}".encode(), usedforsecurity=False
    ).hexdigest()


def legacy_verification_id(claim_id: str) -> str:
    """Return the migration-compatible ID for a copied lifecycle result."""
    return "legacy-" + hashlib.md5(  # noqa: S324 - compatibility identity, not security
        str(claim_id).encode(), usedforsecurity=False
    ).hexdigest()

"""Regression coverage for psycopg's indeterminate INSERT rowcount."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from stock_content.adapters.postgres.models import (
    ClaimOccurrenceRow,
    ClaimVerificationResultRow,
    ContentSnapshotRow,
)
from stock_content.adapters.postgres.repositories.claim_occurrence_repository import (
    _insert_ignore as insert_occurrence_ignore,
)
from stock_content.adapters.postgres.repositories.snapshot_repository import (
    _insert_ignore as insert_snapshot_ignore,
)
from stock_content.adapters.postgres.repositories.verification_job_repository import (
    _insert_ignore as insert_verification_ignore,
)


class _UnreliablePsycopgResult:
    """Mimic a driver result whose rowcount cannot identify an insertion."""

    rowcount = -1

    def __init__(self, returned_primary_key: str | None) -> None:
        self._returned_primary_key = returned_primary_key

    def scalar_one_or_none(self) -> str | None:
        return self._returned_primary_key


class _PostgresSession:
    def __init__(self, returned_primary_key: str | None) -> None:
        self.statement = None
        self._result = _UnreliablePsycopgResult(returned_primary_key)

    def get_bind(self):
        return SimpleNamespace(dialect=postgresql.dialect())

    def execute(self, statement):
        self.statement = statement
        return self._result


@pytest.mark.parametrize(
    ("insert_ignore", "model", "values", "conflict_columns", "primary_key"),
    [
        (
            insert_snapshot_ignore,
            ContentSnapshotRow,
            {"content_snapshot_id": "snapshot-1"},
            [ContentSnapshotRow.content_snapshot_id],
            "content_snapshot.content_snapshot_id",
        ),
        (
            insert_occurrence_ignore,
            ClaimOccurrenceRow,
            {"occurrence_id": "occurrence-1"},
            None,
            "claim_occurrence.occurrence_id",
        ),
        (
            insert_verification_ignore,
            ClaimVerificationResultRow,
            {"verification_id": "verification-1"},
            [ClaimVerificationResultRow.verification_id],
            "claim_verification_result.verification_id",
        ),
    ],
)
def test_postgres_insert_ignore_uses_returned_primary_key_not_unreliable_rowcount(
    insert_ignore,
    model,
    values,
    conflict_columns,
    primary_key,
):
    winner = _PostgresSession(values[next(iter(values))])
    loser = _PostgresSession(None)

    if conflict_columns is None:
        assert insert_ignore(winner, model, values) is True
        assert insert_ignore(loser, model, values) is False
    else:
        assert insert_ignore(winner, model, values, conflict_columns) is True
        assert insert_ignore(loser, model, values, conflict_columns) is False

    compiled = str(winner.statement.compile(dialect=postgresql.dialect()))
    assert f"RETURNING {primary_key}" in compiled

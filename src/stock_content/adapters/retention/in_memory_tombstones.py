from __future__ import annotations

from stock_content.domain.retention import RetentionExecution, RetentionExecutionState, Tombstone


class InMemoryTombstoneRepository:
    """Deterministic adapter used by policy tests and dry-run integrations."""
    def __init__(self) -> None:
        self.items: dict[str, Tombstone] = {}
        self.executions: dict[str, RetentionExecution] = {}

    def get(self, tombstone_id: str) -> Tombstone | None:
        return self.items.get(tombstone_id)

    def insert_immutable(self, tombstone: Tombstone) -> Tombstone:
        existing = self.items.get(tombstone.tombstone_id)
        if existing is not None and existing != tombstone:
            raise ValueError("immutable tombstone collision")
        self.items[tombstone.tombstone_id] = tombstone
        return tombstone

    def get_execution(self, tombstone_id: str) -> RetentionExecution | None:
        return self.executions.get(tombstone_id)

    def prepare_execution(self, tombstone: Tombstone) -> RetentionExecution:
        existing = self.executions.get(tombstone.tombstone_id)
        if existing is not None:
            if existing.tombstone != tombstone:
                raise ValueError("immutable tombstone collision")
            return existing
        self.insert_immutable(tombstone)
        execution = RetentionExecution(tombstone, RetentionExecutionState.DELETE_PENDING)
        self.executions[tombstone.tombstone_id] = execution
        return execution

    def mark_deleted(self, tombstone_id: str) -> RetentionExecution:
        execution = self.executions[tombstone_id]
        if execution.state is RetentionExecutionState.DELETED:
            return execution
        execution = RetentionExecution(
            execution.tombstone, RetentionExecutionState.DELETED, execution.attempt_count + 1
        )
        self.executions[tombstone_id] = execution
        return execution

    def mark_failed(self, tombstone_id: str, error_code: str) -> RetentionExecution:
        execution = self.executions[tombstone_id]
        if execution.state is RetentionExecutionState.DELETED:
            return execution
        execution = RetentionExecution(
            execution.tombstone, RetentionExecutionState.DELETE_FAILED, execution.attempt_count + 1, error_code
        )
        self.executions[tombstone_id] = execution
        return execution

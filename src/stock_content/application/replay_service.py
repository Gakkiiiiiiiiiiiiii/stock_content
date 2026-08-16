"""Deterministic Replay（详细修改方案 §4 P0-2/P0-4）。

从 ContentSnapshot 身份重算快照 ID：相同输入必须得到同一身份。
EXACT replay 不重新抓取源内容，仅验证身份与血缘可重现。
"""
from __future__ import annotations

from typing import Any

from stock_content.application.snapshot_service import SnapshotService
from stock_content.domain.lineage import compute_content_snapshot_id, lineage_of, snapshot_identity_payload


class ReplayService:
    def __init__(self, snapshots: SnapshotService) -> None:
        self._snapshots = snapshots

    def replay(self, content_snapshot_id: str) -> dict[str, Any]:
        snapshot = self._snapshots.get(content_snapshot_id)
        if snapshot is None:
            return {"error": "SNAPSHOT_NOT_FOUND", "content_snapshot_id": content_snapshot_id}
        identity = snapshot_identity_payload(
            source_content_hash=snapshot.source_content_hash,
            pipeline_version=snapshot.pipeline_version,
            parser_version=snapshot.parser_version,
            asr_model=snapshot.asr_model,
            asr_model_version=snapshot.asr_model_version,
            vision_model=snapshot.vision_model,
            llm_model=snapshot.llm_model,
            prompt_bundle_version=snapshot.prompt_bundle_version,
            entity_alias_version=snapshot.entity_alias_version,
            verification_policy_version=snapshot.verification_policy_version,
            quant_market_snapshot_ids=snapshot.quant_market_snapshot_ids,
            code_sha=snapshot.code_sha,
            config_hash=snapshot.config_hash,
        )
        recomputed = f"cs-{compute_content_snapshot_id(identity)[:32]}"
        return {
            "content_snapshot_id": content_snapshot_id,
            "replay_mode": "EXACT",
            "identity_match": recomputed == content_snapshot_id,
            "recomputed_snapshot_id": recomputed,
            "artifact_ids": dict(snapshot.artifact_ids),
            "lineage": lineage_of(snapshot).to_dict(),
        }


__all__ = ["ReplayService"]

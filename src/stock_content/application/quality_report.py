"""Versioned, deterministic quality release report and gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    precision: float = 0.0
    recall: float = 0.0
    evidence_coverage: float = 0.0
    temporal_accuracy: float = 0.0
    replay_mismatch_count: int = 0
    resource_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReleaseThresholds:
    min_precision: float = 0.95
    min_recall: float = 0.90
    min_evidence_coverage: float = 0.95
    min_temporal_accuracy: float = 0.95
    max_replay_mismatches: int = 0
    max_semantic_mismatches: int = 0


@dataclass(frozen=True, slots=True)
class QualityReport:
    report_version: str
    golden_set_version: str
    contract_version: str
    metrics: QualityMetrics
    gate_result: str
    blocking_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "golden_set_version": self.golden_set_version,
            "contract_version": self.contract_version,
            "extraction_metrics": {"precision": self.metrics.precision, "recall": self.metrics.recall},
            "evidence_metrics": {"coverage": self.metrics.evidence_coverage},
            "temporal_metrics": {"accuracy": self.metrics.temporal_accuracy},
            "replay_mismatch_count": self.metrics.replay_mismatch_count,
            "resource_metrics": dict(self.metrics.resource_metrics),
            "gate_result": self.gate_result,
            "blocking_reasons": list(self.blocking_reasons),
        }


def evaluate_quality(
    metrics: QualityMetrics,
    *,
    golden_set_version: str = "golden.v1",
    contract_version: str = "content-factor-signal.v5.1",
    thresholds: ReleaseThresholds | None = None,
    semantic_mismatch_count: int = 0,
) -> QualityReport:
    limits = thresholds or ReleaseThresholds()
    reasons: list[str] = []
    checks = (
        (metrics.precision < limits.min_precision, "precision_below_threshold"),
        (metrics.recall < limits.min_recall, "recall_below_threshold"),
        (metrics.evidence_coverage < limits.min_evidence_coverage, "evidence_coverage_below_threshold"),
        (metrics.temporal_accuracy < limits.min_temporal_accuracy, "temporal_accuracy_below_threshold"),
        (metrics.replay_mismatch_count > limits.max_replay_mismatches, "replay_mismatch"),
        (semantic_mismatch_count > limits.max_semantic_mismatches, "semantic_mismatch"),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return QualityReport(
        "quality-report.v1", golden_set_version, contract_version, metrics,
        "BLOCKED" if reasons else "PASS", tuple(reasons),
    )


__all__ = ["QualityMetrics", "QualityReport", "ReleaseThresholds", "evaluate_quality"]

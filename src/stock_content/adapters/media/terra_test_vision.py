"""Explicit test-only adapter for externally reviewed Terra visual fixtures.

This module is deliberately not wired into the application dependency graph.
Callers must instantiate it themselves with ``environment="test"`` and pass
it to :class:`VisionStage`.  It therefore cannot become a production vision
fallback when the configured HTTP model is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

TERRA_TEST_FIXTURE_SCHEMA = "terra-test-vision-fixture.v1"
TERRA_TEST_ADAPTER_VERSION = "terra-test-vision-fixture-adapter.v1"
TERRA_TEST_MODEL_NAME = "terra-test-substitute"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Terra test fixture {field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Terra test fixture {field} must be a list of non-empty strings")
    if required and not value:
        raise ValueError(f"Terra test fixture {field} must not be empty")
    return list(value)


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("Terra test fixture confidence_score must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("Terra test fixture confidence_score must be between 0 and 1")
    return result


def _bbox(value: Any) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Terra test fixture bbox must contain exactly four coordinates")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value
    ):
        raise ValueError("Terra test fixture bbox coordinates must be finite numbers")
    return list(value)


class TerraTestVisionFixtureAnalyzer:
    """Consume a strict Terra fixture in an explicit, non-production test run.

    The fixture is intentionally per-frame and binds each observation to the
    materialized frame's id, timestamp and content hash before analysis.  Its
    canonical content hash is copied into every resulting visual payload so
    the normal VisionArtifact and C4/C5 audit chain remain content-addressed.
    """

    def __init__(self, fixture: dict[str, Any], *, environment: str) -> None:
        if environment != "test":
            raise ValueError("Terra test vision fixture adapter requires explicit environment='test'")
        self._fixture_hash = _canonical_hash(fixture)
        self._items = self._validate_fixture(fixture)
        self._by_frame_id: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_json_file(cls, path: str | Path, *, environment: str) -> "TerraTestVisionFixtureAnalyzer":
        """Load a JSON fixture only when the caller explicitly marks it test-only."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Terra test vision fixture must be readable JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Terra test vision fixture root must be an object")
        return cls(payload, environment=environment)

    @property
    def identity(self) -> dict[str, str]:
        """Safe identity copied into the internal C4/C5 audit only."""
        return {
            "adapter_version": TERRA_TEST_ADAPTER_VERSION,
            "environment": "test",
            "fixture_content_hash": self._fixture_hash,
            "model_name": TERRA_TEST_MODEL_NAME,
            "model_version": self._items[0]["model_version"] if self._items else "",
        }

    def bind_context(self, context: Any) -> dict[str, str]:
        """Refuse a fixture whose records do not exactly name real frame artifacts."""
        artifacts = {str(item.frame_id): item for item in context.artifacts.frames}
        state = {
            str(item.get("frame_id") or ""): item
            for item in context.state.get("frames") or ()
            if isinstance(item, dict) and str(item.get("frame_id") or "")
        }
        bound: dict[str, dict[str, Any]] = {}
        for item in self._items:
            frame_id = item["frame_id"]
            artifact, frame = artifacts.get(frame_id), state.get(frame_id)
            if artifact is None or frame is None:
                raise ValueError(f"Terra test fixture frame_id is not a materialized frame: {frame_id}")
            if int(artifact.timestamp_ms) != item["timestamp_ms"] or int(frame.get("timestamp_ms") or 0) != item[
                "timestamp_ms"
            ]:
                raise ValueError(f"Terra test fixture timestamp does not match frame: {frame_id}")
            if str(artifact.image_hash) != item["frame_content_hash"]:
                raise ValueError(f"Terra test fixture content hash does not match frame: {frame_id}")
            bound[frame_id] = item
        self._by_frame_id = bound
        return dict(self.identity)

    def analyze_for_frame(self, frame: dict[str, Any], _transcript_context: str) -> dict[str, Any]:
        """Return only a pre-bound record; this never calls a model or network."""
        frame_id = str(frame.get("frame_id") or "")
        item = self._by_frame_id.get(frame_id)
        if item is None:
            raise ValueError(f"Terra test fixture was not bound for frame_id: {frame_id}")
        return {
            "visual_summary": item["visual_summary"],
            "labels": list(item["labels"]),
            "themes": list(item["themes"]),
            "symbols": list(item["symbols"]),
            "confidence_score": item["confidence_score"],
            "narration_aligned": item["narration_aligned"],
            "model": item["model_name"],
            "model_version": item["model_version"],
            "bbox": list(item["bbox"]),
            "environment": item["environment"],
            "fixture_content_hash": self._fixture_hash,
            "frame_content_hash": item["frame_content_hash"],
            "adapter_version": TERRA_TEST_ADAPTER_VERSION,
        }

    @staticmethod
    def _validate_fixture(fixture: dict[str, Any]) -> list[dict[str, Any]]:
        required_root = {"schema_version", "environment", "model_name", "model_version", "frames"}
        if set(fixture) != required_root:
            raise ValueError("Terra test fixture root has an invalid schema")
        if fixture.get("schema_version") != TERRA_TEST_FIXTURE_SCHEMA:
            raise ValueError("Terra test fixture schema_version is unsupported")
        if fixture.get("environment") != "test":
            raise ValueError("Terra test fixture environment must be test")
        if fixture.get("model_name") != TERRA_TEST_MODEL_NAME:
            raise ValueError("Terra test fixture model_name must be terra-test-substitute")
        model_version = _text(fixture.get("model_version"), "model_version")
        raw_frames = fixture.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError("Terra test fixture frames must be a non-empty list")
        expected = {
            "frame_id",
            "timestamp_ms",
            "frame_content_hash",
            "bbox",
            "label",
            "labels",
            "visual_summary",
            "themes",
            "symbols",
            "confidence_score",
            "narration_aligned",
            "model_name",
            "model_version",
            "environment",
        }
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_frames:
            if not isinstance(raw, dict) or set(raw) != expected:
                raise ValueError("Terra test fixture frame has an invalid schema")
            frame_id = _text(raw.get("frame_id"), "frame_id")
            if frame_id in seen:
                raise ValueError(f"Terra test fixture has duplicate frame_id: {frame_id}")
            seen.add(frame_id)
            timestamp = raw.get("timestamp_ms")
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                raise ValueError("Terra test fixture timestamp_ms must be a non-negative integer")
            if raw.get("model_name") != TERRA_TEST_MODEL_NAME or raw.get("model_version") != model_version:
                raise ValueError("Terra test fixture frame model provenance does not match root")
            if raw.get("environment") != "test":
                raise ValueError("Terra test fixture frame environment must be test")
            labels = _string_list(raw.get("labels"), "labels", required=True)
            label = _text(raw.get("label"), "label")
            if label not in labels:
                raise ValueError("Terra test fixture label must be included in labels")
            narration_aligned = raw.get("narration_aligned")
            if not isinstance(narration_aligned, bool):
                raise ValueError("Terra test fixture narration_aligned must be boolean")
            records.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp,
                    "frame_content_hash": _text(raw.get("frame_content_hash"), "frame_content_hash"),
                    "bbox": _bbox(raw.get("bbox")),
                    "label": label,
                    "labels": labels,
                    "visual_summary": _text(raw.get("visual_summary"), "visual_summary"),
                    "themes": _string_list(raw.get("themes"), "themes"),
                    "symbols": _string_list(raw.get("symbols"), "symbols"),
                    "confidence_score": _confidence(raw.get("confidence_score")),
                    "narration_aligned": narration_aligned,
                    "model_name": TERRA_TEST_MODEL_NAME,
                    "model_version": model_version,
                    "environment": "test",
                }
            )
        return sorted(records, key=lambda item: (item["timestamp_ms"], item["frame_id"]))


__all__ = [
    "TERRA_TEST_ADAPTER_VERSION",
    "TERRA_TEST_FIXTURE_SCHEMA",
    "TERRA_TEST_MODEL_NAME",
    "TerraTestVisionFixtureAnalyzer",
]

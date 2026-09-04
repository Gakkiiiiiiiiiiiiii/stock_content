"""Shared immutable DTOs used as seams while large stages are decomposed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ServiceInput:
    entity_id: str
    payload: Any
    schema_version: str


@dataclass(frozen=True, slots=True)
class ProjectionOutput:
    entity_id: str
    payload: Any
    schema_version: str
    content_hash: str


__all__ = ["ProjectionOutput", "ServiceInput"]

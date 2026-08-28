from __future__ import annotations

from .temporal_semantics import (
    ClaimTemporalBinding,
    ClaimTemporalRelation,
    temporal_binding_id_of,
    temporal_relation_id_of,
)


def canonical_bindings(bindings: list[ClaimTemporalBinding]) -> list[ClaimTemporalBinding]:
    return sorted(
        bindings,
        key=lambda item: (item.role.value, item.scope.value, item.temporal_binding_id or temporal_binding_id_of(item)),
    )


def canonical_relations(relations: list[ClaimTemporalRelation]) -> list[ClaimTemporalRelation]:
    return sorted(relations, key=lambda item: item.temporal_relation_id or temporal_relation_id_of(item))


def canonical_temporal_payload(bindings: list[ClaimTemporalBinding], relations: list[ClaimTemporalRelation]) -> dict:
    return {
        "bindings": [item.temporal_binding_id for item in canonical_bindings(bindings)],
        "relations": [item.temporal_relation_id for item in canonical_relations(relations)],
    }


binding_id_of = temporal_binding_id_of
relation_id_of = temporal_relation_id_of


__all__ = ["canonical_bindings", "canonical_relations", "canonical_temporal_payload"]

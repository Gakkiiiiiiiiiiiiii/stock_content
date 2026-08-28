from __future__ import annotations

from .temporal_semantics import ClaimTemporalBinding, TemporalRole, TemporalScope


def validate_temporal_binding(binding: ClaimTemporalBinding) -> ClaimTemporalBinding:
    # Revalidation is intentionally fail-closed so callers that receive a
    # model constructed by an adapter cannot bypass the domain invariants.
    validated = ClaimTemporalBinding.model_validate(binding.model_dump())
    if validated.role is TemporalRole.FORECAST_TARGET and validated.scope is not TemporalScope.FORECAST:
        raise ValueError("FORECAST_TARGET must use FORECAST scope")
    if validated.role is TemporalRole.CONDITION_PERIOD and validated.scope is not TemporalScope.INTERVAL:
        raise ValueError("CONDITION_PERIOD must use INTERVAL scope")
    if (
        validated.normalization_status.upper() in {"PARTIAL", "UNRESOLVED"}
        and validated.raw_expression
        and not validated.expression_key
    ):
        raise ValueError("partial or unresolved temporal binding requires expression_key")
    return validated


__all__ = ["validate_temporal_binding"]

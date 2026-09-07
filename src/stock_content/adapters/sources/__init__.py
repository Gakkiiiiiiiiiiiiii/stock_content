"""Source adapter exports, loaded lazily to keep browser ports acyclic."""
from __future__ import annotations

from typing import Any

_EXPORTS = {
    "BilibiliMaterializer": ("bilibili_materializer", "BilibiliMaterializer"),
    "BilibiliResolver": ("bilibili_resolver", "BilibiliResolver"),
    "BilibiliSourceAdapter": ("bilibili", "BilibiliSourceAdapter"),
    "XiaoeHlsResolver": ("xiaoe_page", "XiaoeHlsResolver"),
    "XiaoeHlsSourceAdapter": ("xiaoe", "XiaoeHlsSourceAdapter"),
    "XiaoeMaterializer": ("xiaoe_materializer", "XiaoeMaterializer"),
    "XiaoePageResolver": ("xiaoe_page", "XiaoePageResolver"),
    "XiaoePageSourceAdapter": ("xiaoe", "XiaoePageSourceAdapter"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{module_name}"), symbol)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)

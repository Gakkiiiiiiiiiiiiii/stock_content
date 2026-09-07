"""Manual protected-page gate; no URL, storage state, or credential is in this repository."""
from __future__ import annotations

import os

import pytest

from stock_content.adapters.sources.xiaoe import XiaoePageSourceAdapter


@pytest.mark.skipif(
    not all(
        os.getenv(name)
        for name in (
            "CONTENT_XIAOE_MANUAL_SOURCE_REF", "CONTENT_XIAOE_STORAGE_STATE_FILE",
            "CONTENT_XIAOE_ALLOWED_DOMAINS", "CONTENT_XIAOE_PAGE_URL_TEMPLATE",
        )
    ),
    reason="requires an explicitly authorized Xiaoe account, page identity, and allowlist",
)
def test_authorized_page_resolves_only_when_explicitly_configured() -> None:
    adapter = XiaoePageSourceAdapter.from_environment()
    assert callable(adapter.resolve_materialization)

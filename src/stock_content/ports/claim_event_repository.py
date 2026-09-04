from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from stock_content.domain.claim_state_event import ClaimStateEvent


class ClaimStateEventRepository(Protocol):
    def append(self, event: ClaimStateEvent) -> ClaimStateEvent: ...
    def append_in_session(self, session, event: ClaimStateEvent) -> ClaimStateEvent: ...
    def list_for_claim(self, claim_id: str) -> Iterable[ClaimStateEvent]: ...

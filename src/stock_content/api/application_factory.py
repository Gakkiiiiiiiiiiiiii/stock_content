"""Deferred application composition for API process startup.

Importing API modules must stay free of database, search, and model work.  The
default factory therefore imports the composition root only when FastAPI enters
its lifespan.  Tests can inject a deterministic factory instead.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from stock_content.application.service import ContentApplication


class ApplicationFactory(Protocol):
    """Build a fully verified application at process startup."""

    def create_application(self) -> ContentApplication: ...


class DefaultApplicationFactory:
    """Production composition root; database verification happens in its builder."""

    def create_application(self) -> ContentApplication:
        # Keep the dependency graph (and any optional model/search adapters) out
        # of module import and test collection.
        from stock_content.api.dependencies import build_application

        return build_application()


class StaticApplicationFactory:
    """Explicit application injection used by deterministic tests."""

    def __init__(self, application: ContentApplication) -> None:
        self._application = application

    def create_application(self) -> ContentApplication:
        return self._application


class CallableApplicationFactory:
    """Adapter for concise test factories."""

    def __init__(self, factory: Callable[[], ContentApplication]) -> None:
        self._factory = factory

    def create_application(self) -> ContentApplication:
        return self._factory()


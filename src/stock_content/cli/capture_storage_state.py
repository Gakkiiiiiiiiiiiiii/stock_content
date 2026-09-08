"""Capture an operator-owned Playwright state in an isolated browser."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from stock_content.adapters.browser.playwright_session import BrowserSessionError, save_visible_storage_state
from stock_content.adapters.sources.security import UnsafeSourceURL

_DOMAIN = re.compile(r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}\Z")


def _allowed_domain(value: str) -> str:
    domain = value.strip().lower().rstrip(".")
    if not _DOMAIN.fullmatch(domain):
        raise argparse.ArgumentTypeError("allowed domain must be one exact DNS name")
    return domain


def _bounded_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not 1 <= timeout <= 600:
        raise argparse.ArgumentTypeError("timeout must be between 1 and 600 seconds")
    return timeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save a private Playwright storage state after an operator login")
    parser.add_argument("--page-url", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--allowed-domain", required=True, action="append", type=_allowed_domain)
    parser.add_argument("--timeout-seconds", type=_bounded_timeout, default=300.0)
    return parser


def _wait_for_operator_confirmation(timeout_seconds: float, *, readline=None) -> bool:
    """Wait at most the login window for Enter; EOF and timeout fail closed."""
    print("READY_FOR_LOGIN", file=sys.stderr)
    lines: Queue[str] = Queue(maxsize=1)
    reader = readline or sys.stdin.readline
    Thread(target=lambda: lines.put(reader()), daemon=True).start()
    try:
        return bool(lines.get(timeout=timeout_seconds))
    except Empty:
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        save_visible_storage_state(
            page_url=args.page_url,
            destination=args.destination,
            allowed_domains=frozenset(args.allowed_domain),
            timeout_seconds=args.timeout_seconds,
            confirmation=_wait_for_operator_confirmation,
            require_operator_confirmation=True,
        )
    except UnsafeSourceURL as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except BrowserSessionError as exc:
        print(exc.code, file=sys.stderr)
        return 1
    print("STORAGE_STATE_SAVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

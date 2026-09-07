"""Fail CI artifact scanning without printing secret values or matching lines."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SKIP_PARTS = frozenset(
    {".git", ".pytest_cache", "__pycache__", ".venv", "data", "runtime", "media", "models", "model-cache"}
)
_PATTERNS = (
    re.compile(r"(?i)secret-(?:cookie|url|token)-canary"),
    re.compile(r"(?i)\b(?:authorization|cookie)\s*[:=]\s*(?!<redacted>)[^\s,]+"),
    re.compile(r"(?i)\b(?:token|signature|x-amz-signature|x-amz-credential)=[^&\s]+"),
    re.compile(r"(?i)\bbearer\s+(?!<redacted>)[A-Za-z0-9._~+/-]{8,}"),
)


def _files(roots: list[Path]):
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and not any(part.lower() in _SKIP_PARTS for part in path.parts):
                    yield path


def scan(roots: list[Path]) -> int:
    findings = 0
    scanned = 0
    for path in _files(roots):
        if path.stat().st_size > 10 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1
        if any(pattern.search(text) for pattern in _PATTERNS):
            findings += 1
    if findings:
        print(f"artifact secret scan failed: {findings} file(s) with redacted finding(s)", file=sys.stderr)
        return 1
    print(f"artifact secret scan passed: {scanned} file(s) scanned")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    return scan(parser.parse_args().roots)


if __name__ == "__main__":
    raise SystemExit(main())

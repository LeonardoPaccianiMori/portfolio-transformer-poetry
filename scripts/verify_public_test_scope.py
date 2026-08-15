#!/usr/bin/env python3
"""Verify that public CI excludes only the exact reviewed local-artifact tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "release/local_only_test_allowlist.txt"


def collected_local_tests() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "local_artifact"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }


def main() -> int:
    expected = {
        line.strip()
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    actual = collected_local_tests()
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    for node_id in missing:
        print(f"public-test-scope | ERROR | marker missing: {node_id}")
    for node_id in unexpected:
        print(f"public-test-scope | ERROR | unreviewed local-only test: {node_id}")
    if missing or unexpected:
        return 1
    print(f"public-test-scope | OK | local_only={len(actual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

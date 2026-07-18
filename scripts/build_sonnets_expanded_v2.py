#!/usr/bin/env python3
"""Build the versioned sonnets_expanded_v2 corpus from pinned sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_expansion_build import build_sonnets_expanded_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=ROOT / "data/metadata/poems_manifest.csv",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "data/metadata/wikisource_snapshots/ws_alfieri_rime_1912.json",
    )
    parser.add_argument("--request-delay", type=float, default=6.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"sonnet-build | {message}", flush=True)

    report = build_sonnets_expanded_v2(
        repo_root=ROOT,
        base_manifest_path=args.base_manifest,
        snapshot_path=args.snapshot,
        request_delay=args.request_delay,
        progress=progress,
    )
    print(
        "sonnet-build | complete "
        f"base={report['base_poem_count']} added={report['added_poem_count']} "
        f"total={report['total_poem_count']} splits={report['split_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

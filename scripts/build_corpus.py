#!/usr/bin/env python3
"""Build the local sonnet corpus from Wikisource sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.build import build_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated source keys or 'all'.",
    )
    parser.add_argument(
        "--dataset",
        choices=["core_pre_petrarch", "expanded_with_petrarch"],
        default="expanded_with_petrarch",
    )
    parser.add_argument("--force", action="store_true", help="Replace generated outputs.")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep data/raw and data/interim after a successful build.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between Wikisource requests.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_corpus(
        repo_root=ROOT,
        sources=args.sources,
        dataset=args.dataset,
        force=args.force,
        keep_temp=args.keep_temp,
        seed=args.seed,
        request_delay=args.request_delay,
        verbose=not args.quiet,
    )
    print(
        f"wrote {report['included_rows']} included rows "
        f"out of {report['manifest_rows']} manifest rows"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the V8 train rhyme lexicon for the plan-then-poem pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.rhyme_lexicon import (
    build_lexicon,
    write_lexicon,
)
from sonnet_training.form_targeted_data import load_train_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/metadata/rhyme_lexicon_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_train_rows(args.manifest)
    payload = build_lexicon(
        rows,
        ROOT,
        manifest_path=args.manifest,
        progress=lambda message: print(f"rhyme-lexicon | {message}", flush=True),
    )
    write_lexicon(args.output, payload)
    print(
        "rhyme-lexicon | complete "
        f"sonnets={payload['sonnet_count']} lines={payload['line_count']} "
        f"keys={payload['unique_keys']}",
        flush=True,
    )
    for entry in payload["entries"][:5]:
        print(f"  {entry['key']}: {entry['count']} {entry['words'][:4]}")
    print(f"wrote lexicon: {args.output}")


if __name__ == "__main__":
    main()

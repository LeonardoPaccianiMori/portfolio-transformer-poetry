#!/usr/bin/env python3
"""Write the v1/v4/v5 sonnet corpus scaling summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_corpus_scaling import (
    write_sonnet_corpus_scaling_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(
            "runs/pretraining_quality_swiglu_larger_200k_001/tokenizer.json"
        ),
    )
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "sonnet_corpus_scaling_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_sonnet_corpus_scaling_report(
        repo_root=ROOT,
        manifest_paths={
            "v1": Path("data/metadata/poems_manifest.csv"),
            "v4": Path("data/metadata/sonnets_expanded_v4_manifest.csv"),
            "v5": Path("data/metadata/sonnets_expanded_v5_manifest.csv"),
        },
        tokenizer_path=args.tokenizer_path,
        output_path=args.output,
        dataset=args.dataset,
        context_length=args.context_length,
        progress=lambda message: print(f"scaling-report | {message}", flush=True),
    )
    print(
        "scaling-report | complete "
        f"versions={len(report['versions'])} "
        f"parent_vocab_size={report['parent_vocab_size']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

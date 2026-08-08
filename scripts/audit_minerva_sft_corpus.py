#!/usr/bin/env python3
"""Audit the V5 sonnet texts before another Minerva fine-tuning recipe."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.minerva_sft_audit import audit_minerva_sft_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/metadata/sonnets_expanded_v5_manifest.csv"),
    )
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("reports/minerva_v5_sft_corpus_audit.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/minerva_v5_sft_corpus_audit.md"),
    )
    parser.add_argument(
        "--review-sample",
        type=Path,
        default=Path("reports/minerva_v5_training_text_review_sample.md"),
    )
    parser.add_argument("--review-sample-size", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()

    def progress(message: str) -> None:
        elapsed = round(time.monotonic() - started_at)
        print(f"minerva-corpus-audit | {message} | elapsed={elapsed}s", flush=True)

    print(
        "minerva-corpus-audit | start dataset={dataset} manifest={manifest}".format(
            dataset=args.dataset,
            manifest=args.manifest_path,
        ),
        flush=True,
    )
    report = audit_minerva_sft_corpus(
        repo_root=ROOT,
        manifest_path=args.manifest_path,
        dataset=args.dataset,
        json_report_path=args.json_report,
        markdown_report_path=args.markdown_report,
        review_sample_path=args.review_sample,
        review_sample_size=args.review_sample_size,
        progress=progress,
    )
    print(
        "minerva-corpus-audit | complete poems={poems} issues={issues} "
        "duplicates={duplicates} gate={gate}".format(
            poems=report["selected_poem_count"],
            issues=report["structural_issue_count"],
            duplicates=report["exact_normalized_duplicate_group_count"],
            gate=report["automated_structural_gate"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


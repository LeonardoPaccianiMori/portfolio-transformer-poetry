#!/usr/bin/env python3
"""Encode the four locked PAISA-historical rescue curriculum splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.paisa_historical_encoding import PaisaHistoricalEncodingConfig
from sonnet_corpus.paisa_historical_encoding import encode_paisa_historical_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-report-path",
        type=Path,
        default=ROOT / "reports/paisa_historical_rescue_v1_tokenizer_report.json",
    )
    parser.add_argument(
        "--curriculum-report-path",
        type=Path,
        default=ROOT / "reports/paisa_historical_rescue_v1_curriculum_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/paisa_historical_rescue_v1/encoded",
    )
    parser.add_argument(
        "--local-report-path",
        type=Path,
        default=(
            ROOT
            / "data/local/pretraining/paisa_historical_rescue_v1/encoded_report.json"
        ),
    )
    parser.add_argument(
        "--public-report-path",
        type=Path,
        default=ROOT / "reports/paisa_historical_rescue_v1_encoded_report.json",
    )
    parser.add_argument("--progress-interval-documents", type=int, default=5_000)
    parser.add_argument("--checkpoint-interval-documents", type=int, default=1_000)
    parser.add_argument("--pretoken-cache-entries", type=int, default=250_000)
    parser.add_argument("--max-documents-per-split-run", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = encode_paisa_historical_splits(
        PaisaHistoricalEncodingConfig(
            tokenizer_report_path=args.tokenizer_report_path,
            curriculum_report_path=args.curriculum_report_path,
            output_dir=args.output_dir,
            local_report_path=args.local_report_path,
            public_report_path=args.public_report_path,
            progress_interval_documents=args.progress_interval_documents,
            checkpoint_interval_documents=args.checkpoint_interval_documents,
            pretoken_cache_entries=args.pretoken_cache_entries,
            max_documents_per_split_run=args.max_documents_per_split_run,
        ),
        progress=lambda message: print(f"rescue-encode | {message}", flush=True),
    )
    print(
        "rescue-encode | status="
        f"{report['status']} completed_splits="
        f"{sum(split['status'] == 'complete' for split in report['splits'])}/4",
        flush=True,
    )
    if report["status"] == "complete":
        print(
            "rescue-encode | total tokens="
            f"{report['totals']['tokens']:,} "
            f"output={report['totals']['output_bytes'] / (1024 ** 3):.2f} GiB",
            flush=True,
        )
        print(
            f"rescue-encode | wrote public report: {args.public_report_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

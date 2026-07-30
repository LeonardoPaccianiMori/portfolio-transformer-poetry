#!/usr/bin/env python3
"""Build the local PAISÀ corpus and provenance inventory from its official release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.paisa_activation import PAISA_RELEASE_ARTIFACT_URL
from sonnet_corpus.paisa_build import PaisaBuildConfig, build_paisa_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", default="paisa_modern_italian_v1")
    parser.add_argument("--release-url", default=PAISA_RELEASE_ARTIFACT_URL)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/paisa/paisa_modern_italian_v1",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "reports/paisa_modern_italian_v1_build_report.json",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=ROOT / "data/interim/paisa_modern_italian_v1_build",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--split-salt", default="paisa_modern_italian_v1")
    parser.add_argument("--document-progress-interval", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("paisa-build | start local PAISÀ corpus build", flush=True)
    report = build_paisa_corpus(
        PaisaBuildConfig(
            corpus_version=args.corpus_version,
            release_url=args.release_url,
            processed_dir=args.processed_dir,
            report_path=args.report_path,
            temp_dir=args.temp_dir,
            validation_fraction=args.validation_fraction,
            split_salt=args.split_salt,
            document_progress_interval=args.document_progress_interval,
        ),
        progress=lambda message: print(f"paisa-build | {message}", flush=True),
    )
    counts = report["document_counts"]
    print(
        "paisa-build | complete "
        f"retained_documents={counts['retained']} "
        f"train_documents={counts['train']} "
        f"validation_documents={counts['validation']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

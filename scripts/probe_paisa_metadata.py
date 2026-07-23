#!/usr/bin/env python3
"""Inspect PAISÀ license and provenance metadata without downloading its corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.paisa_probe import PAISA_DESCRIPTION_URL, probe_paisa_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=PAISA_DESCRIPTION_URL)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/local/pretraining/paisa/paisa_metadata_probe.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("paisa-probe | start metadata-only audit", flush=True)
    report = probe_paisa_metadata(
        report_path=args.report,
        source_url=args.source_url,
        progress=lambda message: print(f"paisa-probe | {message}", flush=True),
    )
    result = report["result"]
    print(f"paisa-probe | wrote report: {args.report}", flush=True)
    print(
        "paisa-probe | complete "
        f"status={result['status']} documents={result['document_count']} "
        f"words={result['reported_word_count']} "
        "activation_status=auxiliary_experiment_not_activated",
        flush=True,
    )
    if result["status"] != "ok":
        print(f"paisa-probe | failed error={result['error']}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

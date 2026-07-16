#!/usr/bin/env python3
"""Probe one audit-only Liber Liber work without activating it for training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.liber_liber_probe import probe_liber_liber_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
        help="Broader prose source manifest CSV.",
    )
    parser.add_argument(
        "--source-id",
        default="ll_vico_principj_scienza_nuova",
        help="One audit-only Liber Liber source ID.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/local/pretraining/liber_liber/ll_vico_principj_scienza_nuova_probe.json",
        help="Local JSON inspection report path.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"probe | {message}", flush=True)

    report = probe_liber_liber_source(
        manifest_path=args.manifest,
        source_id=args.source_id,
        report_path=args.report,
        progress=progress,
    )
    result = report["result"]
    if not args.quiet:
        print(
            "probe | complete "
            f"status={result['status']} characters={result['cleaned_character_count']} "
            "activation_status=audit_then_include",
            flush=True,
        )
    if result["status"] != "ok":
        print(f"probe | failed error={result['error']}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

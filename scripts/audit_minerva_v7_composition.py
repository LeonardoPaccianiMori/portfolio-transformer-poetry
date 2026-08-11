#!/usr/bin/env python3
"""Count frozen V7 corpus roles with Minerva and run the composition gate."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.minerva_v7_composition import (
    MinervaV7CompositionConfig,
    build_minerva_v7_composition,
    write_composition_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    config = MinervaV7CompositionConfig(
        repo_root=ROOT,
        policy_path=ROOT / "data/metadata/minerva_7b_v7_composition_policy_v1.json",
        canonical_corpus_dir=ROOT / "data/processed/canonical_italian_corpora_v1",
        v7_manifest_path=ROOT / "data/metadata/sonnets_expanded_v7_manifest.csv",
        replay_text_path=ROOT / "data/local/minerva_7b_staged/replay_train.txt",
        replay_report_path=ROOT / "data/local/minerva_7b_staged/replay_sample_report.json",
        json_report_path=ROOT / "reports/minerva_7b_v7_token_counts_v1.json",
        markdown_report_path=ROOT / "reports/minerva_7b_v7_composition_gate_v1.md",
        tokenizer_cache_dir=ROOT / "data/local/minerva_qlora/huggingface",
        progress_interval=args.progress_interval,
    )

    def progress(completed: int, total: int, label: str) -> None:
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        remaining = (total - completed) / rate if rate else 0.0
        print(
            "minerva-v7-count | completed="
            f"{completed}/{total} ({completed / total:.1%}) label={label} "
            f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
            flush=True,
        )

    print(
        "minerva-v7-count | start device=cpu logical_units=26934 "
        f"progress_interval={args.progress_interval} estimated_runtime=25m-40m",
        flush=True,
    )
    report = build_minerva_v7_composition(config, progress=progress)
    write_composition_reports(
        report, config.json_report_path, config.markdown_report_path
    )
    print(
        "minerva-v7-count | complete status="
        f"{report['status']} tokens={report['totals']['training_tokens']:,} "
        f"elapsed={time.monotonic() - started:.1f}s "
        f"report={config.json_report_path.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

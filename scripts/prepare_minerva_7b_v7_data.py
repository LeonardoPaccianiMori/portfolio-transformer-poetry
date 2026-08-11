#!/usr/bin/env python3
"""Build and independently reproduce the frozen Minerva V7 token pools."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_data import (
    MinervaV7DataConfig,
    prepare_and_verify_minerva_v7_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents-per-pool-run",
        type=int,
        default=None,
        help="Testing/resume control; omit to complete every pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    config = MinervaV7DataConfig(
        repo_root=ROOT,
        policy_path=ROOT
        / "data/metadata/minerva_7b_v7_training_data_policy_v1.json",
        composition_policy_path=ROOT
        / "data/metadata/minerva_7b_v7_composition_policy_v1.json",
        composition_report_path=ROOT / "reports/minerva_7b_v7_token_counts_v1.json",
        canonical_corpus_dir=ROOT / "data/processed/canonical_italian_corpora_v1",
        v7_manifest_path=ROOT / "data/metadata/sonnets_expanded_v7_manifest.csv",
        replay_text_path=ROOT / "data/local/minerva_7b_staged/replay_train.txt",
        replay_report_path=ROOT
        / "data/local/minerva_7b_staged/replay_sample_report.json",
        tokenizer_cache_dir=ROOT / "data/local/minerva_qlora/huggingface",
        output_dir=ROOT / "data/local/minerva_7b_v7/encoded",
        reproduction_output_dir=ROOT
        / "data/local/minerva_7b_v7/encoded_reproduction",
        broader_split_manifest_path=ROOT
        / "data/metadata/minerva_7b_v7_broader_splits_v1.csv",
        json_report_path=ROOT / "reports/minerva_7b_v7_encoded_data_v1.json",
        markdown_report_path=ROOT / "reports/minerva_7b_v7_encoded_data_v1.md",
        max_documents_per_pool_run=args.max_documents_per_pool_run,
    )
    started = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-v7-data | {message} | "
            f"elapsed={_format_duration(time.monotonic() - started)}",
            flush=True,
        )

    print(
        "minerva-v7-data | start device=cpu independent_builds=2 pools_per_build=10 "
        "context=2048 source_span=2049 shard_target_tokens=8,388,608 "
        "estimated_runtime=20m-35m resumable=true",
        flush=True,
    )
    report = prepare_and_verify_minerva_v7_data(config, progress=progress)
    if report["status"] != "active_verified":
        print(
            "minerva-v7-data | paused at a completed-document checkpoint; "
            "rerun the same command to resume",
            flush=True,
        )
        return
    print(
        "minerva-v7-data | complete status={status} documents={documents:,} "
        "tokens={tokens:,} shards={shards:,} elapsed={elapsed} report={report}".format(
            status=report["status"],
            documents=report["totals"]["documents"],
            tokens=report["totals"]["tokens"],
            shards=report["totals"]["shards"],
            elapsed=_format_duration(time.monotonic() - started),
            report=config.json_report_path.relative_to(ROOT),
        ),
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()

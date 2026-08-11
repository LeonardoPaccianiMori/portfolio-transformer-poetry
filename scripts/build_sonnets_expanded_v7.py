#!/usr/bin/env python3
"""Build the checkpoint-8A V7 sonnet split and identity freeze."""

from __future__ import annotations

import sys
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_v7_split import V7SplitConfig, build_v7_sonnet_split


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    started = monotonic()
    print(
        "sonnet-v7-split | start device=cpu total_units=22693 progress_interval=1000 "
        "target_new_split=90/5/5 activation=sonnet_split_only tokens=0 gpu=0 "
        "estimated_runtime=5-20s",
        flush=True,
    )

    def progress(message: str) -> None:
        print(
            f"sonnet-v7-split | {message} elapsed={_duration(monotonic() - started)}",
            flush=True,
        )

    metadata = ROOT / "data/metadata"
    report = build_v7_sonnet_split(
        V7SplitConfig(
            repo_root=ROOT,
            canonical_sonnet_manifest_path=(
                ROOT / "data/processed/canonical_italian_corpora_v1/sonnets_manifest.csv"
            ),
            v6_manifest_path=metadata / "sonnets_expanded_v6_manifest.csv",
            author_group_path=metadata / "sonnets_expanded_v7_author_groups_v1.csv",
            v7_manifest_path=metadata / "sonnets_expanded_v7_manifest.csv",
            json_report_path=ROOT / "reports/sonnets_expanded_v7_split_v1.json",
            markdown_report_path=ROOT / "reports/sonnets_expanded_v7_split_v1.md",
        ),
        progress=progress,
    )
    print(
        "sonnet-v7-split | complete "
        f"splits={report['v7_split_counts']} identity={report['v7_identity_sha256']} "
        f"elapsed={_duration(monotonic() - started)} text_copied=0 tokens=0 mixtures=0 gpu=0",
        flush=True,
    )


if __name__ == "__main__":
    main()

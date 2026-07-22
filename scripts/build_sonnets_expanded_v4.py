#!/usr/bin/env python3
"""Build sonnets_expanded_v4 from v3 plus pinned Varchi sonnets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_expansion_build import build_sonnets_expanded


def main() -> None:
    def progress(message: str) -> None:
        print(f"sonnet-build | {message}", flush=True)

    report = build_sonnets_expanded(
        repo_root=ROOT,
        base_manifest_path=ROOT / "data/metadata/sonnets_expanded_v3_manifest.csv",
        snapshot_path=ROOT / "data/metadata/wikisource_snapshots/ws_varchi_infermita.json",
        output_dataset_id="sonnets_expanded_v4",
        request_delay=6.0,
        progress=progress,
    )
    print(
        "sonnet-build | complete "
        f"base={report['base_poem_count']} added={report['added_poem_count']} "
        f"total={report['total_poem_count']} splits={report['split_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

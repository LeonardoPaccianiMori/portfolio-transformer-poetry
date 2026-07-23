#!/usr/bin/env python3
"""Build sonnets_expanded_v5 from v4 plus all approved expansion sources."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_expansion_build import (
    build_sonnets_expanded_from_snapshots,
)


SOURCE_IDS = (
    "ws_andreini_rime_1601",
    "ws_colonna_rime_1760",
    "ws_stampa_rime_1913",
    "ws_ariosto_rime_varie_1857",
    "ws_sannazaro_rime_disperse",
)
EXPECTED_BASE_COUNT = 1011
EXPECTED_ADDED_COUNT = 864


def main() -> None:
    def progress(message: str) -> None:
        print(f"sonnet-build | {message}", flush=True)

    snapshot_dir = ROOT / "data/metadata/wikisource_snapshots"
    report = build_sonnets_expanded_from_snapshots(
        repo_root=ROOT,
        base_manifest_path=(
            ROOT / "data/metadata/sonnets_expanded_v4_manifest.csv"
        ),
        snapshot_paths=[
            snapshot_dir / f"{source_id}.json" for source_id in SOURCE_IDS
        ],
        output_dataset_id="sonnets_expanded_v5",
        request_delay=6.0,
        progress=progress,
    )
    if report["base_poem_count"] != EXPECTED_BASE_COUNT:
        raise ValueError(
            "v5 base count changed: "
            f"expected {EXPECTED_BASE_COUNT}, got {report['base_poem_count']}"
        )
    if report["added_poem_count"] != EXPECTED_ADDED_COUNT:
        raise ValueError(
            "v5 addition count changed: "
            f"expected {EXPECTED_ADDED_COUNT}, got {report['added_poem_count']}"
        )
    print(
        "sonnet-build | complete "
        f"base={report['base_poem_count']} "
        f"added={report['added_poem_count']} "
        f"total={report['total_poem_count']} "
        f"splits={report['split_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

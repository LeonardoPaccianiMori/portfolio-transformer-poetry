#!/usr/bin/env python3
"""Create committed snapshots for the approved historical Wikisource sources."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_wikisource_snapshot import (
    PretrainingWikisourceSnapshotSelection,
    create_pretraining_wikisource_snapshots,
)


SELECTIONS = (
    PretrainingWikisourceSnapshotSelection(
        source_id="ws_sarpi_istoria_concilio",
        scope="recursive_leaf_pages",
        excluded_page_titles=(
            "Istoria del Concilio tridentino/Indice del primo volume",
            "Istoria del Concilio tridentino/Indice del secondo volume",
            "Istoria del Concilio tridentino/Indice del terzo volume",
        ),
    ),
    PretrainingWikisourceSnapshotSelection(
        source_id="ws_verri_storia_milano",
        scope="explicit_subpages",
    ),
    PretrainingWikisourceSnapshotSelection(
        source_id="ws_verri_osservazioni_tortura",
        scope="explicit_subpages",
    ),
)


def main() -> None:
    snapshots = create_pretraining_wikisource_snapshots(
        manifest_path=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
        audit_report_path=(
            ROOT
            / "data/local/pretraining/wikisource/"
            "historical_core_corrected_hierarchies_probe.json"
        ),
        snapshot_dir=ROOT / "data/metadata/wikisource_snapshots",
        selections=SELECTIONS,
    )
    for snapshot in snapshots:
        print(
            "snapshot | wrote "
            f"{snapshot['source_id']} pages={len(snapshot['page_revisions'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()

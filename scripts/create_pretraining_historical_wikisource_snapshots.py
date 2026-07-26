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
            "Istoria del Concilio tridentino/Indice dei nomi/A",
            "Istoria del Concilio tridentino/Indice dei nomi/B",
            "Istoria del Concilio tridentino/Indice dei nomi/C",
            "Istoria del Concilio tridentino/Indice dei nomi/D",
            "Istoria del Concilio tridentino/Indice dei nomi/E",
            "Istoria del Concilio tridentino/Indice dei nomi/F",
            "Istoria del Concilio tridentino/Indice dei nomi/G",
            "Istoria del Concilio tridentino/Indice dei nomi/HIK",
            "Istoria del Concilio tridentino/Indice dei nomi/L",
            "Istoria del Concilio tridentino/Indice dei nomi/M",
            "Istoria del Concilio tridentino/Indice dei nomi/N",
            "Istoria del Concilio tridentino/Indice dei nomi/O",
            "Istoria del Concilio tridentino/Indice dei nomi/PQ",
            "Istoria del Concilio tridentino/Indice dei nomi/R",
            "Istoria del Concilio tridentino/Indice dei nomi/S",
            "Istoria del Concilio tridentino/Indice dei nomi/T",
            "Istoria del Concilio tridentino/Indice dei nomi/UVWYZ",
        ),
    ),
    PretrainingWikisourceSnapshotSelection(
        source_id="ws_verri_storia_milano",
        scope="explicit_subpages",
        excluded_page_titles=("Storia di Milano/Avvertimento",),
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

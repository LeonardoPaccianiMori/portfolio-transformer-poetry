#!/usr/bin/env python3
"""Create the committed Varchi snapshot from the reviewed local audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_expansion_build import create_sonnet_source_snapshot


def main() -> None:
    snapshot_path = ROOT / "data/metadata/wikisource_snapshots/ws_varchi_infermita.json"
    snapshot = create_sonnet_source_snapshot(
        audit_report_path=ROOT / "data/local/sonnet_audits/ws_varchi_infermita_probe.json",
        snapshot_path=snapshot_path,
        source_id="ws_varchi_infermita",
    )
    print(
        f"snapshot | wrote {snapshot_path} with {len(snapshot['page_revisions'])} eligible revisions",
        flush=True,
    )


if __name__ == "__main__":
    main()

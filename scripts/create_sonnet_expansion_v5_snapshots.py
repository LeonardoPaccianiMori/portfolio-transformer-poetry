#!/usr/bin/env python3
"""Create committed revision snapshots for every approved v5 source."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_expansion_build import create_sonnet_source_snapshot


SOURCE_IDS = (
    "ws_andreini_rime_1601",
    "ws_colonna_rime_1760",
    "ws_stampa_rime_1913",
    "ws_ariosto_rime_varie_1857",
    "ws_sannazaro_rime_disperse",
)
EXPECTED_CANDIDATE_COUNT = 864


def main() -> None:
    snapshot_dir = ROOT / "data/metadata/wikisource_snapshots"
    candidate_count = 0
    for index, source_id in enumerate(SOURCE_IDS, start=1):
        audit_path = (
            ROOT / "data/local/sonnet_audits" / f"{source_id}_probe.json"
        )
        snapshot_path = snapshot_dir / f"{source_id}.json"
        print(
            f"snapshot | source {index}/{len(SOURCE_IDS)}: {source_id}",
            flush=True,
        )
        payload = create_sonnet_source_snapshot(
            audit_report_path=audit_path,
            snapshot_path=snapshot_path,
            source_id=source_id,
        )
        source_candidate_count = len(payload["eligible_candidates"])
        candidate_count += source_candidate_count
        print(
            "snapshot | wrote "
            f"{snapshot_path.relative_to(ROOT)} "
            f"pages={len(payload['page_revisions'])} "
            f"sonnets={source_candidate_count}",
            flush=True,
        )

    if candidate_count != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            "approved snapshot candidate total changed: "
            f"expected {EXPECTED_CANDIDATE_COUNT}, got {candidate_count}"
        )
    print(
        f"snapshot | complete sources={len(SOURCE_IDS)} sonnets={candidate_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a committed source snapshot from a reviewed local sonnet audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_expansion_build import create_sonnet_source_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=ROOT / "data/local/sonnet_audits/ws_alfieri_rime_1912_probe.json",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "data/metadata/wikisource_snapshots/ws_alfieri_rime_1912.json",
    )
    parser.add_argument("--source-id", default="ws_alfieri_rime_1912")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = create_sonnet_source_snapshot(
        audit_report_path=args.audit_report,
        snapshot_path=args.snapshot,
        source_id=args.source_id,
    )
    print(
        f"snapshot | wrote {args.snapshot} with {len(snapshot['page_revisions'])} eligible revisions",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build checkpoint 6A's metadata-only archive registry resolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonnet_corpus.archive_registry_resolution import (  # noqa: E402
    ArchiveRegistryResolutionConfig,
    build_archive_registry_resolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--request-delay", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    config = ArchiveRegistryResolutionConfig(
        repo_root=root,
        registry_path=root / "data/metadata/corpus_archive_expansion_registry.csv",
        cache_dir=root / "data/local/archive_registry_resolution_v1",
        evidence_path=root / "data/metadata/corpus_archive_terms_evidence_v1.csv",
        resolution_path=root / "data/metadata/corpus_archive_resolution_v1.csv",
        composition_gate_path=root / "data/metadata/corpus_archive_composition_gate_v1.csv",
        json_report_path=root / "reports/corpus_archive_registry_resolution_v1.json",
        markdown_report_path=root / "reports/corpus_archive_registry_resolution_v1.md",
        request_timeout_seconds=args.request_timeout,
        request_delay_seconds=args.request_delay,
    )
    started = monotonic()
    print("job=archive_registry_resolution device=cpu total_evidence=18 progress_interval=1 start", flush=True)
    report = build_archive_registry_resolution(config, progress=lambda message: print(message, flush=True))
    elapsed = monotonic() - started
    print(
        "job=archive_registry_resolution complete "
        f"archives={report['archive_count']} evidence={report['evidence_count']} "
        f"eligible={report['eligible_bounded_inventory_count']} activated_characters=0 "
        f"elapsed={elapsed:.1f}s report={config.json_report_path}",
        flush=True,
    )
    print(json.dumps(report["artifact_sha256"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

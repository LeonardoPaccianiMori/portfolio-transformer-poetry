#!/usr/bin/env python3
"""Run checkpoint 6C's final metadata-only archive-discovery pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.archive_discovery import (  # noqa: E402
    ArchiveDiscoveryConfig,
    QUERY_SPECS,
    build_archive_discovery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/archive_discovery_v1",
    )
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ArchiveDiscoveryConfig:
    metadata = ROOT / "data/metadata"
    return ArchiveDiscoveryConfig(
        repo_root=ROOT,
        registry_path=metadata / "corpus_archive_expansion_registry.csv",
        cache_dir=args.cache_dir,
        query_path=metadata / "corpus_archive_discovery_queries_v1.csv",
        evidence_path=metadata / "corpus_archive_discovery_evidence_v1.csv",
        decision_path=metadata / "corpus_archive_discovery_decisions_v1.csv",
        json_report_path=ROOT / "reports/corpus_archive_discovery_v1.json",
        markdown_report_path=ROOT / "reports/corpus_archive_discovery_v1.md",
        request_timeout_seconds=args.request_timeout,
        request_delay_seconds=args.request_delay,
        max_attempts=args.max_attempts,
    )


def main() -> int:
    args = parse_args()
    started = monotonic()
    print(
        "archive-discovery | start device=cpu "
        f"total_queries={len(QUERY_SPECS)} progress_interval=1_request "
        "activation=false corpus_text=false estimated_runtime=2-8m_network_or_under_10s_cached",
        flush=True,
    )
    report = build_archive_discovery(
        config_from_args(args),
        progress=lambda message: print(f"archive-discovery | {message}", flush=True),
    )
    print(
        "archive-discovery | complete "
        f"queries={report['query_count']} surfaces={report['surface_count']} "
        f"decisions={report['candidate_decision_count']} "
        f"eligible={report['eligible_standard_audit_count']} "
        f"registry_additions={report['registry_addition_count']} activated=0 "
        f"elapsed={monotonic() - started:.1f}s "
        "next=checkpoint_6D",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

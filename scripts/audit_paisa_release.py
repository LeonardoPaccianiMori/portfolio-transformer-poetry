#!/usr/bin/env python3
"""Audit PAISÀ release access, provenance, and local storage without downloading text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.paisa_activation import (
    PAISA_RELEASE_URL,
    PaisaActivationAuditConfig,
    audit_paisa_release,
)
from sonnet_corpus.paisa_probe import PAISA_DESCRIPTION_URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description-url", default=PAISA_DESCRIPTION_URL)
    parser.add_argument("--release-url", default=PAISA_RELEASE_URL)
    parser.add_argument(
        "--acquisition-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/paisa",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "reports/paisa_release_activation_audit.json",
    )
    parser.add_argument("--minimum-free-space-multiplier", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("paisa-activation | start metadata-only release activation audit", flush=True)
    report = audit_paisa_release(
        PaisaActivationAuditConfig(
            report_path=args.report_path,
            description_url=args.description_url,
            release_url=args.release_url,
            acquisition_dir=args.acquisition_dir,
            minimum_free_space_multiplier=args.minimum_free_space_multiplier,
        ),
        progress=lambda message: print(f"paisa-activation | {message}", flush=True),
    )
    route = report["release_route"]
    storage = report["storage_preflight"]
    print(
        "paisa-activation | complete "
        f"activation_status={report['activation_status']} "
        f"release_status={route['status']} storage_status={storage['status']}",
        flush=True,
    )
    print(f"paisa-activation | next action: {report['next_action']}", flush=True)


if __name__ == "__main__":
    main()

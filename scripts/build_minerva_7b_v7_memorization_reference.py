#!/usr/bin/env python3
"""Build the private, hash-pinned V7 sonnet-train memorization reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_memorization import build_sonnet_train_reference
from sonnet_analysis.minerva_v7_runtime import load_research_config, load_verified_state, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--tokenizer-state-id", default="untouched_parent")
    parser.add_argument("--encoded-report", type=Path, default=Path("reports/minerva_7b_v7_encoded_data_v1.json"))
    parser.add_argument("--research-config", type=Path, default=Path("configs/minerva_7b_v7_research.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/memorization_reference"))
    args = parser.parse_args()
    config = load_research_config(args.research_config)
    if sha256_file(args.encoded_report) != config["encoded_data_report_sha256"]:
        raise ValueError("encoded data report hash mismatch")
    state = load_verified_state(args.state_audit, args.tokenizer_state_id)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(state["model_dir"], local_files_only=True)
    print(
        "minerva-v7-research | start job=memorization_reference "
        "records=19899 source=sonnets_train v7_test_accessed=False",
        flush=True,
    )
    manifest = build_sonnet_train_reference(
        encoded_report_path=args.encoded_report,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        repo_root=ROOT,
        progress=lambda message: print(f"minerva-v7-research | {message}", flush=True),
    )
    print(
        f"minerva-v7-research | complete records={manifest['record_count']} "
        f"records_bytes={manifest['records_bytes']} output={args.output_dir / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

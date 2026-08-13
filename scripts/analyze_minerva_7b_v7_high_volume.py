#!/usr/bin/env python3
"""Verify and summarize all seven high-volume V7 generation grids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_high_volume_analysis import (
    analyze_high_volume_outputs, blinded_review_markdown,
    build_high_volume_blinded_sample,
)
from sonnet_analysis.minerva_v7_high_volume_generation import load_high_volume_config
from sonnet_analysis.minerva_v7_registry import MODEL_STATES
from sonnet_analysis.minerva_v7_runtime import load_verified_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/high_volume_generation"))
    parser.add_argument("--config", type=Path, default=Path("configs/minerva_7b_v7_high_volume_generation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_high_volume_config(args.config)
    verified = {
        state.state_id: load_verified_state(args.state_audit, state.state_id)
        for state in MODEL_STATES
    }
    print("minerva-v7-research | start job=high_volume_analysis states=7 outputs=20160", flush=True)
    report = analyze_high_volume_outputs(
        state_directories={state_id: args.generation_root / state_id for state_id in verified},
        expected_state_identities={state_id: str(row["state_identity_sha256"]) for state_id, row in verified.items()},
        bootstrap_resamples=int(config["bootstrap"]["resamples"]),
        bootstrap_seed=int(config["bootstrap"]["seed"]),
        confidence_level=float(config["bootstrap"]["confidence_level"]),
    )
    blind = build_high_volume_blinded_sample(
        analysis_report=report,
        selected_prompt_count=int(config["blinded_review"]["prompts"]),
        selection_seed=int(config["blinded_review"]["selection_seed"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "high_volume_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "high_volume_blind_mapping.private.json").write_text(
        json.dumps(blind, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "high_volume_blinded_review.md").write_text(
        blinded_review_markdown(blind), encoding="utf-8"
    )
    print(
        f"minerva-v7-research | complete outputs={len(report['rows'])} "
        f"summaries={len(report['summaries'])} blinded_rows={blind['sample_rows']} "
        f"output={args.output_dir}", flush=True,
    )


if __name__ == "__main__":
    main()

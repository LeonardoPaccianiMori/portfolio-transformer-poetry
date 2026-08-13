#!/usr/bin/env python3
"""Create a bounded, resumable seven-state GPU extraction plan; run no model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_gpu_plan import build_gpu_extraction_plan
from sonnet_analysis.minerva_v7_runtime import load_research_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, default=Path("data/local/minerva_7b_v7/activation_probes_v1.json"))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/gpu_extraction"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--research-config", type=Path, default=Path("configs/minerva_7b_v7_research.json"))
    args = parser.parse_args()
    research = load_research_config(args.research_config)
    print("minerva-v7-analysis | start job=gpu_extraction_plan states=7 probes=48", flush=True)
    audit = json.loads(args.state_audit.read_text(encoding="utf-8"))
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    plan = build_gpu_extraction_plan(
        probe_manifest_path=args.probe_manifest,
        state_audit=audit,
        output_root=args.output_root,
        model_config=model_config,
        expected_probe_sha256=research["probe_manifest_sha256"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"minerva-v7-analysis | complete ready_states={plan['ready_state_count']}/7 "
        f"estimated_all_states_gib={plan['estimates']['all_seven_states_bytes'] / 1024**3:.2f} "
        f"causal_authorized={plan['causal_experiments_authorized']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

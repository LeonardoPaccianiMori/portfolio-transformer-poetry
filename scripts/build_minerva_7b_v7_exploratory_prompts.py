#!/usr/bin/env python3
"""Build the public 120-opening V7 exploratory prompt manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_exploratory_prompts import build_exploratory_prompt_manifest
from sonnet_analysis.minerva_v7_runtime import load_verified_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--encoded-report", type=Path, default=Path("reports/minerva_7b_v7_encoded_data_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/minerva_7b_v7_exploratory_prompts.json"))
    args = parser.parse_args()
    state = load_verified_state(args.state_audit, "untouched_parent")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(state["model_dir"], local_files_only=True)
    print("minerva-v7-research | start job=exploratory_prompt_build candidates=1247 target=120", flush=True)
    manifest = build_exploratory_prompt_manifest(
        encoded_report_path=args.encoded_report, tokenizer=tokenizer, repo_root=ROOT
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"minerva-v7-research | complete prompts={manifest['prompt_count']} "
        f"periods={len(manifest['period_counts'])} authors={manifest['unique_authors']} "
        f"works={manifest['unique_works']} output={args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
